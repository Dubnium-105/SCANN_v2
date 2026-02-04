import os
import argparse
import time
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import average_precision_score

# -------------------------
#  1. Dataset (含强力增强)
# -------------------------

class TripletPNGDset(Dataset):
    def __init__(self, root_dir, split, indices, resize=224,
                 channel_order=(0,1,2),
                 # 使用你算出的 Mean/Std
                 mean=(0.26366636987971404, 0.28187216168254836, 0.28376364009886634),
                 std=(0.08900734308916412, 0.12256077701380531, 0.12825460877519965)):
        
        self.root_dir = root_dir
        self.split = split
        self.channel_order = channel_order
        self.resize = resize

        # 收集文件
        self.samples = []
        for label_name, y in [("negative", 0), ("positive", 1)]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(folder, fn), y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No png found under: {root_dir}")

        # 只取 subset
        self.samples = [self.samples[i] for i in indices]

        # 预处理：Resize + ToTensor (先不做 Normalize，拼合后增强再做)
        self.part_tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(), # [1,H,W] range [0,1]
        ])

        self.normalize3 = transforms.Normalize(list(mean), list(std))

    def __len__(self):
        return len(self.samples)

    def _read_triplet(self, path):
        im = Image.open(path).convert("L")
        w, h = im.size
        if w < 240 or h < 80:
            # 兼容处理：如果小于标准尺寸，抛错或Resize(这里抛错)
            raise ValueError(f"Bad size {w}x{h} for {path}")

        # 切分左中右
        parts = [
            im.crop((0,   0,  80, 80)),
            im.crop((80,  0, 160, 80)),
            im.crop((160, 0, 240, 80)),
        ]
        # 按通道顺序重排
        parts = [parts[i] for i in self.channel_order]
        return parts

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        parts = self._read_triplet(path)

        # 1. 基础转换
        ts = []
        for p in parts:
            t = self.part_tf(p)
            ts.append(t)
        x = torch.cat(ts, dim=0)  # [3, 224, 224]

        # 2. 强力增强 (仅训练集)
        # 注意：必须对 3 个通道同时做同样的几何变换
        if self.split == "train":
            # 水平翻转
            if random.random() < 0.5:
                x = TF.hflip(x)
            # 垂直翻转
            if random.random() < 0.5:
                x = TF.vflip(x)
            # 随机旋转 90/180/270 (天文图旋转不变性)
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        # 3. 归一化
        x = self.normalize3(x)

        return x, torch.tensor(y, dtype=torch.long)


# -------------------------
#  2. Focal Loss
# -------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)

        t = targets.view(-1, 1)
        logp_t = logp.gather(1, t).squeeze(1)
        p_t = p.gather(1, t).squeeze(1)

        loss = -(1 - p_t) ** self.gamma * logp_t

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            a_t = self.alpha.gather(0, targets)
            loss = a_t * loss

        return loss.mean()


# -------------------------
#  3. Metrics
# -------------------------

def confusion_from_threshold(probs, labels, t):
    preds = (probs >= t).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return tn, fp, fn, tp

def prf_fbeta(tn, fp, fn, tp, beta=2.0):
    eps = 1e-12
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f2 = (1 + beta**2) * p * r / (beta**2 * p + r + eps)
    return p, r, f2

def find_threshold_target_recall(probs, labels, target_recall=0.98):
    # 寻找满足 recall >= target 的最大阈值
    ts = np.unique(probs)
    ts = np.sort(ts)[::-1]
    best = {"t": 0.0, "p": 0.0, "r": 0.0, "f2": 0.0, "cm": (0,0,0,0)}
    
    for t in ts:
        tn, fp, fn, tp = confusion_from_threshold(probs, labels, t)
        p, r, f2 = prf_fbeta(tn, fp, fn, tp)
        if r + 1e-12 >= target_recall:
            best = {"t": float(t), "p": p, "r": r, "f2": f2, "cm": (tn, fp, fn, tp)}
            break
    return best

@torch.no_grad()
def evaluate(model, loader, device, target_recall=0.98):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    pr_auc = float(average_precision_score(labels, probs))
    tR = find_threshold_target_recall(probs, labels, target_recall)

    return {"pr_auc": pr_auc, "tR": tR}


# -------------------------
#  4. Main Training Loop
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="root folder")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    
    # 因为用了 Sampler，这里默认权重改为 1.0 (平衡了)
    ap.add_argument("--pos_weight", type=float, default=1.0, help="Class weight in Loss")
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    ap.add_argument("--target_recall", type=float, default=0.98)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    
    args = ap.parse_args()

    # 固定种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 扫描文件
    tmp_samples = []
    for label_name, y in [("negative", 0), ("positive", 1)]:
        folder = os.path.join(args.data, label_name)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".png"):
                tmp_samples.append((os.path.join(folder, fn), y))

    # 2. 划分 Train / Val
    n = len(tmp_samples)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int((1.0 - args.val_ratio) * n)
    train_idx = idx[:split].tolist()
    val_idx = idx[split:].tolist()

    train_set = TripletPNGDset(args.data, "train", train_idx, resize=224)
    val_set   = TripletPNGDset(args.data, "val",   val_idx,   resize=224)

    # 3. 核心优化：WeightedRandomSampler
    # 计算训练集中各类别的样本数
    train_labels = [tmp_samples[i][1] for i in train_idx]
    count_neg = train_labels.count(0)
    count_pos = train_labels.count(1)
    print(f"Train Set: Neg={count_neg}, Pos={count_pos}")
    print(f"Val Set  : {len(val_idx)}")

    # 计算采样权重：数量少的样本权重高
    weight_class = [1.0 / count_neg, 1.0 / count_pos]
    samples_weight = [weight_class[y] for y in train_labels]
    samples_weight = torch.tensor(samples_weight, dtype=torch.double)

    sampler = WeightedRandomSampler(samples_weight, num_samples=len(train_set), replacement=True)

    # Loader (注意: 用了 sampler 就不能 shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False,   num_workers=2, pin_memory=True)

    # 4. Model & Loss
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # 注意：Sampler 已经平衡了 Batch 内的数量，pos_weight 设为 1.0~1.5 即可，不要再设 10 了
    print(f"Using FocalLoss(gamma={args.gamma}, alpha=[1.0, {args.pos_weight}]) with WeightedSampler")
    criterion = FocalLoss(gamma=args.gamma, alpha=[1.0, args.pos_weight]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3) # 增加一点 weight_decay 防止过拟合

    best = {"recall": -1, "f2": -1, "auc": -1, "epoch": -1, "path": "best_model.pth"}

    # 5. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            seen += x.size(0)
        
        train_loss = total_loss / max(seen, 1)
        
        # Eval
        m = evaluate(model, val_loader, device, target_recall=args.target_recall)
        tR = m["tR"]
        
        print(f"Ep {epoch}/{args.epochs-1} | Loss:{train_loss:.4f} | "
              f"Val AUC:{m['pr_auc']:.4f} | "
              f"@Recall~{args.target_recall*100:.0f}% -> T={tR['t']:.3f} P={tR['p']:.4f} R={tR['r']:.4f} F2={tR['f2']:.4f}")
        
        # 简单保存策略：优先保 Recall >= 目标，然后看 F2
        improved = False
        # 如果这是第一个满足 Recall 目标的 epoch
        if tR['r'] >= args.target_recall - 0.001:
            if best['recall'] < args.target_recall - 0.001: 
                improved = True
            elif tR['f2'] > best['f2'] + 0.001: # Recall 都达标的情况下，比 F2
                improved = True
        
        # 如果之前没达标，现在也没达标，就比 Recall
        elif best['recall'] < args.target_recall - 0.001 and tR['r'] > best['recall'] + 0.001:
            improved = True

        if improved:
            best.update({"recall": tR['r'], "f2": tR['f2'], "auc": m['pr_auc'], "epoch": epoch})
            torch.save({
                "state": copy.deepcopy(model.state_dict()),
                "threshold": tR['t'],
                "metrics": m
            }, best["path"])
            print(f"  --> Saved Best! F2={tR['f2']:.4f}")

    print(f"\nDone. Best Epoch {best['epoch']}: Recall={best['recall']:.4f}, F2={best['f2']:.4f}")

if __name__ == "__main__":
    main()