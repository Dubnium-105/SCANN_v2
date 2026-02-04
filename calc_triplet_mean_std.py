import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def iter_pngs(folder):
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(".png"):
                yield os.path.join(root, fn)

def read_triplet_as_float01(path, channel_index=(0,1,2)):
    """
    读取 80x240 三联图，切成三块 80x80 (左/中/右)，按 channel_index 重排，
    返回 shape=(3,H,W) 的 float32，范围 [0,1]
    """
    im = Image.open(path).convert("L")
    w, h = im.size
    if w < 240 or h < 80:
        raise ValueError(f"Bad size {w}x{h} for {path}")

    # 左中右
    parts = [
        im.crop((0,   0,  80, 80)),
        im.crop((80,  0, 160, 80)),
        im.crop((160, 0, 240, 80)),
    ]
    parts = [parts[i] for i in channel_index]

    arrs = []
    for p in parts:
        a = np.asarray(p, dtype=np.float32) / 255.0  # [H,W] in [0,1]
        arrs.append(a)
    x = np.stack(arrs, axis=0)  # [3,H,W]
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", required=True, help="negative folder")
    ap.add_argument("--pos", required=True, help="positive folder")
    ap.add_argument("--max", type=int, default=0, help="max files to use (0=all)")
    ap.add_argument("--order", default="0,1,2", help="channel order for left/mid/right, e.g. 0,1,2")
    args = ap.parse_args()

    order = tuple(int(x.strip()) for x in args.order.split(","))
    files = list(iter_pngs(args.neg)) + list(iter_pngs(args.pos))
    if args.max and args.max > 0:
        files = files[:args.max]

    if len(files) == 0:
        raise RuntimeError("No png files found")

    # Welford：逐步统计 mean/std（按通道）
    n = 0
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)

    for path in tqdm(files, desc="Scanning"):
        x = read_triplet_as_float01(path, channel_index=order)  # [3,80,80]
        # 每张图每通道做像素均值，等价于全像素统计（更高效）
        # 也可以直接 flatten 全像素，但会更慢
        # 这里我们用“全像素”统计更准确：对每张图每通道的所有像素累计
        # 所以逐像素更新：把 (H*W) 个像素当作样本
        c, h, w = x.shape
        x_flat = x.reshape(c, -1).astype(np.float64)  # [3, Npix]
        for i in range(x_flat.shape[1]):
            n += 1
            v = x_flat[:, i]
            delta = v - mean
            mean += delta / n
            m2 += delta * (v - mean)

    var = m2 / max(n - 1, 1)
    std = np.sqrt(var)

    print("\n=== Triplet dataset mean/std (channels = [diff,new,ref] in your chosen order) ===")
    print(f"Total samples (pixels): {n}")
    print(f"mean: {mean.tolist()}")
    print(f"std : {std.tolist()}")

    print("\nCopy into Normalize like:")
    print(f"Normalize(mean={mean.tolist()}, std={std.tolist()})")

if __name__ == "__main__":
    main()
