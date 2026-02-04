# -*- coding: utf-8 -*-
"""
SCANN 批量处理工作线程模块
- BatchWorker: AI 批量推理线程
"""

import os
import json
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import torch
from torchvision import models
from PyQt5.QtCore import QThread, pyqtSignal

from .config import ProcessingConfig
from .database import DatabaseManager
from .processing import process_stage_a


class BatchWorker(QThread):
    """AI 批量推理工作线程"""
    
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)

    def __init__(self, groups, params):
        super().__init__()
        self.groups = groups
        self.params = params
        self._is_running = True
        
        # === AI 初始化 ===
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        self.model = None
        self.has_model = False
        
        # Normalization constants
        self.norm_mean = torch.tensor([0.2601623164967817, 0.2682929013103806, 0.26861570225529907]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.09133092247248126, 0.10773878132887775, 0.10867911864809723]).view(1, 3, 1, 1).to(self.device)
        
        # Load Model
        model_path = self.params.get('model_path', '')
        if not model_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "best_model.pth")
        
        print(f"DEBUG: 正在尝试加载模型: {model_path}")
        print(f"DEBUG: 使用设备: {self.device}")

        if os.path.exists(model_path):
            try:
                # 1. Structure
                self.model = models.resnet18(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, 2)
                
                # 2. Weights
                ckpt = torch.load(model_path, map_location=self.device)
                
                state_dict = None
                if isinstance(ckpt, dict):
                    if "state" in ckpt:
                        state_dict = ckpt["state"]
                    elif "model_state" in ckpt:
                        state_dict = ckpt["model_state"]
                    else:
                        state_dict = ckpt
                else:
                    state_dict = ckpt
                
                # Clean prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # Strict Load
                self.model.load_state_dict(new_state_dict, strict=True)
                self.model.to(self.device)
                self.model.eval()
                self.has_model = True
                
                print(f"✅✅✅ AI 模型加载成功！")
                
            except Exception as e:
                print("\n❌❌❌ AI 模型加载失败！")
                traceback.print_exc()
                self.has_model = False
                raise e
        else:
            print(f"❌ 未找到模型文件: {model_path}")
            self.has_model = False
            raise FileNotFoundError(f"AI Model not found at: {model_path}")

        # Initialize Async DB Writer
        DatabaseManager.init_async()

    def stop(self):
        """停止工作线程"""
        self._is_running = False

    def verify_model_ready(self):
        """Fail-fast 模型验证"""
        if not self.has_model:
            raise RuntimeError(f"AI Model NOT Ready")
        
        print("DEBUG: Performing Model Dry-Run...")
        try:
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            dummy = (dummy - self.norm_mean) / self.norm_std
            with torch.no_grad():
                _ = self.model(dummy)
            print("✅ Dry-Run Passed.")
        except Exception as e:
            print("❌ Dry-Run Failed!")
            traceback.print_exc()
            raise RuntimeError(f"Model Dry-Run Failed: {e}")

    def _compute_params_hash(self):
        """计算参数哈希值，用于缓存校验"""
        key_params = {
            'thresh': self.params['thresh'],
            'min_area': self.params['min_area'],
            'sharpness': self.params['sharpness'],
            'max_sharpness': self.params.get('max_sharpness', 5.0),
            'contrast': self.params['contrast'],
            'edge_margin': self.params.get('edge_margin', 10),
            'kill_flat': self.params['kill_flat'],
            'kill_hist': self.params['kill_hist'],
            'kill_dipole': self.params['kill_dipole'],
            'dynamic_thresh': self.params.get('dynamic_thresh', False),
            'model_path': self.params.get('model_path', ''),
            'topk_cheap': ProcessingConfig.TOPK_CHEAP,
            'topk_union': ProcessingConfig.TOPK_UNION
        }
        s = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    def run(self):
        """主运行函数"""
        # 1. Fail-fast check
        try:
            self.verify_model_ready()
        except Exception as e:
            print(f"❌ Batch Aborted: {e}")
            traceback.print_exc()
            self.finished.emit({})
            raise e

        print("DEBUG: Loading DB summaries...")
        db_summaries = DatabaseManager.load_summaries_map()
        
        results = {}
        total = len(self.groups)
        count = 0
        current_hash = self._compute_params_hash()
        
        sorted_keys = sorted(self.groups.keys())
        
        # --- Parallel Execution Setup ---
        executor = ThreadPoolExecutor(max_workers=ProcessingConfig.NUM_WORKERS)
        futures = set()
        
        worker_config = {
            'crop_sz': ProcessingConfig.CROP_SZ,
            'cheap_mode': ProcessingConfig.CHEAP_MODE,
            'topk_union': ProcessingConfig.TOPK_UNION,
            'topk_cheap': ProcessingConfig.TOPK_CHEAP,
            'topk_rise': ProcessingConfig.TOPK_RISE,
            'topk_contrast': ProcessingConfig.TOPK_CONTRAST
        }

        # Global Inference Batching
        pending_inference_items = []
        pending_results_map = {}

        def flush_inference_batch(force=False):
            nonlocal pending_inference_items, count
            BATCH_SIZE = ProcessingConfig.INFER_CHUNK
            
            while len(pending_inference_items) >= BATCH_SIZE or (force and pending_inference_items):
                chunk_size = BATCH_SIZE if len(pending_inference_items) >= BATCH_SIZE else len(pending_inference_items)
                batch_items = pending_inference_items[:chunk_size]
                pending_inference_items = pending_inference_items[chunk_size:]
                
                try:
                    tensors = [item['tensor'] for item in batch_items]
                    stack = torch.stack(tensors).to(self.device, non_blocking=True)
                    
                    # Resize & Norm on GPU
                    stack = torch.nn.functional.interpolate(stack, size=ProcessingConfig.RESIZE_HW, mode='bilinear', align_corners=False)
                    stack = (stack - self.norm_mean) / self.norm_std
                    
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                            logits = self.model(stack)
                            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    
                    # Distribute results
                    updates_by_name = {}
                    for idx, prob in enumerate(probs):
                        item = batch_items[idx]
                        name = item['name']
                        cand_idx = item['cand_idx']
                        
                        if name not in updates_by_name:
                            updates_by_name[name] = []
                        updates_by_name[name].append((cand_idx, prob))
                        
                    # Apply updates & Check completion
                    for name, updates in updates_by_name.items():
                        cands = pending_results_map[name]['candidates']
                        for c_idx, score in updates:
                            cands[c_idx]['ai_score'] = float(score)
                            
                        pending_results_map[name]['remaining'] -= len(updates)
                        
                        if pending_results_map[name]['remaining'] <= 0:
                            final_cands = [c for c in cands if 'ai_score' in c]
                            p = self.params
                            hs = float(p.get('crowd_high_score', 0.85))
                            hc = int(p.get('crowd_high_count', 10))
                            hp = float(p.get('crowd_high_penalty', 0.50))
                            high_cnt = sum(1 for c in final_cands if c.get('ai_score', 0) >= hs)
                            if high_cnt > hc:
                                for c in final_cands:
                                    if c.get('ai_score', 0) >= hs:
                                        c['ai_score'] = max(0.0, float(c['ai_score']) - hp)
                            crop_rect = pending_results_map[name]['crop_rect']
                            
                            # --- 数据保护：合并已有的手动/判决目标 ---
                            existing_full = DatabaseManager.get_record(name)
                            if existing_full and "candidates" in existing_full:
                                for ec in existing_full["candidates"]:
                                    if ec.get("manual", False) or ec.get("verdict") is not None:
                                        is_dup = False
                                        for nc in final_cands:
                                            if abs(nc['x'] - ec['x']) < 5 and abs(nc['y'] - ec['y']) < 5:
                                                is_dup = True
                                                if ec.get("verdict"):
                                                    nc["verdict"] = ec["verdict"]
                                                    nc["saved"] = ec.get("saved", True)
                                                break
                                        if not is_dup:
                                            final_cands.append(ec)
                            
                            DatabaseManager.update_record(name, final_cands, crop_rect=crop_rect, params_hash=current_hash)
                            results[name] = {"candidates": final_cands, "status": "unseen", "crop_rect": crop_rect}
                            
                            del pending_results_map[name]
                            
                            count += 1
                            if count % 5 == 0:
                                self.progress.emit(count, total, f"AI处理中: {name}")

                except Exception as e:
                    print(f"❌ Global Batch Inference Error")
                    traceback.print_exc()
                    raise e

        # --- Main Loop ---
        for name in sorted_keys:
            if not self._is_running:
                break
            
            summary = db_summaries.get(name)
            if summary:
                cached_hash = summary.get('params_hash', '')
                if summary.get('has_ai', 0) and summary.get('candidates_count', 0) > 0 and cached_hash == current_hash:
                    record = DatabaseManager.get_record(name)
                    if record:
                        results[name] = record
                        count += 1
                        self.progress.emit(count, total, f"已从库加载: {name}")
                        continue
            
            # Submit Task (with bounded buffer)
            while len(futures) >= ProcessingConfig.NUM_WORKERS * 2:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    res = f.result()
                    if not res:
                        continue
                    if 'error' in res:
                        raise RuntimeError(res['error'])
                    
                    r_name = res['name']
                    r_cands = res['candidates']
                    r_tensors = res['patch_tensors']
                    
                    if not r_cands:
                        final_cands = []
                        existing_full = DatabaseManager.get_record(r_name)
                        if existing_full and "candidates" in existing_full:
                            for ec in existing_full["candidates"]:
                                if ec.get("manual", False) or ec.get("verdict") is not None:
                                    final_cands.append(ec)
                        
                        DatabaseManager.update_record(r_name, final_cands, crop_rect=res['crop_rect'], params_hash=current_hash)
                        count += 1
                        continue

                    pending_results_map[r_name] = {
                        'candidates': r_cands,
                        'remaining': len(r_cands),
                        'crop_rect': res['crop_rect']
                    }
                    
                    for i, t in enumerate(r_tensors):
                        pending_inference_items.append({'name': r_name, 'cand_idx': i, 'tensor': t})
                    
                    flush_inference_batch()

            if not self._is_running:
                break
            
            # Submit new task
            future = executor.submit(process_stage_a, name, self.groups[name], self.params, worker_config)
            futures.add(future)

        # Drain remaining
        while futures:
            if not self._is_running:
                break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                if not res:
                    continue
                if 'error' in res:
                    raise RuntimeError(res['error'])
                
                r_name = res['name']
                r_cands = res['candidates']
                r_tensors = res['patch_tensors']
                
                if not r_cands:
                    final_cands = []
                    existing_full = DatabaseManager.get_record(r_name)
                    if existing_full and "candidates" in existing_full:
                        for ec in existing_full["candidates"]:
                            if ec.get("manual", False) or ec.get("verdict") is not None:
                                final_cands.append(ec)
                                
                    DatabaseManager.update_record(r_name, final_cands, crop_rect=res['crop_rect'], params_hash=current_hash)
                    count += 1
                    continue

                pending_results_map[r_name] = {
                    'candidates': r_cands,
                    'remaining': len(r_cands),
                    'crop_rect': res['crop_rect']
                }
                
                for i, t in enumerate(r_tensors):
                    pending_inference_items.append({'name': r_name, 'cand_idx': i, 'tensor': t})
                
                flush_inference_batch()

        # Final flush
        flush_inference_batch(force=True)
        
        executor.shutdown()
        self.finished.emit(results)
