# -*- coding: utf-8 -*-
"""
SCANN 图像处理模块
- process_stage_a: Stage A 预处理（轮廓检测、特征提取、Top-K筛选）
- _prepare_patch_tensor_80_static: Patch 裁剪与张量转换
"""

import time
import traceback
import cv2
import numpy as np
import torch

from .config import ProcessingConfig


def _prepare_patch_tensor_80_static(gray_a, gray_b, gray_c, cx, cy, crop_sz=80):
    """
    CPU Side: Crop -> Stack(A,B,C) -> Tensor
    Returns: [3, 80, 80] Float Tensor (0~1) on CPU
    """
    half = crop_sz // 2
    h, w = gray_a.shape[:2]
    
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_sz, y1 + crop_sz
    
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    
    def get_crop(img):
        if sx1 >= sx2 or sy1 >= sy2: 
            return np.zeros((crop_sz, crop_sz), dtype=np.uint8)
        crop = img[sy1:sy2, sx1:sx2]
        if (sx2 - sx1) != crop_sz or (sy2 - sy1) != crop_sz:
            padded = np.zeros((crop_sz, crop_sz), dtype=np.uint8)
            dx1 = sx1 - x1
            dy1 = sy1 - y1
            dx2 = dx1 + (sx2 - sx1)
            dy2 = dy1 + (sy2 - sy1)
            padded[dy1:dy2, dx1:dx2] = crop
            return padded
        return crop

    pa = get_crop(gray_a)
    pb = get_crop(gray_b)
    pc = get_crop(gray_c)
    
    # Merge 3 channels
    merged = np.stack([pa, pb, pc], axis=2)  # (80, 80, 3)
    
    # HWC -> CHW, Float, Scale
    tensor = torch.from_numpy(merged.transpose(2, 0, 1)).float()
    tensor /= 255.0
    
    return tensor


def process_stage_a(name, paths, params, config_dict):
    """
    Stage A Worker Function:
    1. Read Images
    2. Auto Crop
    3. Generate Candidates (Heuristics)
    4. Compute Cheap Score
    5. Top-K Filter
    6. Prepare Patch Tensors (CPU)
    
    Args:
        name: 图像组名称
        paths: 包含 'a', 'b', 'c' 三个路径的字典
        params: 用户配置参数
        config_dict: 处理配置
        
    Returns:
        dict: 包含 candidates, patch_tensors, crop_rect 等信息
    """
    try:
        t0 = time.time()
        
        # 1. Read Images
        if not all(k in paths for k in ['a', 'b', 'c']):
            return None
        img_a = cv2.imread(paths['a']) 
        img_b = cv2.imread(paths['b'])
        img_c = cv2.imread(paths['c'])
        if img_a is None or img_b is None or img_c is None:
            return None

        # 2. Auto Crop
        x_off, y_off, w, h = 0, 0, img_a.shape[1], img_a.shape[0]
        if params['auto_crop']:
            gray_full = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            _, thr_w = cv2.threshold(gray_full, 240, 255, cv2.THRESH_BINARY_INV)
            ctrs, _ = cv2.findContours(thr_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if ctrs:
                c_max = max(ctrs, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c_max)
                pad = 2
                x_off = max(0, bx + pad)
                y_off = max(0, by + pad)
                w = max(1, bw - 2 * pad)
                h = max(1, bh - 2 * pad)
        crop_rect = (x_off, y_off, w, h)
        
        gray_a = cv2.cvtColor(img_a[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(img_c[y_off:y_off+h, x_off:x_off+w], cv2.COLOR_BGR2GRAY)
        
        # 3. Generate Candidates
        candidates = []
        blurred = cv2.GaussianBlur(gray_a, (3, 3), 0)
        
        actual_thresh = params['thresh']
        if params.get('dynamic_thresh', False):
            actual_thresh = np.median(gray_a) + params['thresh']
            
        _, bin_img = cv2.threshold(blurred, actual_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = gray_a.shape
        p_min_area = params['min_area']
        p_edge = params.get('edge_margin', 10)
        p_sharp = params['sharpness']
        p_max_sharp = params.get('max_sharpness', 5.0)
        p_contrast = params['contrast']
        do_flat = params['kill_flat']
        do_dipole = params['kill_dipole']
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < p_min_area or area > 600:
                continue
            
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx < p_edge) or (by < p_edge) or (bx + bw > w_img - p_edge) or (by + bh > h_img - p_edge):
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            # Transient Check
            check_r = 3
            y0_r, y1_r = max(0, cy - check_r), min(h_img, cy + check_r + 1)
            x0_r, x1_r = max(0, cx - check_r), min(w_img, cx + check_r + 1)
            roi_b = gray_b[y0_r:y1_r, x0_r:x1_r]
            roi_c = gray_c[y0_r:y1_r, x0_r:x1_r]
            if roi_b.size == 0 or roi_c.size == 0:
                continue
            
            val_b = float(np.max(roi_b))
            val_c = float(np.max(roi_c))
            rise = val_b - val_c
            
            roi_spot = gray_a[by:by+bh, bx:bx+bw]
            if roi_spot.size == 0:
                continue
            peak = float(np.max(roi_spot))
            mean = float(np.mean(roi_spot))
            median_spot = float(np.median(roi_spot))
            sharpness = peak / (mean + 1e-6)
            contrast = peak - median_spot
            
            if do_flat:
                if sharpness < p_sharp:
                    continue
                if sharpness > p_max_sharp:
                    continue
                if contrast < p_contrast:
                    continue
            
            extent = float(area) / (bw * bh)
            aspect = float(bw) / bh if bh > 0 else 0
            if area > 20 and extent > 0.90:
                continue
            if aspect > 3.0 or aspect < 0.33:
                continue
            
            if do_dipole:
                pad_d = 4
                dy0, dy1 = max(0, by - pad_d), min(h_img, by + bh + pad_d)
                dx0, dx1 = max(0, bx - pad_d), min(w_img, bx + bw + pad_d)
                if cv2.minMaxLoc(gray_a[dy0:dy1, dx0:dx1])[0] < 15:
                    continue
                
            candidates.append({
                'x': cx, 'y': cy, 'area': area,
                'sharp': sharpness, 'contrast': contrast,
                'peak': peak, 'rise': rise,
                'val_b': val_b, 'val_c': val_c,
                'crop_off': (x_off, y_off),
                'manual': False
            })

        # 4. Cheap Score & Top-K
        if candidates:
            # --- Cheap Score ---
            if config_dict['cheap_mode'] == 'robust_z' and len(candidates) > 5:
                rises = np.array([c['rise'] for c in candidates])
                conts = np.array([c['contrast'] for c in candidates])
                sharps = np.array([c['sharp'] for c in candidates])
                areas = np.array([c['area'] for c in candidates])
                
                def get_z(arr):
                    med = np.median(arr)
                    mad = np.median(np.abs(arr - med))
                    if mad < 1e-6:
                        return arr - med
                    return (arr - med) / (1.4826 * mad)
                    
                z_rise = get_z(rises)
                z_cont = get_z(conts)
                z_sharp = get_z(sharps)
                z_area = get_z(areas)
                
                scores = (ProcessingConfig.W_RISE * np.clip(z_rise, -5, 5) + 
                          ProcessingConfig.W_CONTRAST * np.clip(z_cont, -5, 5) +
                          ProcessingConfig.W_SHARP * np.clip(z_sharp, -5, 5) - 
                          ProcessingConfig.W_AREA_PENALTY * np.abs(z_area))
                for i, c in enumerate(candidates):
                    c['cheap_score'] = float(scores[i])
            else:
                for c in candidates:
                    c['cheap_score'] = c['rise']
            
            # --- Top-K Union ---
            if config_dict['topk_union']:
                c_cheap = sorted(candidates, key=lambda x: x['cheap_score'], reverse=True)[:config_dict['topk_cheap']]
                c_rise = sorted(candidates, key=lambda x: x['rise'], reverse=True)[:config_dict['topk_rise']]
                c_cont = sorted(candidates, key=lambda x: x['contrast'], reverse=True)[:config_dict['topk_contrast']]
                
                unique_map = {}
                for c in c_cheap + c_rise + c_cont:
                    key = (c['x'], c['y'])
                    if key not in unique_map:
                        unique_map[key] = c
                top_candidates = list(unique_map.values())
            else:
                candidates.sort(key=lambda x: x['cheap_score'], reverse=True)
                top_candidates = candidates[:config_dict['topk_cheap']]
        else:
            top_candidates = []

        # 5. Prepare Patch Tensors (CPU)
        patch_tensors = []
        final_candidates = []
        
        for cand in top_candidates:
            try:
                t = _prepare_patch_tensor_80_static(gray_a, gray_b, gray_c, cand['x'], cand['y'], crop_sz=config_dict['crop_sz'])
                patch_tensors.append(t)
                final_candidates.append(cand)
            except Exception:
                pass  # Skip failed patches

        t_stage_a = time.time() - t0
        return {
            'name': name,
            'candidates': final_candidates,
            'patch_tensors': patch_tensors,
            'crop_rect': crop_rect,
            'n_raw': len(candidates),
            't_stage_a': t_stage_a
        }

    except Exception as e:
        return {'error': str(e), 'name': name, 'traceback': traceback.format_exc()}
