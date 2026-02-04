# -*- coding: utf-8 -*-
"""
SCANN 配置管理模块
- ProcessingConfig: 处理参数配置
- ConfigManager: 用户配置持久化
"""

import os
import json

# ================= 核心配置区 =================
class ProcessingConfig:
    """图像处理与推理的核心参数配置"""
    # === Top-K 配置 ===
    TOPK_CHEAP = 20        # 按 cheap_score
    TOPK_RISE  = 20        # 按 rise
    TOPK_CONTRAST = 20     # 按 contrast
    TOPK_UNION = True      # 是否启用并集保底
    
    INFER_CHUNK = 512      # 推理分块
    CROP_SZ = 80
    RESIZE_HW = (224, 224) # 训练输入
    
    # --- Cheap Score 配置 ---
    # 模式: 'robust_z' (推荐) 或 'rise_only' (仅调试用)
    CHEAP_MODE = 'robust_z' 
    
    # robust_z 模式下的权重
    W_RISE = 2.0
    W_CONTRAST = 1.0
    W_SHARP = 0.5
    W_AREA_PENALTY = 0.3   # * abs(z_area)

    # 并行配置
    NUM_WORKERS = 4        # 预处理线程数


# ================= 配置文件路径 =================
CONFIG_FILE = os.path.join(os.getcwd(), "SCANN_config.json")


class ConfigManager:
    """用户配置的加载与保存"""
    
    @staticmethod
    def load():
        """加载配置文件，返回配置字典"""
        default = {
            "last_folder": "",
            "thresh": 80,
            "min_area": 6,
            "sharpness": 1.2,
            "contrast": 15,
            "kill_flat": True,
            "kill_hist": True,
            "kill_dipole": True,
            "auto_crop": True,
            "edge_margin": 10,
            "auto_clear_cache": False,
            "dynamic_thresh": False,
            "max_sharpness": 5.0,
            "model_path": "",
            "crowd_high_score": 0.85,
            "crowd_high_count": 10,
            "crowd_high_penalty": 0.50,
            "jpg_download_dir": "",
            "fits_download_dir": ""
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    default.update(data)
            except Exception:
                pass
        return default

    @staticmethod
    def save(data):
        """保存配置到文件"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
