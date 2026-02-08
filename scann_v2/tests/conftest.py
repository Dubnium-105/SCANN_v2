"""测试套件共享 fixtures

提供合成 FITS 数据、临时目录、模拟配置等可复用测试夹具。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


# ─── 合成 FITS 数据 ───


@pytest.fixture
def synth_fits_data_16bit() -> np.ndarray:
    """生成一张 128x128 的 16bit 合成 FITS 图像"""
    rng = np.random.default_rng(42)
    # 背景 + 星点
    bg = rng.normal(loc=1000, scale=50, size=(128, 128)).astype(np.uint16)
    # 添加一个亮星 (高斯)
    y, x = np.mgrid[0:128, 0:128]
    star = 5000 * np.exp(-(((x - 64) ** 2 + (y - 64) ** 2) / (2 * 3 ** 2)))
    return (bg + star.astype(np.uint16)).astype(np.uint16)


@pytest.fixture
def synth_fits_data_32bit() -> np.ndarray:
    """生成一张 128x128 的 32bit 合成 FITS 图像"""
    rng = np.random.default_rng(42)
    bg = rng.normal(loc=50000, scale=500, size=(128, 128)).astype(np.int32)
    y, x = np.mgrid[0:128, 0:128]
    star = 100000 * np.exp(-(((x - 64) ** 2 + (y - 64) ** 2) / (2 * 3 ** 2)))
    return (bg + star.astype(np.int32)).astype(np.int32)


@pytest.fixture
def synth_float_image() -> np.ndarray:
    """生成一张 float32 0~1 范围的图像"""
    rng = np.random.default_rng(42)
    return rng.random((128, 128), dtype=np.float32)


@pytest.fixture
def synth_image_pair(synth_fits_data_16bit) -> tuple[np.ndarray, np.ndarray]:
    """生成一对图像 (新图, 旧图带微小偏移)"""
    new_img = synth_fits_data_16bit.copy()
    old_img = np.roll(synth_fits_data_16bit, shift=(3, -2), axis=(0, 1))
    return new_img, old_img


# ─── 临时目录与文件 ───


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """临时目录"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def fits_file_pair(tmp_dir, synth_fits_data_16bit) -> tuple[Path, Path]:
    """在临时目录创建一对 FITS 文件 (新/旧)"""
    try:
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy not installed")

    new_dir = tmp_dir / "new"
    old_dir = tmp_dir / "old"
    new_dir.mkdir()
    old_dir.mkdir()

    hdr = fits.Header()
    hdr["OBJECT"] = "TestField"
    hdr["DATE-OBS"] = "2024-01-15T20:30:00"
    hdr["TELESCOP"] = "TestScope"

    new_path = new_dir / "field_001.fits"
    old_path = old_dir / "field_001.fits"

    shifted = np.roll(synth_fits_data_16bit, shift=(2, -1), axis=(0, 1))

    fits.writeto(str(new_path), synth_fits_data_16bit, header=hdr, overwrite=True)
    fits.writeto(str(old_path), shifted, header=hdr, overwrite=True)

    return new_path, old_path


@pytest.fixture
def sample_fits_folder(tmp_dir, synth_fits_data_16bit) -> Path:
    """创建包含多个 FITS 文件的文件夹"""
    try:
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy not installed")

    folder = tmp_dir / "fits_folder"
    folder.mkdir()
    hdr = fits.Header()
    hdr["OBJECT"] = "TestField"

    for i in range(5):
        rng = np.random.default_rng(i)
        data = synth_fits_data_16bit + rng.integers(-50, 50, size=synth_fits_data_16bit.shape, dtype=np.int16)
        data = np.clip(data, 0, 65535).astype(np.uint16)
        fits.writeto(str(folder / f"img_{i:03d}.fits"), data, header=hdr, overwrite=True)

    # 也创建一些 .fit 文件
    fits.writeto(str(folder / "extra.fit"), synth_fits_data_16bit, header=hdr, overwrite=True)

    return folder


# ─── 配置 ───


@pytest.fixture
def sample_config_dict() -> dict:
    """示例配置字典 (与 config.py 的 JSON 结构对齐)"""
    return {
        "new_folder": "",
        "old_folder": "",
        "save_folder": "",
        "telescope": {
            "focal_length_mm": 2000.0,
            "pixel_size_um": 9.0,
            "pixel_scale_arcsec": 0.0,
            "camera_rotation_deg": 0.0,
        },
        "observatory": {
            "code": "C42",
            "name": "Test Observatory",
            "longitude": 116.0,
            "latitude": 40.0,
            "altitude": 900.0,
        },
        "thresh": 80,
        "min_area": 6,
        "sharpness": 1.2,
        "max_sharpness": 5.0,
        "contrast": 15,
        "edge_margin": 10,
        "dynamic_thresh": False,
        "kill_flat": True,
        "kill_dipole": True,
        "model_path": "best_model.pth",
        "crowd_high_score": 0.85,
        "crowd_high_count": 10,
        "crowd_high_penalty": 0.50,
        "save_bit_depth": 16,
        "blink_speed_ms": 500,
        "mpcorb_path": "",
        "limit_magnitude": 20.0,
    }


@pytest.fixture
def config_file(tmp_dir, sample_config_dict) -> Path:
    """在临时目录创建配置文件"""
    cfg_path = tmp_dir / "config.json"
    cfg_path.write_text(json.dumps(sample_config_dict, indent=2), encoding="utf-8")
    return cfg_path


# ─── 数据库 ───


@pytest.fixture
def db_path(tmp_dir) -> Path:
    """临时数据库路径"""
    return tmp_dir / "test_candidates.db"
