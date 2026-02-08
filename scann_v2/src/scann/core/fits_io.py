"""FITS 文件 I/O 操作模块

职责:
- 读取/写入 FITS 文件
- 提取文件头信息
- 保存约束: 整数格式(16/32bit)、不修改原始头信息
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from scann.core.models import BitDepth, FitsHeader, FitsImage


def read_fits(path: Union[str, Path]) -> FitsImage:
    """读取 FITS 文件，返回数据和头信息

    Args:
        path: FITS 文件路径

    Returns:
        FitsImage: 包含数据和头信息的对象

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式无效
    """
    from astropy.io import fits as astropy_fits

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS 文件不存在: {path}")

    with astropy_fits.open(str(path)) as hdul:
        # 查找第一个含有数据的 HDU
        data = None
        header_dict = {}
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data.copy()
                header_dict = dict(hdu.header)
                break

        if data is None:
            raise ValueError(f"FITS 文件中没有图像数据: {path}")

    # 将 FITS 大端序 ('>i2', '>i4' 等) 转换为本机字节序
    if data.dtype.byteorder not in ('=', '|', sys.byteorder[0]):
        data = data.astype(data.dtype.newbyteorder('='))

    header = FitsHeader(raw=header_dict)
    return FitsImage(data=data, header=header, path=path)


def read_header(path: Union[str, Path]) -> FitsHeader:
    """仅读取 FITS 文件头（不加载数据，更快）

    Args:
        path: FITS 文件路径

    Returns:
        FitsHeader: 文件头信息
    """
    from astropy.io import fits as astropy_fits

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS 文件不存在: {path}")

    header_dict = {}
    with astropy_fits.open(str(path)) as hdul:
        for hdu in hdul:
            if hdu.header:
                header_dict = dict(hdu.header)
                break

    return FitsHeader(raw=header_dict)


def write_fits(
    path: Union[str, Path],
    data: np.ndarray,
    header: Optional[FitsHeader] = None,
    bit_depth: BitDepth = BitDepth.INT16,
) -> Path:
    """保存 FITS 文件

    约束:
    - 数据必须保存为整数格式 (16 or 32 bit)
    - 不修改原始 FITS 文件头

    Args:
        path: 保存路径
        data: 像素数据
        header: 原始文件头（原样保留）
        bit_depth: 保存位深度 (16 or 32)

    Returns:
        Path: 保存的文件路径
    """
    from astropy.io import fits as astropy_fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为整数类型
    if bit_depth == BitDepth.INT16:
        save_data = np.clip(data.astype(np.float64), -32768, 32767).astype(np.int16)
    else:
        save_data = data.astype(np.float64).astype(np.int32)

    # FITS 结构性关键字 (由 astropy 根据实际数据自动管理)
    _STRUCTURAL_KEYS = frozenset({
        "SIMPLE", "BITPIX", "NAXIS", "EXTEND",
        "BZERO", "BSCALE", "BLANK",
        "PCOUNT", "GCOUNT",
    })

    # 构建 Header（保持原始内容不变，但跳过结构性关键字）
    hdr = None
    if header is not None:
        hdr = astropy_fits.Header()
        for key, value in header.raw.items():
            if key in ("", "COMMENT", "HISTORY"):
                continue
            # 跳过 FITS 结构性关键字和 NAXISn
            if key in _STRUCTURAL_KEYS or key.startswith("NAXIS"):
                continue
            try:
                hdr[key] = value
            except (ValueError, KeyError):
                pass  # 跳过无法写入的特殊键

    hdu = astropy_fits.PrimaryHDU(data=save_data, header=hdr)
    hdu.writeto(str(path), overwrite=True)
    return path
