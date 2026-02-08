"""文件管理模块

职责:
- 扫描 FITS 文件夹
- 新旧图配对
- 训练数据文件组织
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class FitsFileInfo:
    """FITS 文件信息"""
    path: Path
    stem: str         # 文件名主干
    size_bytes: int
    modified_time: float

    @property
    def filename(self) -> str:
        """完整文件名 (含后缀)"""
        return self.path.name


@dataclass
class FitsImagePair:
    """新旧图像配对信息"""
    name: str            # 配对名称
    new_path: Path       # 新图路径
    old_path: Path       # 旧图路径


FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fts2"}


def scan_fits_folder(folder: str) -> List[FitsFileInfo]:
    """扫描 FITS 文件夹

    Args:
        folder: 文件夹路径

    Returns:
        FITS 文件信息列表
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder}")

    results = []
    for f in sorted(folder_path.iterdir()):
        if f.is_file() and f.suffix.lower() in FITS_EXTENSIONS:
            stat = f.stat()
            results.append(FitsFileInfo(
                path=f,
                stem=f.stem,
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
            ))
    return results


def match_new_old_pairs(
    new_folder: str,
    old_folder: str,
) -> Tuple[List[FitsImagePair], List[str], List[str]]:
    """配对新旧 FITS 文件

    通过文件名主干匹配。

    Args:
        new_folder: 新图文件夹
        old_folder: 旧图文件夹

    Returns:
        (配对列表, 仅新图列表, 仅旧图列表)
    """
    new_files = scan_fits_folder(new_folder)
    old_files = scan_fits_folder(old_folder)

    # 构建名称→路径映射
    new_map = {f.stem.lower(): f for f in new_files}
    old_map = {f.stem.lower(): f for f in old_files}

    pairs = []
    only_new = []
    only_old = []

    # 匹配
    all_stems = set(new_map.keys()) | set(old_map.keys())
    for stem in sorted(all_stems):
        if stem in new_map and stem in old_map:
            pairs.append(FitsImagePair(
                name=new_map[stem].stem,  # 使用原始大小写
                new_path=new_map[stem].path,
                old_path=old_map[stem].path,
            ))
        elif stem in new_map:
            only_new.append(new_map[stem].stem)
        else:
            only_old.append(old_map[stem].stem)

    return pairs, only_new, only_old
