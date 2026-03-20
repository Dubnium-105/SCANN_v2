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


def _is_aligned_crop_artifact(path: Path) -> bool:
    """是否为对齐裁剪产物文件。"""
    return path.stem.lower().endswith("__aligned_crop")


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
        if (
            f.is_file()
            and f.suffix.lower() in FITS_EXTENSIONS
            and not _is_aligned_crop_artifact(f)
        ):
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

    通过文件名主干匹配，支持智能前缀兼容（处理 FW_、fw_ 等前缀差异）。

    Args:
        new_folder: 新图文件夹
        old_folder: 旧图文件夹

    Returns:
        (配对列表, 仅新图列表, 仅旧图列表)
    """
    new_files = scan_fits_folder(new_folder)
    old_files = scan_fits_folder(old_folder)

    # 构建名称→路径映射（原始大小写和小写）
    new_map = {f.stem.lower(): f for f in new_files}
    old_map = {f.stem.lower(): f for f in old_files}
    new_stem_to_file = {f.stem: f for f in new_files}  # 原始大小写映射

    # ─── 智能配对: 处理时间戳和 FW_ 等常见前缀差异 ───
    # 与标注工具的前缀兼容机制保持一致
    _STRIP_PREFIXES = ("FW_", "fw_", "Fw_")

    def _extract_datetime_prefix(stem: str) -> str | None:
        """提取时间戳前缀，格式如: 20221227T214251__（支持大小写）"""
        if len(stem) < 17:
            return None
        prefix = stem[:15]
        if (
            prefix[0:8].isdigit()
            and prefix[8].lower() == "t"
            and prefix[9:15].isdigit()
            and stem[15:17] == "__"
        ):
            return prefix
        return None

    def _normalize_stem(stem: str) -> str:
        """去除时间戳前缀和 FW_ 等常见前缀用于匹配"""
        # 先去除时间戳前缀（如 20221227T214251__）
        datetime_prefix = _extract_datetime_prefix(stem)
        if datetime_prefix is not None:
            stem = stem[17:]  # 去除 "YYYYMMDDThhmmss__"
        
        # 再去除 FW_ 等前缀
        for prefix in _STRIP_PREFIXES:
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    # 为旧图建立 normalized_stem → original_stem 映射
    old_norm_map: dict[str, str] = {}
    for stem in old_map:
        norm = _normalize_stem(stem)
        old_norm_map[norm] = stem

    # 尝试将 new 文件与 old 文件匹配（先精确匹配，再去前缀匹配）
    matched_old_stems: set[str] = set()
    new_to_old: dict[str, str] = {}  # new_stem → old_stem
    for stem in new_map:
        if stem in old_map:
            new_to_old[stem] = stem
            matched_old_stems.add(stem)
        else:
            # 尝试 normalize 后匹配
            norm = _normalize_stem(stem)
            if norm in old_norm_map:
                old_stem = old_norm_map[norm]
                if old_stem not in matched_old_stems:
                    new_to_old[stem] = old_stem
                    matched_old_stems.add(old_stem)

    pairs = []
    only_new = []
    only_old = []

    # 配对: 以 new 为主 + 未匹配的 old
    unmatched_old = set(old_map.keys()) - matched_old_stems

    # 处理配对
    for new_stem in new_to_old:
        old_stem = new_to_old[new_stem]
        # 使用原始大小写的新图文件名进行 normalize
        new_file = new_map[new_stem]
        norm_name = _normalize_stem(new_file.stem)
        pairs.append(FitsImagePair(
            name=norm_name,
            new_path=new_file.path,
            old_path=old_map[old_stem].path,
        ))

    # 处理仅新图
    for new_stem in new_map:
        if new_stem not in new_to_old:
            # 使用原始大小写进行 normalize
            new_file = new_map[new_stem]
            norm_name = _normalize_stem(new_file.stem)
            only_new.append(norm_name)

    # 处理仅旧图
    for old_stem in unmatched_old:
        # 使用原始大小写进行 normalize
        old_file = old_map[old_stem]
        norm_name = _normalize_stem(old_file.stem)
        only_old.append(norm_name)

    return pairs, only_new, only_old
