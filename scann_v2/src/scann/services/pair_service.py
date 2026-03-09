"""图像配对服务。

职责:
- 扫描新旧图文件夹
- 执行图像配对
- 读取配对对应的 FITS 图像
- 提供配对图像路径解析边界
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from scann.core.fits_io import read_fits
from scann.core.models import ImagePair
from scann.data.file_manager import FitsFileInfo, FitsImagePair, match_new_old_pairs, scan_fits_folder


class PairService:
    """封装图像配对相关的应用服务。

    当前提交仅建立服务边界，不持有 UI 或会话状态。
    """

    def __init__(
        self,
        scan_folder_fn: Callable[[str], list[FitsFileInfo]] = scan_fits_folder,
        match_pairs_fn: Callable[[str, str], tuple[list[FitsImagePair], list[str], list[str]]] = match_new_old_pairs,
        read_fits_fn: Callable[[str | Path], object] = read_fits,
    ) -> None:
        self._scan_folder = scan_folder_fn
        self._match_pairs = match_pairs_fn
        self._read_fits = read_fits_fn

    def scan_new_folder(self, folder: str | Path) -> list[FitsFileInfo]:
        """扫描新图文件夹。"""
        return self._scan_folder(str(folder))

    def scan_old_folder(self, folder: str | Path) -> list[FitsFileInfo]:
        """扫描旧图文件夹。"""
        return self._scan_folder(str(folder))

    def match_pairs(
        self,
        new_folder: str | Path,
        old_folder: str | Path,
    ) -> tuple[list[FitsImagePair], list[str], list[str]]:
        """匹配新旧图像配对。"""
        return self._match_pairs(str(new_folder), str(old_folder))

    def resolve_pair_image_paths(self, pair: FitsImagePair) -> tuple[Path, Path]:
        """解析配对图像路径。

        本提交仅返回原始新旧图路径，后续提交再迁移对齐产物复用逻辑。
        """
        return Path(pair.new_path), Path(pair.old_path)

    def load_pair(self, pair: FitsImagePair) -> ImagePair:
        """读取一对 FITS 图像并转换为内存模型。"""
        new_path, old_path = self.resolve_pair_image_paths(pair)
        new_image = self._read_fits(new_path)
        old_image = self._read_fits(old_path)
        return ImagePair(
            name=pair.name,
            new_image=new_image,
            old_image=old_image,
            aligned=False,
        )