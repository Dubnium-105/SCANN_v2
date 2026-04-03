"""Dataset helpers for legacy v1 triplet experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .legacy_manifest import entries_for_split, load_legacy_manifest

STORED_COMPONENT_ORDER = ("diff", "new", "old")
INPUT_MODE_TO_CHANNELS = {
    "stored_triplet": ("diff", "new", "old"),
    "new_old_diff": ("new", "old", "diff"),
    "diff_only": ("diff", "diff", "diff"),
    "diff_new": ("diff", "new", "new"),
    "diff_old": ("diff", "old", "old"),
}
SUPPORTED_RESIZE_MODES = {"resize", "pad_resize", "keep"}
DEFAULT_COMPONENT_MEAN = {
    "diff": 0.26366636987971404,
    "new": 0.28187216168254836,
    "old": 0.28376364009886634,
}
DEFAULT_COMPONENT_STD = {
    "diff": 0.08900734308916412,
    "new": 0.12256077701380531,
    "old": 0.12825460877519965,
}
ImageSizeSpec = Union[int, List[int], Tuple[int, int], str]


def semantic_channels_for_input_mode(input_mode: str) -> tuple[str, str, str]:
    normalized = str(input_mode).strip().lower()
    if normalized not in INPUT_MODE_TO_CHANNELS:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return INPUT_MODE_TO_CHANNELS[normalized]


def normalize_image_size_spec(image_size: ImageSizeSpec) -> int | list[int] | str:
    if isinstance(image_size, str):
        normalized = image_size.strip().lower()
        if normalized == "keep":
            return "keep"
        raise ValueError(f"Unsupported image_size string: {image_size}")

    if isinstance(image_size, tuple):
        image_size = list(image_size)

    if isinstance(image_size, list):
        if len(image_size) != 2:
            raise ValueError("image_size list must contain exactly two values: [height, width]")
        height = int(image_size[0])
        width = int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size values must be positive")
        return [height, width]

    size = int(image_size)
    if size <= 0:
        raise ValueError("image_size must be positive")
    return size


def resolve_image_size_spec(
    image_size: ImageSizeSpec,
    *,
    resize_mode: str,
) -> tuple[int, int] | None:
    normalized_resize_mode = str(resize_mode).strip().lower()
    normalized_image_size = normalize_image_size_spec(image_size)

    if normalized_resize_mode == "keep":
        return None

    if normalized_image_size == "keep":
        raise ValueError("image_size='keep' can only be used together with resize_mode='keep'")

    if isinstance(normalized_image_size, list):
        return int(normalized_image_size[0]), int(normalized_image_size[1])

    size = int(normalized_image_size)
    return size, size


def format_image_size_spec(image_size: ImageSizeSpec) -> str:
    normalized = normalize_image_size_spec(image_size)
    if normalized == "keep":
        return "keep"
    if isinstance(normalized, list):
        return f"{normalized[0]}x{normalized[1]}"
    return str(int(normalized))


def image_size_specs_equal(lhs: ImageSizeSpec, rhs: ImageSizeSpec) -> bool:
    return normalize_image_size_spec(lhs) == normalize_image_size_spec(rhs)


class LegacyTripletExperimentDataset(Dataset):
    """Load v1 triplet PNGs with explicit semantic channel mapping."""

    def __init__(
        self,
        manifest: str | Path | dict[str, Any],
        *,
        split: str,
        dataset_root: str | Path | None = None,
        input_mode: str = "new_old_diff",
        image_size: ImageSizeSpec = 224,
        resize_mode: str = "resize",
        normalize: bool = True,
        augment: bool = False,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        enable_rotate_90: bool = False,
    ):
        self.manifest = load_legacy_manifest(manifest)
        self.split = str(split).strip().lower()
        self.dataset_root = Path(dataset_root or self.manifest.get("dataset_root") or ".").resolve()
        self.input_mode = str(input_mode).strip().lower()
        self.channel_names = semantic_channels_for_input_mode(self.input_mode)
        self.resize_mode = str(resize_mode).strip().lower()
        self.image_size = normalize_image_size_spec(image_size)
        self.resize_size = resolve_image_size_spec(self.image_size, resize_mode=self.resize_mode)
        self.normalize = bool(normalize)
        self.augment = bool(augment) and self.split == "train"
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.enable_rotate_90 = bool(enable_rotate_90)

        if self.resize_mode not in SUPPORTED_RESIZE_MODES:
            raise ValueError(f"Unsupported resize mode: {resize_mode}")

        self.entries = entries_for_split(self.manifest, self.split)
        if not self.entries:
            raise ValueError(f"No entries found for split: {self.split}")

        self.mean = [DEFAULT_COMPONENT_MEAN[name] for name in self.channel_names]
        self.std = [max(DEFAULT_COMPONENT_STD[name], 1e-6) for name in self.channel_names]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.dataset_root / str(entry["relative_path"])
        x = self._load_tensor(image_path)
        y = torch.tensor(int(entry["label"]), dtype=torch.long)
        return x, y

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        if not image_path.is_file():
            raise FileNotFoundError(f"Triplet image not found: {image_path}")

        components = self._read_semantic_components(image_path)
        channels = [TF.to_tensor(components[name]) for name in self.channel_names]
        x = torch.cat(channels, dim=0)
        x = self._apply_resize_mode(x)

        if self.augment:
            if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
                x = TF.hflip(x)
            if self.vertical_flip_prob > 0.0 and random.random() < self.vertical_flip_prob:
                x = TF.vflip(x)
            if self.enable_rotate_90:
                quarter_turns = random.randint(0, 3)
                if quarter_turns:
                    x = torch.rot90(x, quarter_turns, dims=[1, 2])

        if self.normalize:
            x = TF.normalize(x, self.mean, self.std)
        return x

    @staticmethod
    def _read_semantic_components(image_path: Path) -> dict[str, Image.Image]:
        image = Image.open(image_path).convert("L")
        width, height = image.size
        if width < 240 or height < 80:
            raise ValueError(f"Triplet image size is invalid: {image_path} -> {width}x{height}")

        return {
            "diff": image.crop((0, 0, 80, 80)),
            "new": image.crop((80, 0, 160, 80)),
            "old": image.crop((160, 0, 240, 80)),
        }

    def _apply_resize_mode(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize_mode == "keep":
            return x

        if self.resize_mode == "pad_resize":
            height, width = x.shape[-2:]
            side = max(height, width)
            pad_left = (side - width) // 2
            pad_right = side - width - pad_left
            pad_top = (side - height) // 2
            pad_bottom = side - height - pad_top
            x = TF.pad(x, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        if self.resize_size is None:
            raise RuntimeError("resize_size must be resolved for resize modes other than 'keep'")

        target_height, target_width = self.resize_size
        return TF.resize(x, [target_height, target_width], antialias=True)

    def labels(self) -> list[int]:
        return [int(entry["label"]) for entry in self.entries]

    def entry(self, index: int) -> dict[str, Any]:
        return dict(self.entries[index])
