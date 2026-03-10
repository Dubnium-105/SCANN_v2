"""兼容层：保留旧 DetectionPipeline 导入路径。"""

from __future__ import annotations

from scann.core.candidate_detector import detect_candidates
from scann.core.image_aligner import align
from scann.core.models import AlignResult, Candidate
from scann.services.detection_image_adapter import img_brief_stats as _img_brief_stats
from scann.services.detection_image_adapter import robust_to_uint8 as _robust_to_uint8
from scann.services.detection_patch_extractor import MODEL_INPUT_SIZE
from scann.services.detection_pipeline import DEFAULT_PATCH_SIZE
from scann.services.detection_pipeline import DetectionPipeline as _DetectionPipeline
from scann.services.detection_pipeline import PipelineResult


class DetectionPipeline(_DetectionPipeline):
    """保留旧模块路径，内部实现委托给新 detection_pipeline 模块。"""

    def _align_images(self, new_data, old_data) -> AlignResult:
        return align(new_data, old_data)

    def _detect_candidates(self, new_data, old_data, params) -> list[Candidate]:
        return detect_candidates(new_data, old_data, params=params)


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "MODEL_INPUT_SIZE",
    "PipelineResult",
    "DetectionPipeline",
    "_img_brief_stats",
    "_robust_to_uint8",
    "align",
    "detect_candidates",
]