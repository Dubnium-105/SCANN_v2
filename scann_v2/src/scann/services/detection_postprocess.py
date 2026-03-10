"""Detection post-processing helpers."""

from __future__ import annotations

from typing import List

from scann.core.models import Candidate


def nms_candidates(
    candidates: List[Candidate],
    min_dist: int,
) -> List[Candidate]:
    """对候选体做简单的空间 NMS。"""
    ordered = sorted(
        candidates,
        key=lambda candidate: candidate.ai_score,
        reverse=True,
    )
    keep: List[Candidate] = []
    for candidate in ordered:
        too_close = False
        for kept in keep:
            dist = ((candidate.x - kept.x) ** 2 + (candidate.y - kept.y) ** 2) ** 0.5
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            keep.append(candidate)
    return keep