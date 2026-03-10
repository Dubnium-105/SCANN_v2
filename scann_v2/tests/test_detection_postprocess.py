from scann.core.models import Candidate
from scann.services.detection_postprocess import nms_candidates


def test_nms_candidates_keeps_highest_score_within_min_distance():
    candidates = [
        Candidate(x=10, y=10, ai_score=0.95),
        Candidate(x=12, y=11, ai_score=0.62),
        Candidate(x=40, y=40, ai_score=0.83),
    ]

    kept = nms_candidates(candidates, min_dist=5)

    assert [(candidate.x, candidate.y, candidate.ai_score) for candidate in kept] == [
        (10, 10, 0.95),
        (40, 40, 0.83),
    ]


def test_nms_candidates_keeps_separated_candidates():
    candidates = [
        Candidate(x=5, y=5, ai_score=0.4),
        Candidate(x=25, y=25, ai_score=0.7),
        Candidate(x=50, y=50, ai_score=0.6),
    ]

    kept = nms_candidates(candidates, min_dist=10)

    assert [(candidate.x, candidate.y) for candidate in kept] == [
        (25, 25),
        (50, 50),
        (5, 5),
    ]