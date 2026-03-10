from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    """查询结果。"""

    source: str
    name: str
    object_type: str
    distance_arcsec: float = 0.0
    magnitude: float = 0.0
    url: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)