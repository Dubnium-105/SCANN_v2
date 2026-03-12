from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


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


@dataclass
class QueryResponse:
    """结构化查询响应，兼容旧的列表式访问。"""

    results: list[QueryResult] = field(default_factory=list)
    error: str = ""

    @property
    def has_error(self) -> bool:
        return bool(self.error)

    def __iter__(self) -> Iterator[QueryResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> QueryResult:
        return self.results[index]

    def __bool__(self) -> bool:
        return bool(self.results)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return self.results == other
        if isinstance(other, QueryResponse):
            return self.results == other.results and self.error == other.error
        return NotImplemented