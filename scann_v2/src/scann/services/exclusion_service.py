"""已知天体排除服务

职责:
- 综合 MPCORB + 外部查询排除已知天体
"""

from __future__ import annotations

from typing import List, Optional

from scann.core.models import Candidate, FitsHeader, ObservatoryConfig


class ExclusionService:
    """已知天体排除"""

    def __init__(
        self,
        mpcorb_path: Optional[str] = None,
        observatory: Optional[ObservatoryConfig] = None,
        limit_magnitude: float = 20.0,
    ):
        self.mpcorb_path = mpcorb_path
        self.observatory = observatory or ObservatoryConfig()
        self.limit_magnitude = limit_magnitude
        self._asteroids = None

    def load_mpcorb(self) -> int:
        """加载 MPCORB 数据

        Returns:
            加载的小行星数量
        """
        if not self.mpcorb_path:
            return 0

        from scann.core.mpcorb import filter_by_magnitude, load_mpcorb

        all_asteroids = load_mpcorb(self.mpcorb_path)
        self._asteroids = filter_by_magnitude(all_asteroids, self.limit_magnitude)
        return len(self._asteroids)

    def check_candidates(
        self,
        candidates: List[Candidate],
        header: Optional[FitsHeader] = None,
    ) -> List[Candidate]:
        """检查候选体是否为已知天体

        Args:
            candidates: 候选体列表
            header: FITS 头 (用于坐标转换和时间)

        Returns:
            更新后的候选体列表 (已知天体被标记)
        """
        # TODO: 实现坐标匹配逻辑
        # 1. 从 header 获取 WCS 和观测时间
        # 2. 将候选体像素坐标转为 WCS
        # 3. 计算已知小行星在观测时刻的位置
        # 4. 匹配并标记
        return candidates
