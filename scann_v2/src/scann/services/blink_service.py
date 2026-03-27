"""闪烁服务

职责:
- 管理新旧图闪烁状态
- 可选闪烁速度
- 反色状态持久 (切换图片不重置)
"""

from __future__ import annotations

from enum import Enum, auto


class BlinkState(Enum):
    """当前显示的图像"""
    NEW = auto()
    MARKED = auto()
    OLD = auto()


class BlinkService:
    """闪烁逻辑管理

    与 GUI 定时器解耦，仅管理状态。
    """

    def __init__(
        self,
        speed_ms: int = 500,
        sequence: tuple[BlinkState, ...] | None = None,
    ):
        self._speed_ms = speed_ms
        self._sequence = sequence or (BlinkState.NEW, BlinkState.OLD)
        if not self._sequence:
            self._sequence = (BlinkState.NEW, BlinkState.OLD)
        self._state = self._sequence[0]
        self._running = False
        self._inverted = False  # 反色状态：切换图片不重置

    @property
    def speed_ms(self) -> int:
        return self._speed_ms

    @speed_ms.setter
    def speed_ms(self, value: int) -> None:
        if value < 50:
            value = 50  # 最小 50ms
        self._speed_ms = value

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> BlinkState:
        return self._state

    @property
    def is_inverted(self) -> bool:
        """反色状态 (按下保持常开，直到再次按下才关闭)"""
        return self._inverted

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def toggle(self) -> bool:
        """切换闪烁状态

        Returns:
            切换后是否正在运行
        """
        self._running = not self._running
        return self._running

    def tick(self) -> BlinkState:
        """时钟节拍 - 切换显示状态

        由 GUI 定时器调用，每次返回应显示的图像。

        Returns:
            当前应显示的图像
        """
        if not self._running:
            return self._state

        try:
            idx = self._sequence.index(self._state)
        except ValueError:
            idx = 0

        self._state = self._sequence[(idx + 1) % len(self._sequence)]

        return self._state

    def toggle_invert(self) -> bool:
        """切换反色状态

        Returns:
            切换后的反色状态
        """
        self._inverted = not self._inverted
        return self._inverted

    def reset(self) -> None:
        """重置到初始状态 (不重置反色)"""
        self._state = self._sequence[0]
        self._running = False

    def set_state(self, state: BlinkState) -> None:
        """手动设置当前显示状态

        Args:
            state: 要设置的状态 (NEW 或 OLD)
        """
        self._state = state

    def set_sequence(self, sequence: tuple[BlinkState, ...]) -> None:
        """更新闪烁序列，并在需要时回退到首个状态。"""
        if not sequence:
            sequence = (BlinkState.NEW, BlinkState.OLD)
        self._sequence = sequence
        if self._state not in self._sequence:
            self._state = self._sequence[0]
