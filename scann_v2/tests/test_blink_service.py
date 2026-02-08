"""闪烁服务单元测试

需求:
- 可选速度 (默认 0.5 秒)
- 切换图片时反色状态保持
"""

import pytest


class TestBlinkService:
    """测试 Blink 状态机"""

    def test_initial_state_stopped(self):
        from scann.services.blink_service import BlinkService, BlinkState

        svc = BlinkService()
        assert svc.is_running is False
        assert svc.current_state == BlinkState.NEW

    def test_toggle_starts(self):
        from scann.services.blink_service import BlinkService, BlinkState

        svc = BlinkService()
        running = svc.toggle()
        assert running is True
        assert svc.is_running is True

    def test_toggle_twice_stops(self):
        from scann.services.blink_service import BlinkService, BlinkState

        svc = BlinkService()
        svc.toggle()
        running = svc.toggle()
        assert running is False
        assert svc.is_running is False

    def test_tick_alternates(self):
        from scann.services.blink_service import BlinkService, BlinkState

        svc = BlinkService()
        svc.toggle()  # 开始

        state1 = svc.tick()
        state2 = svc.tick()

        assert state1 != state2
        assert state1 in (BlinkState.NEW, BlinkState.OLD)
        assert state2 in (BlinkState.NEW, BlinkState.OLD)

    def test_custom_speed(self):
        from scann.services.blink_service import BlinkService

        svc = BlinkService(speed_ms=250)
        assert svc.speed_ms == 250

    def test_invert_persists(self):
        """需求: 切换图片时反色状态保持"""
        from scann.services.blink_service import BlinkService

        svc = BlinkService()
        svc.toggle()  # 开始

        svc.toggle_invert()
        assert svc.is_inverted is True

        svc.tick()  # 切换到旧图
        assert svc.is_inverted is True  # 反色状态保持

        svc.tick()  # 切换回新图
        assert svc.is_inverted is True  # 仍然保持

    def test_invert_toggles(self):
        from scann.services.blink_service import BlinkService

        svc = BlinkService()
        assert svc.is_inverted is False
        svc.toggle_invert()
        assert svc.is_inverted is True
        svc.toggle_invert()
        assert svc.is_inverted is False
