"""MPC 观测报告模块单元测试

需求: 生成 MPC 标准 80 列格式观测报告
"""

from datetime import datetime

import pytest


class TestObservationReport:
    """测试 MPC 80 列格式"""

    def test_format_80col_length(self):
        from scann.core.observation_report import Observation, format_80col_line

        obs = Observation(
            designation="     K24A01A",
            discovery=False,
            obs_datetime=datetime(2024, 1, 15, 20, 30, 0),
            ra_deg=188.73658,
            dec_deg=12.58242,
            magnitude=20.5,
            mag_band="R",
            observatory_code="C42",
        )
        line = format_80col_line(obs)
        assert len(line) == 80

    def test_report_contains_observatory_code(self):
        from scann.core.observation_report import Observation, format_80col_line

        obs = Observation(
            designation="     K24A01A",
            discovery=False,
            obs_datetime=datetime(2024, 1, 15, 20, 30, 0),
            ra_deg=188.73658,
            dec_deg=12.58242,
            magnitude=20.5,
            mag_band="R",
            observatory_code="C42",
        )
        line = format_80col_line(obs)
        assert line.rstrip().endswith("C42")

    def test_generate_mpc_report_multiple(self):
        from scann.core.observation_report import (
            Observation,
            generate_mpc_report,
        )

        obs_list = [
            Observation(
                designation="     K24A01A",
                discovery=(i == 0),
                obs_datetime=datetime(2024, 1, 15 + i, 20, 30, 0),
                ra_deg=188.73658,
                dec_deg=12.58242,
                magnitude=20.5,
                mag_band="R",
                observatory_code="C42",
            )
            for i in range(3)
        ]
        report = generate_mpc_report(obs_list)
        lines = report.strip("\n").split("\n")
        # 至少包含观测行
        assert len([l for l in lines if len(l) == 80]) >= 3
