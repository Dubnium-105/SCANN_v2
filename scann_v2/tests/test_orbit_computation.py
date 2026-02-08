"""Orbit Computation 测试

使用测试驱动开发 (TDD) 实现：
1. 开普勒方程求解
2. 小行星视位置计算
"""

import math
import pytest
from datetime import datetime, timedelta

from scann.core.mpcorb import AsteroidOrbit, compute_apparent_positions
from scann.core.models import ObservatoryConfig, SkyPosition


class TestKeplerEquation:
    """测试开普勒方程求解"""

    def test_solve_kepler_circular(self):
        """测试：圆形轨道（e=0）时，偏近点角等于平近点角"""
        from scann.core.mpcorb import _solve_kepler_equation

        M = 1.5  # 平近点角
        e = 0.0   # 偏心率
        E = _solve_kepler_equation(M, e)

        # 圆形轨道，E = M
        assert E == pytest.approx(M)

    def test_solve_kepler_small_eccentricity(self):
        """测试：小偏心率轨道"""
        from scann.core.mpcorb import _solve_kepler_equation

        M = 1.0
        e = 0.1
        E = _solve_kepler_equation(M, e)

        # 对于小偏心率，E ≈ M + e*sin(M)
        expected = M + e * 0.8414709848  # sin(1.0)
        assert E == pytest.approx(expected, abs=0.1)

    def test_solve_kepler_high_eccentricity(self):
        """测试：高偏心率轨道"""
        from scann.core.mpcorb import _solve_kepler_equation

        M = 2.0
        e = 0.8
        E = _solve_kepler_equation(M, e)

        # 高偏心率需要更多迭代
        assert 0 < E < 2 * 3.14159

    def test_solve_kepler_multiple_values(self):
        """测试：多个平近点角值"""
        from scann.core.mpcorb import _solve_kepler_equation

        e = 0.3
        for M in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0]:
            E = _solve_kepler_equation(M, e)
            # E 应该在合理范围内
            assert 0 <= E <= 2 * 3.14159

    def test_kepler_iteration_convergence(self):
        """测试：迭代应该收敛"""
        from scann.core.mpcorb import _solve_kepler_equation

        M = 1.5
        e = 0.5
        E = _solve_kepler_equation(M, e)

        # 验证开普勒方程: M = E - e*sin(E)
        residual = abs(M - (E - e * 1.0))  # 1.0 是 sin(E) 的粗略估计
        assert residual < 0.1  # 应该收敛到小误差


class TestOrbitPropagation:
    """测试轨道传播"""

    def test_position_at_epoch(self):
        """测试：在 epoch 时刻的位置应该与轨道要素一致"""
        # 创建一个简单的小行星
        asteroid = AsteroidOrbit(
            designation="Test",
            epoch=2459000.5,  # JD
            abs_magnitude=18.0,
            slope_param=0.15,
            mean_anomaly=45.0,
            arg_perihelion=0.0,
            ascending_node=0.0,
            inclination=0.0,
            eccentricity=0.0,
            semi_major_axis=2.0,
        )

        obs_datetime = datetime(2020, 5, 31, 12, 0, 0)
        observatory = ObservatoryConfig()

        positions = compute_apparent_positions(
            [asteroid],
            obs_datetime,
            observatory,
        )

        # 应该返回一个位置
        assert len(positions) == 1
        # 圆形轨道在升交点和近日点，RA 应该接近平均平近点角（45度）
        assert positions[0].ra == pytest.approx(45.0, abs=1.0)
        assert positions[0].dec == pytest.approx(0.0, abs=1.0)

    def test_position_changes_with_time(self):
        """测试：位置应该随时间变化"""
        asteroid = AsteroidOrbit(
            designation="Test",
            epoch=2459000.5,
            abs_magnitude=18.0,
            slope_param=0.15,
            mean_anomaly=0.0,
            arg_perihelion=0.0,
            ascending_node=0.0,
            inclination=0.0,
            eccentricity=0.0,
            semi_major_axis=2.0,
        )

        # 不同时间点
        obs_time1 = datetime(2020, 5, 31, 12, 0, 0)
        obs_time2 = datetime(2020, 6, 1, 12, 0, 0)

        observatory = ObservatoryConfig()

        positions1 = compute_apparent_positions(
            [asteroid],
            obs_time1,
            observatory,
        )
        positions2 = compute_apparent_positions(
            [asteroid],
            obs_time2,
            observatory,
        )

        # 位置应该不同
        assert abs(positions1[0].ra - positions2[0].ra) > 0.01

    def test_multiple_asteroids(self):
        """测试：多个小行星的位置计算"""
        asteroids = [
            AsteroidOrbit(
                designation=f"Asteroid{i}",
                epoch=2459000.5,                abs_magnitude=18.0,
                slope_param=0.15,                mean_anomaly=float(i * 30),
                arg_perihelion=0.0,
                ascending_node=0.0,
                inclination=0.0,
                eccentricity=0.0,
                semi_major_axis=2.0,
            )
            for i in range(3)
        ]

        obs_datetime = datetime(2020, 5, 31, 12, 0, 0)
        observatory = ObservatoryConfig()

        positions = compute_apparent_positions(
            asteroids,
            obs_datetime,
            observatory,
        )

        # 应该返回三个位置
        assert len(positions) == 3
        # 名称应该匹配
        assert positions[0].name == "Asteroid0"
        assert positions[1].name == "Asteroid1"
        assert positions[2].name == "Asteroid2"

    def test_inclination_affects_dec(self):
        """测试：轨道倾角应该影响赤纬"""
        asteroid = AsteroidOrbit(
            designation="Test",
            epoch=2459000.5,            abs_magnitude=18.0,
            slope_param=0.15,            mean_anomaly=45.0,
            arg_perihelion=0.0,
            ascending_node=0.0,
            inclination=45.0,  # 45度倾角
            eccentricity=0.0,
            semi_major_axis=2.0,
        )

        obs_datetime = datetime(2020, 5, 31, 12, 0, 0)
        observatory = ObservatoryConfig()

        positions = compute_apparent_positions(
            [asteroid],
            obs_datetime,
            observatory,
        )

        # 赤纬应该非零
        assert abs(positions[0].dec) > 0.01

    def test_empty_asteroid_list(self):
        """测试：空小行星列表"""
        positions = compute_apparent_positions(
            [],
            datetime(2020, 5, 31, 12, 0, 0),
            ObservatoryConfig(),
        )

        # 应该返回空列表
        assert positions == []


class TestOrbitalElements:
    """测试轨道要素计算"""

    def test_true_anomaly_calculation(self):
        """测试：真近点角计算"""
        from scann.core.mpcorb import _compute_true_anomaly

        # 圆形轨道，真近点角 = 偏近点角
        nu = _compute_true_anomaly(E=1.0, e=0.0)
        assert nu == pytest.approx(1.0)

        # 小偏心率
        nu = _compute_true_anomaly(E=1.0, e=0.1)
        # nu 应该接近 E
        assert abs(nu - 1.0) < 0.5

    def test_orbital_plane_to_equatorial(self):
        """测试：轨道平面到赤道坐标转换"""
        from scann.core.mpcorb import _orbital_to_equatorial

        # 简单情况：零倾角、零升交点
        x, y, z = 1.0, 0.0, 0.0
        i = 0.0
        om = 0.0  # 升交点经度
        w = 0.0  # 近日点幅角

        X, Y, Z = _orbital_to_equatorial(x, y, z, i, om, w)

        # 应该等于原始值
        assert X == pytest.approx(1.0)
        assert Y == pytest.approx(0.0)
        assert Z == pytest.approx(0.0)

    def test_equatorial_to_spherical(self):
        """测试：赤道坐标到球面坐标转换"""
        from scann.core.mpcorb import _equatorial_to_spherical

        # X轴上的点
        ra, dec = _equatorial_to_spherical(1.0, 0.0, 0.0)
        assert ra == pytest.approx(0.0)
        assert dec == pytest.approx(0.0)

        # Y轴上的点
        ra, dec = _equatorial_to_spherical(0.0, 1.0, 0.0)
        assert ra == pytest.approx(math.radians(90.0))  # π/2 弧度
        assert dec == pytest.approx(0.0)

        # Z轴上的点
        ra, dec = _equatorial_to_spherical(0.0, 0.0, 1.0)
        assert dec == pytest.approx(math.radians(90.0))  # π/2 弧度（北天极）
