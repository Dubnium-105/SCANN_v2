"""图像处理模块单元测试"""

import numpy as np
import pytest


class TestHistogramStretch:
    """测试 histogram stretch (仅显示用, 不保存)"""

    def test_output_range_0_1(self, synth_float_image):
        from scann.core.image_processor import histogram_stretch

        result = histogram_stretch(synth_float_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self, synth_fits_data_16bit):
        from scann.core.image_processor import histogram_stretch

        result = histogram_stretch(synth_fits_data_16bit.astype(np.float32))
        assert result.dtype == np.float32

    def test_stretch_preserves_shape(self, synth_float_image):
        from scann.core.image_processor import histogram_stretch

        result = histogram_stretch(synth_float_image)
        assert result.shape == synth_float_image.shape

    def test_constant_image_does_not_crash(self):
        from scann.core.image_processor import histogram_stretch

        const = np.full((64, 64), 100.0, dtype=np.float32)
        result = histogram_stretch(const)
        assert result.shape == (64, 64)


class TestInvert:
    """测试反色 (仅显示用, 切换图片时保持反色状态)"""

    def test_invert_flips(self, synth_float_image):
        from scann.core.image_processor import invert

        result = invert(synth_float_image)
        np.testing.assert_allclose(result, 1.0 - synth_float_image, atol=1e-6)

    def test_double_invert_identity(self, synth_float_image):
        from scann.core.image_processor import invert

        result = invert(invert(synth_float_image))
        np.testing.assert_allclose(result, synth_float_image, atol=1e-6)


class TestDenoise:
    """测试降噪 (可选功能)"""

    def test_denoise_reduces_noise(self):
        from scann.core.image_processor import denoise

        rng = np.random.default_rng(42)
        noisy = rng.random((64, 64)).astype(np.float32)
        result = denoise(noisy)
        # 降噪后标准差应该减小
        assert result.std() < noisy.std()

    def test_denoise_preserves_shape(self, synth_float_image):
        from scann.core.image_processor import denoise

        result = denoise(synth_float_image)
        assert result.shape == synth_float_image.shape


class TestPseudoFlatField:
    """测试伪平场 (可选功能)"""

    def test_flat_field_output_shape(self, synth_float_image):
        from scann.core.image_processor import pseudo_flat_field

        result = pseudo_flat_field(synth_float_image)
        assert result.shape == synth_float_image.shape

    def test_flat_field_reduces_gradient(self):
        from scann.core.image_processor import pseudo_flat_field

        # 创建一个有渐变的图像
        gradient = np.linspace(0, 1, 128).astype(np.float32)
        img = np.tile(gradient, (128, 1))

        result = pseudo_flat_field(img)
        # 平场后应该更均匀
        row_means = result.mean(axis=0)
        assert row_means.std() < gradient.std()


class TestComputeStatistics:
    """测试统计信息计算"""

    def test_returns_dict(self, synth_float_image):
        from scann.core.image_processor import compute_statistics

        stats = compute_statistics(synth_float_image)
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_values_reasonable(self, synth_float_image):
        from scann.core.image_processor import compute_statistics

        stats = compute_statistics(synth_float_image)
        assert stats["min"] >= 0
        assert stats["max"] <= 1
        assert 0 <= stats["mean"] <= 1
        assert stats["std"] >= 0
