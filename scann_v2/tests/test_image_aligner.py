"""图像对齐模块单元测试

需求: 以新图为基准, 只移动旧图。不移动新图。
"""

import numpy as np
import pytest


class TestImageAligner:
    """测试图像对齐"""

    def test_align_returns_result(self, synth_image_pair):
        from scann.core.image_aligner import align

        new_img, old_img = synth_image_pair
        result = align(new_img.astype(np.float32), old_img.astype(np.float32))
        assert result is not None
        assert result.aligned_old is not None

    def test_aligned_shape_matches(self, synth_image_pair):
        from scann.core.image_aligner import align

        new_img, old_img = synth_image_pair
        new_f = new_img.astype(np.float32)
        old_f = old_img.astype(np.float32)
        result = align(new_f, old_f)
        assert result.aligned_old.shape == new_f.shape

    def test_new_image_unchanged(self, synth_image_pair):
        """核心需求: 对齐后新图不能变"""
        from scann.core.image_aligner import align

        new_img, old_img = synth_image_pair
        new_f = new_img.astype(np.float32)
        new_copy = new_f.copy()
        old_f = old_img.astype(np.float32)

        align(new_f, old_f)

        np.testing.assert_array_equal(new_f, new_copy)

    def test_shift_vector_reasonable(self, synth_image_pair):
        from scann.core.image_aligner import align

        new_img, old_img = synth_image_pair
        result = align(new_img.astype(np.float32), old_img.astype(np.float32))
        # 已知偏移为 (3, -2)
        assert abs(result.dx) < 20
        assert abs(result.dy) < 20

    def test_align_identical_images(self, synth_fits_data_16bit):
        from scann.core.image_aligner import align

        img = synth_fits_data_16bit.astype(np.float32)
        result = align(img, img.copy())
        # 完全相同图像，偏移应接近 0
        assert abs(result.dx) < 1
        assert abs(result.dy) < 1

    def test_batch_align(self, synth_image_pair):
        from scann.core.image_aligner import batch_align

        new_img, old_img = synth_image_pair
        new_imgs = [new_img.astype(np.float32)] * 3
        old_imgs = [old_img.astype(np.float32)] * 3
        results = batch_align(new_imgs, old_imgs)
        assert len(results) == 3
        for r in results:
            assert r.aligned_old is not None
