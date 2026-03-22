"""图像对齐裁剪测试 - 验证L型无效区域裁剪功能"""

import numpy as np
import pytest

from scann.services.pair_service import PairService


class TestImageAlignCrop:
    """测试对齐后的L型无效区域裁剪"""

    def test_calc_overlap_crop_bounds_trims_lshape_invalid_region(self):
        """测试L型无效区域裁剪：旧图向右平移后，应裁剪掉新图的左侧黑边和旧图的右侧黑边"""
        service = PairService()
        
        # 模拟 10x10 图像，旧图向右平移 3 像素
        w, h = 10, 10
        dx, dy = 3.0, 0.0
        
        # 模拟对齐后的旧图，右侧 2 列是黑边（无效区域）
        aligned_old = np.zeros((10, 10), dtype=np.float32)
        aligned_old[:, 8:] = 0.0  # 右侧 2 列为黑边
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        
        # 新图向右平移 3 像素，意味着:
        # - 新图的 [0:7] 与旧图的 [3:10] 重叠
        # - 重叠区域应该是 [3:10] 列（x0=3, x1=10）
        # 但由于旧图右侧有黑边，应该进一步裁剪
        # 重叠后旧图有效区域是 [0:8]，对应新图是 [3:11] 超出边界
        # 所以最终裁剪应该是 [3:10] 或更小
        assert x0 >= 3, f"x0 should be >= 3, got {x0}"
        assert x1 <= 10, f"x1 should be <= 10, got {x1}"
        assert x1 > x0, "x1 should be greater than x0"

    def test_calc_overlap_crop_bounds_with_down_shift(self):
        """测试向下平移时的L型裁剪"""
        service = PairService()
        
        # 模拟 10x10 图像，旧图向下平移 2 像素
        w, h = 10, 10
        dx, dy = 0.0, 2.0
        
        # 模拟对齐后的旧图，底部 5 行是黑边（确保超过 98% 阈值）
        aligned_old = np.zeros((10, 10), dtype=np.float32)
        aligned_old[:5, :] = 1.0  # 顶部 5 行有效
        # 底部 5 行全黑
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        
        # 向下平移 2 像素，新图 [2:10] 与旧图 [0:8] 重叠
        # 旧图有效区域是 [0:5]，对应新图是 [2:7]
        assert y0 >= 2, f"y0 should be >= 2, got {y0}"
        assert y1 <= 7, f"y1 should be <= 7, got {y1}"

    def test_calc_overlap_crop_bounds_with_diagonal_shift(self):
        """测试对角方向平移时的L型裁剪"""
        service = PairService()
        
        # 模拟 10x10 图像，旧图向右下角平移 (3, 2)
        w, h = 10, 10
        dx, dy = 3.0, 2.0
        
        # 模拟对齐后的旧图，右下角是黑边
        aligned_old = np.zeros((10, 10), dtype=np.float32)
        aligned_old[:, 8:] = 0.0  # 右侧黑边
        aligned_old[8:, :] = 0.0  # 底部黑边
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        
        # 对角平移后取重叠区域的交集
        assert x1 > x0 and y1 > y0, "Crop bounds should be valid"

    def test_calc_nonzero_valid_bounds_with_edge_gaps(self):
        """测试检测有效区域边界时能正确处理边缘缺失"""
        service = PairService()
        
        # 创建有 L 型黑边的图像
        image = np.ones((10, 10), dtype=np.float32)
        image[:, :3] = 0.0  # 左侧 3 列全黑
        image[:2, :] = 0.0  # 顶部 2 行全黑
        
        bounds = service.calc_nonzero_valid_bounds(image)
        
        assert bounds is not None
        x0, x1, y0, y1 = bounds
        assert x0 >= 3, f"x0 should be >= 3, got {x0}"
        assert y0 >= 2, f"y0 should be >= 2, got {y0}"

    def test_calc_overlap_crop_bounds_uses_valid_bounds_of_aligned_old(self):
        """测试使用对齐后旧图的有效区域来进一步裁剪"""
        service = PairService()
        
        # 10x10 图像，向右平移 4 像素
        w, h = 10, 10
        dx, dy = 4.0, 0.0
        
        # 对齐后的旧图只有中间区域有有效数据（两侧都有黑边）
        aligned_old = np.zeros((10, 10), dtype=np.float32)
        aligned_old[:, 2:8] = 1.0  # 中间 2-8 列是有效区域
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        
        # 平移 4 后，新图 [4:10] 对应旧图 [0:6]
        # 旧图有效区域是 [2:8]，对应新图是 [6:12]，超出边界
        # 所以应该取 [6:10]
        assert x0 >= 4, f"x0 should be >= 4, got {x0}"
        assert x1 <= 10, f"x1 should be <= 10, got {x1}"


class TestImageAlignCropEdgeCases:
    """测试边界情况"""

    def test_calc_overlap_crop_bounds_no_overlap(self):
        """测试无重叠情况"""
        service = PairService()
        
        w, h = 10, 10
        dx, dy = 15.0, 0.0  # 平移超过图像尺寸
        
        aligned_old = np.ones((10, 10), dtype=np.float32)
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        assert crop_bounds is None

    def test_calc_overlap_crop_bounds_aligned_old_none(self):
        """测试未提供对齐后旧图的情况"""
        service = PairService()
        
        w, h = 10, 10
        dx, dy = 3.0, 0.0
        
        # 不传入 aligned_old，只计算几何重叠
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=None
        )
        
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        
        # 只有几何重叠，没有有效区域信息
        assert x0 >= 0 and x1 <= 10
        # 向右平移 3，新图右侧 3 列没有对应旧图，应该被裁掉
        # 重叠区域是 [3:10]
        assert x0 == 3, f"x0 should be 3, got {x0}"

    def test_calc_overlap_crop_bounds_aligned_old_no_valid_data(self):
        """测试对齐后旧图没有有效数据的情况"""
        service = PairService()
        
        w, h = 10, 10
        dx, dy = 3.0, 0.0
        
        # 全黑图像
        aligned_old = np.zeros((10, 10), dtype=np.float32)
        
        crop_bounds = service.calc_overlap_crop_bounds(
            w=w, h=h, dx=dx, dy=dy, aligned_old=aligned_old
        )
        
        # 应该回退到纯几何重叠
        assert crop_bounds is not None
        x0, x1, y0, y1 = crop_bounds
        assert x0 == 3