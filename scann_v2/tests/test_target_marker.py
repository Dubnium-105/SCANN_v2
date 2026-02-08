"""目标标记模块单元测试

需求:
- 方框/十字线标记
- 保存文件名包含 FITS header 中的 date-time
"""

import numpy as np
import pytest


class TestTargetMarker:
    """测试目标标记功能"""

    def test_mark_on_image_returns_array(self):
        from scann.core.models import MarkerType
        from scann.ai.target_marker import mark_on_image

        img = np.zeros((128, 128), dtype=np.float32)
        marked = mark_on_image(img, x=64, y=64, marker_type=MarkerType.CROSSHAIR)
        assert isinstance(marked, np.ndarray)
        assert marked.shape[:2] == (128, 128)

    def test_mark_changes_image(self):
        from scann.core.models import MarkerType
        from scann.ai.target_marker import mark_on_image

        img = np.zeros((128, 128), dtype=np.float32)
        marked = mark_on_image(img, x=64, y=64, marker_type=MarkerType.CROSSHAIR)
        # 标记后不应与原图完全相同 (至少在标记处)
        assert not np.array_equal(img, marked)

    def test_mark_bbox(self):
        from scann.core.models import MarkerType
        from scann.ai.target_marker import mark_on_image

        img = np.zeros((128, 128), dtype=np.float32)
        marked = mark_on_image(img, x=64, y=64, marker_type=MarkerType.BOUNDING_BOX, size=20)
        assert marked is not None

    def test_generate_filename_contains_datetime(self):
        from scann.core.models import FitsHeader
        from scann.ai.target_marker import generate_marked_filename

        # 模拟 FITS header 中的日期
        header = FitsHeader(raw={"DATE-OBS": "2024-01-15T20:30:00"})
        filename = generate_marked_filename("field_001.fits", header=header)
        assert "2024" in filename
        assert "0115" in filename or "01-15" in filename
        assert filename.endswith(".fits")

    def test_generate_filename_no_date(self):
        from scann.ai.target_marker import generate_marked_filename

        filename = generate_marked_filename("field_001.fits", header=None)
        assert filename.endswith(".fits")
        assert "field_001" in filename
