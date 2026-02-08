"""SuspectTableWidget 可疑目标表格 单元测试

TDD 测试:
1. 空初始化 → 表格 0 行
2. set_candidates → 填充行数据 + 列内容
3. update_candidate → 单行刷新
4. 单击 → candidate_selected 信号
5. 双击 → candidate_double_clicked 信号
6. 判决显示 → REAL/BOGUS/UNKNOWN 图标
7. 已知天体 → 灰色行
8. selected_index 属性
"""

import pytest
from unittest.mock import Mock

from PyQt5.QtCore import Qt

from scann.core.models import Candidate, TargetVerdict
from scann.gui.widgets.suspect_table import SuspectTableWidget


@pytest.fixture
def table(qapp):
    """创建 SuspectTableWidget 实例"""
    return SuspectTableWidget()


@pytest.fixture
def sample_candidates():
    """生成 3 个样本候选体"""
    return [
        Candidate(x=100, y=200, ai_score=0.95),
        Candidate(x=300, y=400, ai_score=0.72),
        Candidate(x=500, y=600, ai_score=0.35, is_known=True, known_id="V1234"),
    ]


class TestSuspectTableInit:
    """测试初始化"""

    def test_empty_table(self, table):
        assert table.table.rowCount() == 0

    def test_column_count(self, table):
        assert table.table.columnCount() == 5

    def test_headers(self, table):
        headers = []
        for col in range(table.table.columnCount()):
            item = table.table.horizontalHeaderItem(col)
            headers.append(item.text() if item else "")
        assert headers == ["#", "AI 评分", "像素坐标", "WCS 坐标", "判决"]

    def test_no_candidates_initially(self, table):
        assert table._candidates == []

    def test_selected_index_negative_when_empty(self, table):
        assert table.selected_index == -1


class TestSetCandidates:
    """测试设置候选体列表"""

    def test_populates_rows(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert table.table.rowCount() == 3

    def test_index_column(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert table.table.item(0, 0).text() == "1"
        assert table.table.item(1, 0).text() == "2"
        assert table.table.item(2, 0).text() == "3"

    def test_score_column(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert "0.95" in table.table.item(0, 1).text()
        assert "0.72" in table.table.item(1, 1).text()

    def test_pixel_column(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert "(100, 200)" in table.table.item(0, 2).text()

    def test_verdict_column_default_unknown(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert "──" in table.table.item(0, 4).text()

    def test_stores_candidates_reference(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert table._candidates is sample_candidates

    def test_replace_candidates(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        assert table.table.rowCount() == 3
        table.set_candidates([Candidate(x=1, y=2)])
        assert table.table.rowCount() == 1

    def test_empty_list_clears_table(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        table.set_candidates([])
        assert table.table.rowCount() == 0


class TestUpdateCandidate:
    """测试单行更新"""

    def test_update_verdict_display(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        sample_candidates[0].verdict = TargetVerdict.REAL
        table.update_candidate(0)
        assert "✅" in table.table.item(0, 4).text()

    def test_update_bogus_display(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        sample_candidates[1].verdict = TargetVerdict.BOGUS
        table.update_candidate(1)
        assert "❌" in table.table.item(1, 4).text()

    def test_update_out_of_range_no_crash(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        table.update_candidate(99)  # 不应崩溃

    def test_update_negative_index_no_crash(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        table.update_candidate(-1)  # 不应崩溃


class TestKnownObjects:
    """测试已知天体灰色显示"""

    def test_known_object_gray_text(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        # 第 3 个候选体 is_known=True
        for col in range(table.NUM_COLS):
            item = table.table.item(2, col)
            if item:
                # 灰色 foreground
                fg = item.foreground().color()
                assert fg.red() < 150  # 灰色偏暗


class TestSignals:
    """测试信号发射"""

    def test_cell_click_emits_selected(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        received = []
        table.candidate_selected.connect(lambda idx: received.append(idx))
        table._on_cell_clicked(1, 0)
        assert received == [1]

    def test_cell_click_updates_coord_label(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        table._on_cell_clicked(0, 0)
        assert "100" in table.lbl_coord.text()
        assert "200" in table.lbl_coord.text()

    def test_cell_double_click_emits(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        received = []
        table.candidate_double_clicked.connect(lambda idx: received.append(idx))
        table._on_cell_double_clicked(2, 0)
        assert received == [2]

    def test_cell_click_out_of_range_no_emit(self, table):
        received = []
        table.candidate_selected.connect(lambda idx: received.append(idx))
        table._on_cell_clicked(5, 0)
        assert received == []


class TestCopyCoordinates:
    """测试坐标复制"""

    def test_copy_emits_signal(self, table, sample_candidates):
        table.set_candidates(sample_candidates)
        table.table.setCurrentCell(0, 0)
        received = []
        table.copy_coordinates_requested.connect(lambda idx: received.append(idx))
        table._on_copy_coord()
        assert received == [0]
