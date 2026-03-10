# SCANN v2 第二轮提交清单归档

> 最后更新：2026年3月10日

本文档已从执行清单精简为归档页，只保留第二轮提交 15 到 19 的主题索引。完整总结见 `refactor_summary.md`。

---

## 1. 第二轮提交索引

15. 引入 MainWindowBuilder 并迁移 UI 构建
16. 提取动作与信号装配层
17. 删除主窗口兼容壳方法
18. 迁移主窗口残留辅助动作
19. 清理 import、更新架构文档并做终态验收

以上 5 项均已完成。

---

## 2. 阶段产出

第二轮的关键产出是把主窗口从“已拆边界”推进到“终态收口”：

- UI 构建迁入 `MainWindowBuilder`
- signal / shortcut 装配迁入 `MainWindowWiring`
- 历史 `_impl` 兼容壳方法删除
- 保存、标注、帮助等辅助动作迁入轻量 controller
- 主窗口冗余 import 清理，文档状态同步

第二轮结束后的状态：

- `main_window.py` 已可视为窗口骨架与装配入口
- 主窗口剩余方法主要是转发、少量视图控制和生命周期 glue
- 第一阶段 GUI 解耦重构正式收官

---

## 3. 后续文档

- 当前整体总结：`refactor_summary.md`
- 第二轮结果归档：`refactor_closure_plan.md`
- 当前架构说明：`architecture.md`