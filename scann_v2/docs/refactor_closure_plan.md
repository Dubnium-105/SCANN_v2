# SCANN v2 第二轮收口总结归档

> 最后更新：2026年3月10日

本文档已从收口方案精简为归档总结页，保留第二轮提交 15 到 19 的目标和结果。完整总结见 `refactor_summary.md`。

---

## 1. 第二轮范围

第二轮对应提交 15 到 19，目标是把第一轮建立的边界真正收口为终态结构。

核心目标：

- 把 UI 构建迁移到 composition 层
- 把 signal 和 shortcut wiring 迁出主窗口
- 删除历史 `_impl` 兼容壳方法
- 迁移保存、标注、帮助等残留辅助动作
- 清理主窗口冗余 import，并完成文档验收

---

## 2. 第二轮结果

提交 15 到 19 已全部完成，并完成以下结构落地：

- `MainWindowBuilder`
- `MainWindowWiring`
- 19 个 `_impl` 兼容壳方法删除
- `FileActionsController`
- `AnnotationController`
- `HelpController`
- 主窗口 direct import 收口与文档状态同步

阶段结论：

- `main_window.py` 已收敛为骨架式装配入口
- composition、controller、presenter 分层已经清晰可解释
- 第一阶段架构重构可以视为正式完成

---

## 3. 后续文档

- 当前整体总结：`refactor_summary.md`
- 第二轮提交索引：`refactor_closure_commit_checklist.md`
- 当前架构说明：`architecture.md`