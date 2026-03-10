# SCANN v2 第一轮提交清单归档

> 最后更新：2026年3月10日

本文档已从执行清单精简为归档页，只保留第一轮提交 1 到 14 的主题索引。完整总结见 `refactor_summary.md`。

---

## 1. 第一轮提交索引

1. 建立回归保护网
2. 引入状态与候选 Presenter 骨架
3. 主窗口接入 Presenter
4. 引入 PairService 骨架
5. 引入 PairController 骨架并接线
6. 迁移配对加载逻辑到 PairController
7. 引入 ImageSessionController 并迁移显示状态
8. 引入 DetectionController 骨架
9. 拆分 DetectionPipeline 辅助模块
10. 迁移检测工作流到 DetectionController
11. 拆分 QueryService 为 utils 与 clients
12. 引入 QueryController 并迁移查询入口
13. 引入 ModelService、ModelController、TrainingController
14. 引入 ConfigService、PreferencesController 并清理主窗口

以上 14 项均已完成。

---

## 2. 阶段产出

第一轮的关键产出是把主窗口中的核心业务主链迁移到独立模块：

- presenter：状态栏与候选体展示
- service：配对、模型、配置等基础编排边界
- controller：图像会话、配对、检测、查询、模型、训练、配置入口

第一轮结束后的状态：

- 主窗口已显著变薄
- 但仍需要第二轮收口处理 UI 组装、signal wiring、兼容壳方法和残留辅助动作

---

## 3. 后续文档

- 当前整体总结：`refactor_summary.md`
- 第二轮收口归档：`refactor_closure_commit_checklist.md`
- 当前架构说明：`architecture.md`