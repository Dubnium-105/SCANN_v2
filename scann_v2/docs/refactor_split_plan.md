# SCANN v2 第一轮重构方案归档

> 最后更新：2026年3月10日

本文档已从执行方案精简为归档索引页，用于说明第一轮重构的原始目标和结果。完整总结见 `refactor_summary.md`，当前架构说明见 `architecture.md`。

---

## 1. 第一轮范围

第一轮对应提交 1 到 14，目标是先建立稳定边界，而不是直接做终态收口。

核心目标：

- 降低 `main_window.py` 的职责密度
- 恢复 Core → Service → GUI 的分层方向
- 把配对、图像会话、检测、查询、模型、训练、配置等流程迁到独立 controller / service
- 建立能支撑后续收口的回归测试入口

非目标：

- 不修改检测算法本身
- 不调整用户可见菜单和主工作流
- 不借机重写 UI 框架或数据模型

---

## 2. 第一轮结果

提交 1 到 14 已全部完成，并完成以下结构落地：

- `StatusPresenter`、`CandidatePresenter`
- `PairService`、`PairController`
- `ImageSessionController`
- `DetectionController` 与检测辅助模块拆分
- `QueryController` 与 `QueryService` 拆分
- `ModelService`、`ModelController`、`TrainingController`
- `ConfigService`、`PreferencesController`

阶段结论：

- 主窗口主业务入口已基本改为委托调用
- 第一轮结束时仍保留装配热点、兼容壳方法和残留辅助动作
- 后续收口工作已在提交 15 到 19 完成

---

## 3. 后续文档

- 当前整体总结：`refactor_summary.md`
- 第二轮收口归档：`refactor_closure_plan.md`
- 当前架构说明：`architecture.md`