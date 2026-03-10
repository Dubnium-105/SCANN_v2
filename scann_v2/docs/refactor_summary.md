# SCANN v2 解耦重构总结

> 最后更新：2026年3月10日

本文档是本轮 GUI 解耦重构的统一总结。原始计划与逐提交清单仍保留在对应文档中，但已压缩为归档索引页，避免重复维护。

---

## 1. 重构范围

本轮重构分为两个阶段：

- 第一轮提交 1 到 14：先拆业务边界，建立 controller、service、presenter 的基本结构
- 第二轮提交 15 到 19：完成主窗口收口，迁移 UI 组装、信号 wiring、辅助动作，并清理历史兼容层与冗余依赖

重构目标不是改功能，而是把 `main_window.py` 从“业务热点类”收敛为“窗口骨架 + 组件持有 + controller 装配 + 少量生命周期 glue code”。

---

## 2. 主要结构变化

### 2.1 主窗口职责收敛

`src/scann/gui/main_window.py` 已不再承担以下重职责：

- 菜单、中央区域、状态栏、histogram dock 的构建细节
- 大段 signal 绑定与快捷键注册
- 候选体、配对、检测、查询、训练、配置等完整业务流程
- 保存图像、标注入口、帮助入口等残留辅助动作细节
- 第一轮迁移期间保留的大批 `_impl` 兼容壳方法

主窗口当前保留的职责主要是：

- 初始化 `ui_parts`、presenter、controller、service
- 转发窗口事件到对应 controller
- 维护少量窗口级状态与生命周期入口

### 2.2 GUI 分层明确化

GUI 层现在拆分为四类职责：

- `composition/`：负责 UI 构建和 signal / shortcut 装配
- `controllers/`：负责 Qt 事件入口与业务编排委托
- `presenters/`：负责表格、marker、状态栏等展示更新
- `main_window.py`：负责装配与生命周期 glue

其中新增或落地的关键模块包括：

- `MainWindowBuilder`
- `MainWindowWiring`
- `ImageSessionController`
- `PairController`
- `DetectionController`
- `QueryController`
- `ModelController`
- `TrainingController`
- `PreferencesController`
- `FileActionsController`
- `AnnotationController`
- `HelpController`
- `CandidatePresenter`
- `StatusPresenter`

### 2.3 跨层依赖收口

重构后已完成以下收口：

- 主窗口未使用的 `core`、`ai`、widget 级 direct import 已删除
- service 层未引用 Qt widget 或 dialog
- builder / wiring 只承载组装职责，不承载业务流程
- 配对、检测、查询、模型、训练、配置等主链路已具备独立 controller 入口

---

## 3. 分阶段结果

### 第一轮：结构拆边界（提交 1 到 14）

完成内容：

- 建立主窗口核心回归保护网
- 引入 `StatusPresenter`、`CandidatePresenter`
- 引入 `PairService`、`PairController`
- 引入 `ImageSessionController`
- 引入 `DetectionController` 并拆分检测辅助模块
- 拆分 `QueryService` 并引入 `QueryController`
- 引入 `ModelService`、`ModelController`、`TrainingController`
- 引入 `ConfigService`、`PreferencesController`

阶段结果：

- 主要业务边界已从主窗口中抽离
- 主窗口开始转为委托式入口
- 但仍保留较多 UI 构建、装配热点和兼容壳方法

### 第二轮：终态收口（提交 15 到 19）

完成内容：

- 迁移菜单、中央布局、状态栏、histogram dock 到 `MainWindowBuilder`
- 迁移 signal / shortcut wiring 到 `MainWindowWiring`
- 删除 19 个历史 `_impl` 兼容壳方法
- 迁移保存图像、标注入口、帮助入口到轻量 controller
- 清理主窗口冗余 import，并同步文档验收状态

阶段结果：

- `main_window.py` 已接近目标中的骨架式形态
- 主窗口 remaining method 已基本可解释为初始化、事件转发、少量窗口级视图控制和生命周期 glue
- 第一阶段架构重构可以视为完成

---

## 4. 当前终态判断

按终态标准复核，当前结构可归纳为：

- 主窗口不再包含菜单、中央布局、状态栏、快捷键的大段构建逻辑
- 主窗口不再保留第一轮迁移期的大批兼容壳方法
- GUI 层形成 `main_window -> controller/service`、`main_window -> composition`、`main_window -> presenter` 的清晰装配关系
- 剩余 direct import 已明显收敛，且主要服务于装配或状态类型
- 测试入口已覆盖 builder、wiring、controller 和主窗口主要回归路径

---

## 5. 验证结果

第十九次提交完成后，已运行以下回归集：

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_features.py`
- `pytest scann_v2/tests/test_preferences_controller.py`
- `pytest scann_v2/tests/test_detection_controller.py`
- `pytest scann_v2/tests/test_pair_controller.py`
- `pytest scann_v2/tests/test_query_controller.py`

结果：132 项通过。

另已确认：

- `src/scann/services/` 下无 Qt 相关 import
- 旧测试中仍 patch 主窗口 `QFileDialog` 的遗留点已切换到 `scann.gui.controllers.pair_controller.QFileDialog`

---

## 6. 文档使用建议

建议按以下方式阅读：

- 想看当前架构：`architecture.md`
- 想看本轮重构总结：`refactor_summary.md`
- 想看第一轮计划归档：`refactor_split_plan.md`
- 想看第一轮提交归档：`refactor_commit_checklist.md`
- 想看第二轮收口归档：`refactor_closure_plan.md`
- 想看第二轮提交归档：`refactor_closure_commit_checklist.md`

后续如果不再需要逐提交历史，可优先维护本文件与 `architecture.md`，其余文档仅作为归档记录。