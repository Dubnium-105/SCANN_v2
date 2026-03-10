# SCANN v2 重构终态差距修正方案

> 最后更新：2026年3月10日

> 适用背景：原 1 到 14 次提交已完成第一轮控制器与服务边界拆分，但距离 `refactor_split_plan.md` 中定义的终态标准仍存在收口差距。本文件用于指导第二轮“收口式解耦”。

---

## 1. 当前判断

按终态标准验收，当前项目状态应定义为：

- 配置、模型、训练、查询、检测、配对、图像会话的主要边界已经建立
- 主窗口中的大部分业务入口已经改为委托 controller
- 但 `main_window.py` 仍未达到“窗口骨架 + 组件持有 + controller 装配 + 少量生命周期 glue code”的目标形态

这意味着：

- 第一轮重构已经完成“拆边界”
- 第二轮重构需要完成“清兼容层、清装配热点、清残留直接依赖”

---

## 2. 与终态标准的主要差距

### 2.1 主窗口仍然过重

当前 `src/scann/gui/main_window.py` 仍保留以下大块职责：

- 菜单栏与动作创建
- 中央区域与状态栏组装
- 直方图 dock 与快捷键初始化
- 图像保存与标记图导出
- 标注工具入口、帮助入口、部分窗口级视图控制

这与终态定义中的以下要求不一致：

- 主窗口只保留窗口骨架、组件持有和 controller 初始化
- GUI 组装职责应进入 `gui/composition/`

### 2.2 兼容壳方法仍大量存在

当前主窗口中仍保留大量 `_xxx_impl` 兼容入口，例如：

- 候选体标记与导航兼容入口
- 配对切换兼容入口
- 批量处理与检测兼容入口
- 最近目录与配对选择兼容入口

这些方法在第一轮迁移期是合理的，但在终态阶段会带来两个问题：

- 主窗口噪声过高，难以辨别真实职责
- 未来继续改 controller 时，容易保留多余调用路径

### 2.3 直接 import 仍未收口

当前主窗口仍直接依赖多个 `core`、`data`、`ai` 层对象。虽然部分 import 是为了装配 service，但也存在明显的历史残留。终态要求不是“零 import”，而是：

- 主窗口不再直接持有多数跨层实现细节
- 主窗口 import 应服务于装配，而不是业务执行

因此需要把直接 import 分成两类处理：

- 必要装配依赖：保留或迁移到 composition/builder
- 历史兼容残留：删除

### 2.4 composition 层已完成第一阶段落地

当前 `gui/composition/` 已新增并投入使用：

- `main_window_builder.py`
- `main_window_wiring.py`

这意味着 `MainWindow` 已不再承担大段 widget 创建、signal 绑定和快捷键注册代码，但仍需要继续完成：

- 历史兼容壳方法清理
- 残留辅助动作收束
- direct import 收口

### 2.5 主窗口中仍有零散流程性职责

以下职责即使不属于“核心业务主链”，也会持续阻止主窗口收口：

- `_on_save_image`
- `_on_save_marked_image`
- `_on_open_annotation`
- 帮助/文档/关于等窗口级动作处理
- 分割器宽度持久化与局部窗口视图同步

这些职责需要明确归位，否则主窗口会长期处于“已拆一半”的状态。

---

## 3. 第二轮重构目标

本轮不再新增业务能力，只完成结构收口。

### 3.1 结构目标

- 将主窗口中的 UI 构建迁移到 composition 层
- 删除绝大部分 `_impl` 兼容壳方法
- 清理无效 direct import
- 将残留的窗口级动作按职责收束到 controller 或轻量 helper
- 将 `main_window.py` 压缩到“可读、可审查、可定位”的组装器形态

### 3.2 完成标准

当满足以下条件时，可认为终态标准真正达成：

1. `main_window.py` 不再包含菜单、中央布局、状态栏、快捷键的大段构建细节
2. `main_window.py` 中不再保留批量 `_impl` 兼容入口，除极少数测试兼容 glue 外
3. `main_window.py` 中绝大多数方法为：初始化、信号连接、事件转发、生命周期 glue
4. 主窗口 direct import 仅保留装配必需项，删除历史遗留未使用依赖
5. 新增或更新测试能够覆盖 builder 装配、动作绑定、兼容层删除后的主要回归点

---

## 4. 建议的修正路径

建议将第二轮收口拆为 5 次提交，每次只处理一种结构问题。

### 提交 A：引入 MainWindowBuilder，迁移 UI 构建

状态：已完成（2026年3月10日）

目标：

- 新增 `src/scann/gui/composition/main_window_builder.py`
- 迁移 `_init_menu_bar`
- 迁移 `_init_central_ui`
- 迁移 `_init_status_bar`
- 迁移 `_init_histogram_dock`
- 让主窗口只保留 builder 调用与结果持有

边界要求：

- builder 负责创建控件和动作
- builder 不负责业务逻辑
- controller 初始化仍留在主窗口

### 提交 B：提取动作与信号装配层

状态：已完成（2026年3月10日）

目标：

- 将 `_connect_signals` 与 `_init_shortcuts` 从主窗口迁出
- 可选新增 `src/scann/gui/composition/action_factory.py`
- 明确 action 创建、快捷键绑定、signal wiring 的归属

边界要求：

- 不在本提交改业务流程
- 不在本提交删除兼容入口

完成记录：

- 已新增 `src/scann/gui/composition/main_window_wiring.py`
- 已将 `_connect_signals` 与 `_init_shortcuts` 从 `main_window.py` 迁出
- 已把退出动作的连接一并收口到 wiring 层，builder 仅保留 QAction 创建
- 已新增 `tests/test_main_window_wiring.py` 覆盖代表性的动作绑定、signal 装配和窗口级快捷键注册

### 提交 C：清理兼容壳方法

状态：已完成（2026年3月10日）

目标：

- 删除已无必要的 `_xxx_impl` 兼容方法
- 将测试、信号连接、外部调用点全部切换到正式入口
- 保证一项交互只保留一条调用路径

重点对象：

- 候选体相关兼容入口
- 配对相关兼容入口
- 检测批处理相关兼容入口
- 最近目录与配对选择兼容入口

完成记录：

- 已删除 `src/scann/gui/main_window.py` 中第十七次提交清单列出的 19 个 `_impl` 兼容壳方法
- 已确认 `main_window_wiring.py`、`DetectionController`、`PairController` 与现有测试均已使用正式入口，无需保留历史别名
- 已在 `tests/test_main_window.py` 增加回归测试，确保这些兼容入口不会被重新引入

### 提交 D：迁移主窗口残留辅助动作

状态：已完成（2026年3月10日）

目标：

- 把图像保存、标记图导出、标注工具入口、帮助菜单入口从主窗口中进一步收束
- 允许以轻量 controller 或 helper 的方式处理，不要求为了拆分而过度抽象

推荐归位：

- 保存相关动作可进入轻量 `file_actions_controller.py` 或现有 Pair/Preferences 相关 helper
- 标注工具入口可单独封装为 `annotation_controller.py` 或保持为单一 helper
- 帮助/关于入口可进入轻量 `help_controller.py`

完成记录：

- 已新增 `src/scann/gui/controllers/file_actions_controller.py`，承接 `_on_save_image` 与 `_on_save_marked_image` 的保存流程
- 已新增 `src/scann/gui/controllers/annotation_controller.py`，承接标注对话框创建与非模态打开逻辑
- 已新增 `src/scann/gui/controllers/help_controller.py`，承接快捷键帮助、文档、关于与计划任务占位入口
- `src/scann/gui/main_window.py` 已收敛为对应动作的薄转发入口，不再直接处理这些辅助动作细节
- 已扩展 `tests/test_main_window_features.py` 覆盖保存、标注、帮助入口的主要回归点

### 提交 E：清理 import 与文档验收

目标：

- 清理主窗口未使用的 `core`、`data`、`ai` import
- 对照终态标准重新更新 `architecture.md`
- 补一轮人工检查结论
- 将文档中的“已完成”状态与实际结构一致化

---

## 5. 实施原则

第二轮修正必须遵守以下原则：

1. 不借收口机会顺手改算法、阈值、文案、交互
2. 不用“大删除后慢慢修补”的方式推进
3. 每次提交都应保留最小可运行状态
4. builder/composition 层不承载业务流程，只承载组装与 wiring
5. service 层仍不得引用 Qt widget 或 dialog

---

## 6. 风险与控制措施

## 风险 1：builder 迁移引发 UI 初始化缺失

控制措施：

- 为菜单动作、关键 widget、状态栏组件增加存在性测试
- builder 返回结构化引用对象，避免主窗口靠字符串属性猜测

## 风险 2：删除兼容壳方法后测试和连接断裂

控制措施：

- 先统计调用点，再删壳方法
- 删除前先让测试和 signal 改用正式入口

## 风险 3：过度拆分导致新模块碎片化

控制措施：

- 只有职责边界足够清晰时才新建 controller
- 对帮助菜单、保存动作这类轻量功能，优先使用 helper 或小 controller，而不是再造复杂层级

---

## 7. 建议的验收问题

每完成一次提交，都要回答以下问题：

1. 主窗口是否真的变短、变清晰，而不是只把代码平移后继续保留壳方法？
2. 是否减少了调用路径数量？
3. 是否减少了主窗口对 `core`、`data`、`ai` 的直接依赖？
4. 是否把新的结构责任放在正确层，而不是把业务逻辑塞进 builder？
5. 是否有对应测试覆盖本次拆分的主要回归风险？

---

## 8. 对现有文档的关系说明

本文件不是替代：

- `refactor_split_plan.md`
- `refactor_commit_checklist.md`

本文件是它们在第一轮 14 次提交完成后的“终态收口补充方案”。

对应逐提交执行清单见：

- `docs/refactor_closure_commit_checklist.md`