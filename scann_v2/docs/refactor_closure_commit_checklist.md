# SCANN v2 重构终态收口逐提交清单

> 最后更新：2026年3月10日

> 对应方案文档：`refactor_closure_plan.md`

本文档用于承接第一轮 14 次提交之后的“终态收口”工作。目标不是继续扩展功能，而是按终态标准完成最后一轮解耦、清理和文档验收。

---

## 1. 使用方式

执行规则：

1. 每次提交只完成一个结构主题
2. 不在收口提交里改算法、阈值、UI 文案或数据模型
3. 每次提交结束后运行本项最小测试集
4. 如果本次提交没有让 `main_window.py` 更清晰，就说明拆分方向可能不对
5. 删除兼容壳方法前，必须先切换所有调用点和测试

---

## 2. 提交总览

建议总计 5 次提交完成终态收口：

15. 引入 MainWindowBuilder 并迁移 UI 构建
16. 提取动作与信号装配层
17. 删除主窗口兼容壳方法
18. 迁移主窗口残留辅助动作
19. 清理 import、更新架构文档并做终态验收

---

## 3. 逐提交清单

## 提交 15：引入 MainWindowBuilder 并迁移 UI 构建

### 当前状态

- 已完成（2026年3月10日）

### 目标

把主窗口中最重的 UI 构建代码迁到 composition 层，为最终收口创造空间。

### 修改内容

- 新增：
  - `src/scann/gui/composition/main_window_builder.py`
  - 可选 `src/scann/gui/composition/__init__.py`
- 修改：
  - `src/scann/gui/main_window.py`
- 可选新增测试：
  - `tests/test_main_window_builder.py`

### 迁移方法

- `_init_menu_bar`
- `_init_central_ui`
- `_init_status_bar`
- `_init_histogram_dock`

### 本提交不做的事

- 不迁移 signal 绑定
- 不删除兼容壳方法
- 不调整 controller 边界

### 完成标准

- 主窗口不再直接持有大段控件创建代码
- builder 能返回主窗口后续装配所需引用

### 完成记录

- 已新增 `src/scann/gui/composition/main_window_builder.py` 与 `src/scann/gui/composition/__init__.py`
- 已将 `_init_menu_bar`、`_init_central_ui`、`_init_status_bar`、`_init_histogram_dock` 的构建逻辑迁入 builder
- `MainWindow` 已改为持有 builder 返回的 `ui_parts` 并统一挂载属性引用
- 已新增 `tests/test_main_window_builder.py` 覆盖 builder 构建与属性挂载回归点

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_builder.py`

---

## 提交 16：提取动作与信号装配层

### 当前状态

- 已完成（2026年3月10日）

### 目标

把动作创建、signal 绑定和快捷键装配从主窗口中抽离，降低窗口类的噪声。

### 修改内容

- 新增：
  - `src/scann/gui/composition/action_factory.py`
  - 或 `src/scann/gui/composition/main_window_wiring.py`
- 修改：
  - `src/scann/gui/main_window.py`
  - `src/scann/gui/composition/main_window_builder.py`
- 可选新增测试：
  - `tests/test_main_window_wiring.py`

### 迁移方法

- `_connect_signals`
- `_init_shortcuts`
- 各类 QAction 创建与 wiring

### 本提交不做的事

- 不删除 `_impl` 兼容方法
- 不迁移保存图像或帮助动作的实际处理逻辑

### 完成标准

- 主窗口中不再保留成片信号绑定代码
- 动作创建和绑定方式可独立测试

### 完成记录

- 已新增 `src/scann/gui/composition/main_window_wiring.py`
- `src/scann/gui/main_window.py` 已改为通过 wiring 层完成 signal 绑定与快捷键装配
- builder 中的退出 QAction 连接已迁出，`main_window_builder.py` 仅保留静态 QAction 创建
- 已新增 `tests/test_main_window_wiring.py`，覆盖控件信号、菜单动作、查询动作和窗口级快捷键注册

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_wiring.py`

---

## 提交 17：删除主窗口兼容壳方法

### 当前状态

- 已完成（2026年3月10日）

### 目标

去掉第一轮迁移期保留的历史过渡入口，确保每个功能只有一个正式调用路径。

### 修改内容

- 修改：
  - `src/scann/gui/main_window.py`
  - 涉及 signal 绑定或测试引用的 controller / 测试文件

### 重点删除对象

- `_on_mark_real_impl`
- `_on_mark_bogus_impl`
- `_on_next_candidate_impl`
- `_on_candidate_selected_impl`
- `_on_candidate_double_clicked_impl`
- `_focus_candidate_impl`
- `_prev_pair_impl`
- `_next_pair_impl`
- `_open_new_folder_impl`
- `_add_recent_folder_impl`
- `_open_old_folder_impl`
- `_update_recent_menu_impl`
- `_open_recent_folder_impl`
- `_on_batch_align_impl`
- `_on_batch_process_impl`
- `_run_batch_process_impl`
- `_build_detection_params_impl`
- `_on_batch_detect_impl`
- `_select_pair_impl`

### 本提交不做的事

- 不引入新的 builder/controller
- 不顺手改业务逻辑

### 完成标准

- 主窗口的兼容壳方法显著减少或清零
- 所有调用点都改为正式入口

### 完成记录

- 已从 `src/scann/gui/main_window.py` 删除 `_on_mark_real_impl` 等 19 个 `_impl` 兼容壳方法
- 已确认 signal wiring、DetectionController、PairController 与现有调用点均切换为正式入口
- 已在 `tests/test_main_window.py` 新增回归测试，确保历史兼容壳方法不会重新出现

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_features.py`
- `pytest scann_v2/tests/test_detection_controller.py`
- `pytest scann_v2/tests/test_pair_controller.py`

---

## 提交 18：迁移主窗口残留辅助动作

### 当前状态

- 已完成（2026年3月10日）

### 目标

将剩余不属于主窗口骨架的辅助动作继续收束，避免长期残留在窗口类中。

### 修改内容

- 新增：
  - 可选 `src/scann/gui/controllers/help_controller.py`
  - 可选 `src/scann/gui/controllers/file_actions_controller.py`
  - 可选 `src/scann/gui/controllers/annotation_controller.py`
- 修改：
  - `src/scann/gui/main_window.py`

### 迁移对象

- `_on_save_image`
- `_on_save_marked_image`
- `_on_open_annotation`
- `_on_open_docs`
- `_on_about`
- 视情况处理 `_on_open_scheduler`

### 本提交不做的事

- 不改设置、检测、查询、配对主链
- 不强行把简单帮助动作拆成复杂架构

### 完成标准

- 主窗口只保留极少数真正依赖 QMainWindow 生命周期的动作
- 辅助动作职责有清晰归位

### 完成记录

- 已新增 `src/scann/gui/controllers/file_actions_controller.py`，迁移图像保存与标记图导出逻辑
- 已新增 `src/scann/gui/controllers/annotation_controller.py`，迁移标注工具对话框入口
- 已新增 `src/scann/gui/controllers/help_controller.py`，迁移快捷键帮助、文档、关于与计划任务占位入口
- `src/scann/gui/main_window.py` 中对应方法已改为转发到轻量 controller，窗口类不再直接执行这些辅助动作流程
- 已更新 `tests/test_main_window_features.py`，覆盖保存图像、标注入口、帮助文档、关于和计划任务占位提示回归点

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_features.py -k "save or annotation or about or docs"`

---

## 提交 19：清理 import、更新架构文档并做终态验收

### 当前状态

- 未开始

### 目标

完成最后一轮无效依赖清理、架构文档同步与终态验收，避免代码状态和文档状态再次偏离。

### 修改内容

- 修改：
  - `src/scann/gui/main_window.py`
  - `docs/architecture.md`
  - `docs/refactor_split_plan.md`
  - `docs/refactor_commit_checklist.md`
  - `docs/refactor_closure_plan.md`
  - `docs/refactor_closure_commit_checklist.md`

### 核查对象

- 删除主窗口中无效 direct import
- 确认 service 层未引用 Qt
- 确认 builder 不承载业务逻辑
- 确认主窗口已接近终态结构

### 完成标准

- 文档中的“已完成”与实际结构一致
- `main_window.py` 的 remaining import 和 remaining method 可以自洽地解释为装配或生命周期 glue
- 本轮收口后可正式宣布第一阶段架构重构结束

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_features.py`
- `pytest scann_v2/tests/test_preferences_controller.py`
- `pytest scann_v2/tests/test_detection_controller.py`
- `pytest scann_v2/tests/test_pair_controller.py`
- `pytest scann_v2/tests/test_query_controller.py`

---

## 4. 每次提交后的统一检查

每一步提交后，除了本项最小测试集，还建议做以下检查：

1. `main_window.py` 是否真的更短、更清晰
2. 是否删掉了旧调用路径，而不是又叠加一层过渡代码
3. 是否把组装逻辑和业务逻辑区分清楚
4. 是否减少了主窗口对 `core`、`data`、`ai` 的直接依赖
5. 文档状态是否仍然与代码现实一致

---

## 5. 人工验收短清单

完成全部收口提交后，至少人工验证以下流程：

1. 启动后窗口布局、菜单、状态栏、快捷键都正常可用
2. 打开新旧图文件夹、切换配对、图像显示和候选体联动正常
3. 闪烁、反色、拉伸、缩放行为正常
4. 查询、MPC 报告、右键菜单动作正常
5. 模型加载、检测、训练入口正常
6. 首选项保存、退出重开恢复状态正常
7. 保存图像、标记图、标注工具入口正常