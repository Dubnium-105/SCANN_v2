# SCANN v2 重构逐提交清单

> 最后更新：2026年3月10日

本文档是 [refactor_split_plan.md](refactor_split_plan.md) 的落地版执行清单，按“每次提交只做一件事”的原则组织。目标是让重构过程具备以下特点：

- 每次提交范围可控
- 每次提交后都能运行验证
- 任意一步出问题都容易回滚
- 主窗口复杂度持续下降，而不是先升后降

---

## 1. 使用方式

执行规则：

1. 严格按顺序推进，除非前一步已稳定且验证通过
2. 每次提交只覆盖本清单中的一个提交项
3. 每次提交完成后，至少运行该项列出的最小测试集
4. 如果某一步引入大量兼容胶水代码但未立即消化，不要继续下一步
5. 任何提交都不得同时改动主窗口、服务拆分、查询拆分、配置拆分这四类大主题中的两个以上

提交命名建议：

- `refactor(gui): introduce status and candidate presenters`
- `refactor(gui): extract pair service and controller`
- `refactor(services): split detection helpers from pipeline`

---

## 2. 提交总览

建议总计 14 次提交完成第一轮结构重构。

1. 建立回归保护网（已完成）
2. 引入状态与候选 Presenter 骨架（已完成）
3. 主窗口接入 Presenter（已完成）
4. 引入 PairService 骨架（已完成）
5. 引入 PairController 骨架并接线（已完成）
6. 迁移配对加载逻辑到 PairController（已完成）
7. 引入 ImageSessionController 并迁移显示状态（已完成）
8. 引入 DetectionController 骨架（已完成）
9. 拆分 DetectionPipeline 辅助模块（已完成）
10. 迁移检测工作流到 DetectionController（已完成）
11. 拆分 QueryService 为 utils 与 clients（已完成）
12. 引入 QueryController 并迁移查询入口（已完成）
13. 引入 ModelService、ModelController、TrainingController（已完成）
14. 引入 ConfigService、PreferencesController 并清理主窗口（已完成）

如果希望风险更低，可以把第 13 和第 14 再各自拆成两次提交。

---

## 3. 逐提交清单

## 提交 1：建立回归保护网

### 当前状态

- 已完成

### 目标

先把最关键行为固定住，避免后续“结构变好了但行为悄悄变了”。

### 修改内容

- 新增以下测试文件：
  - `tests/test_main_window_pair_flow.py`
  - `tests/test_main_window_query_flow.py`
  - `tests/test_detection_pipeline_regression.py`
  - `tests/test_query_service_regression.py`
- 如现有 fixture 不够用，补充 `tests/conftest.py`

### 本提交不做的事

- 不拆任何生产代码
- 不改 `main_window.py` 逻辑
- 不改服务层结构

### 完成标准

- 新增测试能覆盖以下行为：
  - 打开新旧图文件夹与配对切换
  - 右键查询与菜单查询入口
  - `DetectionPipeline.process_pair()` 的关键输出
  - `QueryService` 对外接口的关键返回格式

### 建议运行

- `pytest scann_v2/tests/test_main_window_pair_flow.py`
- `pytest scann_v2/tests/test_main_window_query_flow.py`
- `pytest scann_v2/tests/test_detection_pipeline_regression.py`
- `pytest scann_v2/tests/test_query_service_regression.py`

### 风险提示

- 若这一步做得过轻，后续重构无法判断是否回归

---

## 提交 2：引入状态与候选 Presenter 骨架

### 当前状态

- 已完成

### 目标

先把展示层职责从主窗口中抽出，但先不迁移所有调用点。

### 修改内容

- 新增：
  - `src/scann/gui/presenters/status_presenter.py`
  - `src/scann/gui/presenters/candidate_presenter.py`
  - 可选 `src/scann/gui/presenters/__init__.py`
- 在 presenter 中定义最小接口：
  - `StatusPresenter.show_message(...)`
  - `CandidatePresenter.set_candidates(...)`
  - `CandidatePresenter.refresh_markers(...)`

### 本提交不做的事

- 不删除主窗口原有方法
- 不迁移复杂业务逻辑

### 完成标准

- 新文件创建完成
- 接口命名稳定
- 不影响现有行为

### 建议运行

- `pytest scann_v2/tests/test_main_window.py`
- `pytest scann_v2/tests/test_main_window_features.py -k "logger or suspect or marker"`

### 备注

- 这是纯铺路提交，允许功能零变化

---

## 提交 3：主窗口接入 Presenter

### 当前状态

- 已完成

### 目标

用 presenter 接管最简单的展示职责，减少主窗口直接操作 UI 细节的面积。

### 修改内容

- 修改：
  - `src/scann/gui/main_window.py`
- 可能补充：
  - `tests/test_main_window_features.py`

### 迁移点

- `_show_message` 改为委托 `StatusPresenter`
- `set_candidates` 的表格刷新和 marker 刷新改为部分委托 `CandidatePresenter`
- `_update_markers` 优先转成 presenter 调用

### 本提交不做的事

- 不改变候选体数据结构
- 不调整候选选择逻辑归属

### 完成标准

- 主窗口中展示更新的直接实现减少
- 现有交互行为无变化

### 建议运行

- `pytest scann_v2/tests/test_main_window_features.py -k "candidate or marker or status"`
- `pytest scann_v2/tests/test_suspect_table.py`

---

## 提交 4：引入 PairService 骨架

### 当前状态

- 已完成

### 目标

先建立文件配对的服务边界，再迁移控制逻辑。

### 修改内容

- 新增：
  - `src/scann/services/pair_service.py`
- 可选新增：
  - `tests/test_pair_service.py`

### 最小接口

- `scan_new_folder(folder)`
- `scan_old_folder(folder)`
- `match_pairs(new_folder, old_folder)`
- `load_pair(pair)`
- `resolve_pair_image_paths(pair)`

### 本提交不做的事

- 不让主窗口实际使用该服务
- 不迁移事件处理代码

### 完成标准

- PairService 可以包住当前 `file_manager` 和 `fits_io` 的组合调用

### 建议运行

- `pytest scann_v2/tests/test_file_manager.py`
- `pytest scann_v2/tests/test_fits_io.py`
- `pytest scann_v2/tests/test_pair_service.py`

---

## 提交 5：引入 PairController 骨架并接线

### 当前状态

- 已完成

### 目标

把主窗口中文件夹和配对相关事件的入口先集中起来，但先不迁移所有内部逻辑。

### 修改内容

- 新增：
  - `src/scann/gui/controllers/pair_controller.py`
  - 可选 `src/scann/gui/controllers/__init__.py`
- 修改：
  - `src/scann/gui/main_window.py`

### 迁移方式

- 主窗口保留原槽函数名称
- 槽函数内部改为一行委托 `self.pair_controller.xxx(...)`

### 本提交不做的事

- 不迁移复杂配对逻辑本体
- 不迁移图像显示逻辑

### 完成标准

- 打开新旧文件夹、上一对/下一对、列表选择等入口均经过 controller

### 建议运行

- `pytest scann_v2/tests/test_main_window_pair_flow.py`
- `pytest scann_v2/tests/test_main_window.py -k "pair or open"`

---

## 提交 6：迁移配对加载逻辑到 PairController

### 当前状态

- 已完成

### 目标

把配对扫描、配对切换、路径解析、对齐产物判断从主窗口中实际移出。

### 修改内容

- 修改：
  - `src/scann/gui/main_window.py`
  - `src/scann/gui/controllers/pair_controller.py`
  - `src/scann/services/pair_service.py`
- 可选新增：
  - `src/scann/gui/presenters/image_presenter.py`
  - `tests/test_pair_controller.py`

### 迁移方法

- `_on_open_new_folder`
- `_on_open_old_folder`
- `_add_recent_folder`
- `_on_update_recent_menu`
- `_open_recent_folder`
- `_on_prev_pair`
- `_on_next_pair`
- `_load_pair`
- `_aligned_artifact_paths`
- `_pair_has_aligned_artifacts`
- `_resolve_pair_image_paths`
- `_calc_nonzero_valid_bounds`
- `_calc_overlap_crop_bounds`
- `_on_pair_selected`

### 完成标准

- 主窗口不再持有配对加载细节实现
- 配对列表与当前图像仍然正确同步

### 建议运行

- `pytest scann_v2/tests/test_main_window_pair_flow.py`
- `pytest scann_v2/tests/test_pair_controller.py`
- `pytest scann_v2/tests/test_file_manager.py`

### 风险提示

- 这一步开始容易出现状态归属混乱，必须明确当前 pair 状态由谁维护

---

## 提交 7：引入 ImageSessionController 并迁移显示状态

### 当前状态

- 已完成

### 目标

把闪烁、反色、拉伸、当前图像显示等视图状态从主窗口剥离出去。

### 修改内容

- 新增：
  - `src/scann/gui/controllers/image_session_controller.py`
  - 可选 `tests/test_image_session_controller.py`
- 修改：
  - `src/scann/gui/main_window.py`

### 迁移方法

- `_on_blink_toggle`
- `_on_blink_tick`
- `_on_blink_speed_changed`
- `_on_invert_toggle`
- `_on_show_new`
- `_on_show_old`
- `_show_image`
- `_on_toggle_histogram`
- `_on_stretch_changed`
- `_on_mouse_moved`
- `_on_zoom_changed`
- `set_image_data`

### 完成标准

- `MainWindow` 不再直接实现显示状态机
- `BlinkService` 仍为唯一闪烁业务状态源

### 建议运行

- `pytest scann_v2/tests/test_blink_service.py`
- `pytest scann_v2/tests/test_blink_speed_slider.py`
- `pytest scann_v2/tests/test_histogram_panel.py`
- `pytest scann_v2/tests/test_coordinate_label.py`
- `pytest scann_v2/tests/test_image_session_controller.py`

---

## 提交 8：引入 DetectionController 骨架

### 当前状态

- 已完成

### 目标

先把检测相关菜单和候选体操作的入口收束到一个控制器里。

### 修改内容

- 新增：
  - `src/scann/gui/controllers/detection_controller.py`
  - 可选 `tests/test_detection_controller.py`
- 修改：
  - `src/scann/gui/main_window.py`

### 迁移方式

- 槽函数先委托 controller
- 具体检测实现仍暂时留在旧代码或旧 service

### 本提交不做的事

- 不拆 `detection_service.py` 内部结构

### 完成标准

- 批量对齐、批量处理、批量检测、候选体标记入口集中到 controller

### 建议运行

- `pytest scann_v2/tests/test_main_window_features.py -k "detect or candidate or batch"`
- `pytest scann_v2/tests/test_detection_controller.py`

---

## 提交 9：拆分 DetectionPipeline 辅助模块

### 当前状态

- 已完成

### 目标

在不改变对外接口的前提下，拆掉 `detection_service.py` 里最重的私有实现细节。

### 修改内容

- 新增：
  - `src/scann/services/detection_pipeline.py`
  - `src/scann/services/detection_patch_extractor.py`
  - `src/scann/services/detection_window_scanner.py`
  - `src/scann/services/detection_postprocess.py`
  - `src/scann/services/detection_image_adapter.py`
- 修改：
  - `src/scann/services/detection_service.py`
  - 或者将其保留为兼容导出层
- 新增测试：
  - `tests/test_detection_postprocess.py`
  - `tests/test_detection_window_scanner.py`

### 推荐做法

- 先复制私有函数到新文件
- 再让旧 `DetectionPipeline` 调新 helper
- 最后再决定是否重命名旧文件

### 完成标准

- `DetectionPipeline.process_pair()` 对外签名不变
- 滑窗、NMS、patch 逻辑可独立单测

### 建议运行

- `pytest scann_v2/tests/test_detection_pipeline_regression.py`
- `pytest scann_v2/tests/test_detection_postprocess.py`
- `pytest scann_v2/tests/test_detection_window_scanner.py`
- `pytest scann_v2/tests/test_nms.py`

### 风险提示

- 不要在这一提交顺手改算法阈值或行为

---

## 提交 10：迁移检测工作流到 DetectionController

### 当前状态

- 已完成

### 目标

把主窗口里的检测用例真正迁移到 controller，主窗口只保留事件绑定。

### 修改内容

- 修改：
  - `src/scann/gui/controllers/detection_controller.py`
  - `src/scann/gui/main_window.py`
  - 可能修改 `src/scann/gui/presenters/candidate_presenter.py`

### 迁移方法

- `_on_batch_align`
- `_on_batch_process`
- `_run_batch_process`
- `_build_detection_params`
- `_on_batch_detect`
- `_on_mark_real`
- `_on_mark_bogus`
- `_on_next_candidate`
- `_on_candidate_selected`
- `_on_candidate_double_clicked`
- `_focus_candidate`

### 完成标准

- 主窗口不再直接构建检测参数和管线对象
- 候选体选择、打标、跳转仍然一致

### 建议运行

- `pytest scann_v2/tests/test_detection_controller.py`
- `pytest scann_v2/tests/test_detection_pipeline.py`
- `pytest scann_v2/tests/test_candidate_detector.py`
- `pytest scann_v2/tests/test_suspect_table.py`

---

## 提交 11：拆分 QueryService 为 utils 与 clients

### 当前状态

- 已完成

### 目标

把通用计算与外部数据源适配从 `QueryService` 中分离出来。

### 修改内容

- 新增：
  - `src/scann/services/query_utils.py`
  - `src/scann/services/query_models.py`
  - `src/scann/services/query_clients/__init__.py`
  - `src/scann/services/query_clients/vsx_client.py`
  - `src/scann/services/query_clients/mpc_client.py`
  - `src/scann/services/query_clients/simbad_client.py`
  - `src/scann/services/query_clients/tns_client.py`
  - `src/scann/services/query_clients/satellite_client.py`
- 修改：
  - `src/scann/services/query_service.py`
- 新增测试：
  - `tests/test_query_utils.py`
  - `tests/test_query_clients.py`

### 完成标准

- `QueryService` 主要负责聚合接口
- 坐标转换和距离计算独立可测
- 各查询源可单独 mock 与测试

### 建议运行

- `pytest scann_v2/tests/test_query_service_regression.py`
- `pytest scann_v2/tests/test_query_utils.py`
- `pytest scann_v2/tests/test_query_clients.py`
- `pytest scann_v2/tests/test_query_service.py`

### 验证记录

- 2026年3月10日：`pytest scann_v2/tests/test_query_service_regression.py scann_v2/tests/test_query_utils.py scann_v2/tests/test_query_clients.py scann_v2/tests/test_query_service.py scann_v2/tests/test_query_service_apis.py` 通过（29 passed, 2 skipped）

---

## 提交 12：引入 QueryController 并迁移查询入口

### 当前状态

- 已完成

### 目标

让主窗口不再直接处理查询路由、上下文菜单查询和 MPC 报告构建。

### 修改内容

- 新增：
  - `src/scann/gui/controllers/query_controller.py`
  - 可选 `src/scann/services/report_service.py`
- 修改：
  - `src/scann/gui/main_window.py`
- 新增测试：
  - `tests/test_query_controller.py`

### 迁移方法

- `_on_image_clicked`
- `_on_image_right_click`
- `_do_query`
- `_on_menu_query`
- `_on_mpc_report`
- `_on_context_mpc_report`
- `_on_context_add_candidate`
- `_on_copy_wcs_coordinates`

### 完成标准

- 右键查询、菜单查询、复制坐标、报告生成入口都通过 controller/service 完成

### 建议运行

- `pytest scann_v2/tests/test_main_window_query_flow.py`
- `pytest scann_v2/tests/test_query_controller.py`
- `pytest scann_v2/tests/test_mpc_report_dialog.py`
- `pytest scann_v2/tests/test_observation_report.py`

---

## 提交 13：引入 ModelService、ModelController、TrainingController

### 当前状态

- 已完成

### 目标

收紧 AI 模型生命周期和训练流程边界。

### 修改内容

- 新增：
  - `src/scann/services/model_service.py`
  - `src/scann/gui/controllers/model_controller.py`
  - `src/scann/gui/controllers/training_controller.py`
- 修改：
  - `src/scann/gui/main_window.py`
- 新增测试：
  - `tests/test_model_service.py`
  - 可选 `tests/test_training_controller.py`

### 迁移方法

- `_on_open_training`
- `_on_training_started`
- `_on_training_progress`
- `_on_training_finished`
- `_on_training_error`
- `_on_training_stopped`
- `_on_load_model`
- `_on_model_info`

### 完成标准

- 主窗口不再直接维护 `_inference_engine` 生命周期
- 训练对话框和训练 worker 回调经由 controller 统一协调

### 建议运行

- `pytest scann_v2/tests/test_model_service.py`
- `pytest scann_v2/tests/test_model.py`
- `pytest scann_v2/tests/test_inference_full_image.py`
- `pytest scann_v2/tests/test_trainer.py`
- `pytest scann_v2/tests/test_main_window_features.py -k "model or training"`

### 风险提示

- 训练流程通常涉及线程和信号，尽量不要与其他大改同提交

---

## 提交 14：引入 ConfigService、PreferencesController 并清理主窗口

### 当前状态

- 已完成

### 目标

完成关闭恢复、配置保存、设置入口迁移，并把主窗口收缩为组装器。

### 修改内容

- 新增：
  - `src/scann/services/config_service.py`
  - `src/scann/gui/controllers/preferences_controller.py`
  - 可选 `src/scann/data/config_repository.py`
  - 可选 `src/scann/gui/composition/main_window_builder.py`
- 修改：
  - `src/scann/gui/main_window.py`
  - `src/scann/core/config.py` 或保留兼容层
- 新增测试：
  - `tests/test_preferences_controller.py`

### 迁移方法

- `_on_open_preferences`
- `_on_select_mpcorb_file`
- `_save_runtime_state`
- `_restore_ui_state`
- `closeEvent`
- `resizeEvent`

### 额外清理

- 删除主窗口内不再使用的私有方法
- 删除不再需要的直接 import
- 更新架构文档与开发说明

### 完成标准

- `main_window.py` 主要只剩：初始化、组件持有、controller 装配、少量生命周期 glue code
- 配置读取保存与 UI 恢复行为一致

### 建议运行

- `pytest scann_v2/tests/test_preferences_controller.py`
- `pytest scann_v2/tests/test_config.py`
- `pytest scann_v2/tests/test_settings_dialog.py`
- `pytest scann_v2/tests/test_main_window.py`

---

## 4. 每次提交后的统一检查

每一步提交后，除了本项最小测试集，还建议做以下检查：

1. `main_window.py` 的 import 是否减少或至少没有增加
2. 是否新增了明显的循环依赖
3. 是否把 Qt 控件操作错误地下沉到了 service 层
4. 是否出现多个模块同时维护同一份状态
5. 是否保留了必要兼容层，避免一次迁移过猛

---

## 5. 不建议的做法

以下做法会显著提高重构失败概率：

1. 在拆结构的同时顺手改算法阈值、UI 文案、数据模型字段
2. 先把 `main_window.py` 大片删除，再慢慢补 controller
3. 一个提交里同时拆 Detection、Query、Training 三条链路
4. 没有新增测试就把核心工作流移出主窗口
5. service 层直接引用 Qt widget 或 dialog

---

## 6. 建议的里程碑检查点

### 里程碑 A

完成提交 1 到提交 3 后，检查：

- 展示逻辑已开始离开主窗口
- 测试保护网可用

### 里程碑 B

完成提交 4 到提交 7 后，检查：

- 配对和图像会话两条主链已脱离主窗口实现细节
- 主窗口文件长度应开始显著下降

### 里程碑 C

完成提交 8 到提交 12 后，检查：

- 检测和查询两条业务主链已通过 controller/service 工作
- UI 不再直接跨层操作大部分领域逻辑

### 里程碑 D

完成提交 13 到提交 14 后，检查：

- 模型、训练、配置都已完成边界收束
- `main_window.py` 已接近目标形态

---

## 7. 推荐起手顺序

如果现在就要开始动代码，建议严格从以下 3 个提交开始，不要跳：

1. 提交 1：建立回归保护网
2. 提交 2：引入 Presenter 骨架
3. 提交 3：主窗口接入 Presenter

原因：

- 这三步风险最低
- 最容易验证
- 能最快为后续 controller 化提供缓冲层
