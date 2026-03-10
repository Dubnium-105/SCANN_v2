# SCANN v2 重构拆分方案

> 最后更新：2026年3月10日

> 当前执行进度：已完成提交 14（引入 ConfigService、PreferencesController 并清理主窗口）

本文档给出 SCANN v2 当前代码结构的完整重构拆分方案，目标是降低 GUI 热点类的复杂度，恢复 Core → Service → GUI 的单向依赖，降低跨层直接耦合，同时保证功能连续可用、测试可逐步迁移。

---

## 1. 背景与问题定义

当前 v2 的包级结构总体清晰，但存在以下明显热点：

- `src/scann/gui/main_window.py` 体量过大，已同时承担 UI 组装、文件流程、检测流程、模型管理、训练集成、查询集成、配置保存、运行态恢复等职责
- GUI 层直接依赖 `core`、`services`、`ai`、`data` 多层，削弱了 Service 层作为应用编排层的价值
- `services/detection_service.py` 同时承担流程编排和算法实现细节，职责偏重
- `services/query_service.py` 将外部源接入、角距离计算、坐标解析、聚合查询放在一个文件，后续扩展成本高
- `core/config.py` 同时承担配置模型和文件持久化，语义上偏向基础设施层

这些问题尚未导致架构失控，但已经会直接影响以下方面：

- 新功能接入成本上升
- 回归风险集中在主窗口和几个大服务文件
- 单元测试难度增加，更多测试被迫走 UI 集成路径
- 未来引入后台任务、异步查询、数据库持久化时边界会继续恶化

---

## 2. 重构目标

本次重构的目标不是“重写”，而是在保持现有功能和测试资产的前提下完成结构收敛。

### 2.1 结构目标

- 恢复分层约束：GUI 主要依赖 Application/Service，不直接跨层调用大量 Core/AI/Data 细节
- 将 `MainWindow` 降为“组装器 + 事件入口”，不再承担完整业务流程
- 将复杂用例抽为独立控制器或用例服务，按场景拆分
- 将算法细节与流程编排分离
- 为后续引入线程、后台任务、数据库、缓存提供稳定边界

### 2.2 可量化目标

- `main_window.py` 从约 2000 行缩减到 500 到 800 行以内
- GUI 对 `core` 的直接 import 数量减少 60% 以上
- 检测、查询、模型管理、配对加载等用例具备独立测试入口
- 新增文件后每个文件尽量控制在 300 到 500 行以内，超过时必须有明确的子领域理由

### 2.3 非目标

- 不在本次重构中重写检测算法本身
- 不在本次重构中更换 PyQt 框架或引入 QML
- 不在本次重构中大规模改动数据模型字段
- 不在本次重构中修改用户可见工作流和菜单结构

---

## 3. 重构原则

### 3.1 先拆边界，再拆实现

优先把跨层调用改成窄接口，再处理大函数内部细节。先建立稳定边界，后续局部优化才有意义。

### 3.2 兼容优先

每一阶段都允许旧入口保留一段时间，先在 `MainWindow` 中委托新对象，待测试稳定后再删除旧代码。

### 3.3 逐步迁移

每次只移动一个垂直场景，确保可以独立提交、独立回归。

### 3.4 测试跟随重构

每拆一个模块，就同步补其独立测试，不把回归压力全部压到 `test_main_window*.py`。

---

## 4. 目标架构

建议将现有结构演进为下述形态：

```text
src/scann/
  app.py
  logger_config.py
  core/
    ...
  ai/
    ...
  data/
    ...
  services/
    blink_service.py
    detection_pipeline.py
    query_service.py
    exclusion_service.py
    model_service.py
    pair_service.py
    report_service.py
    config_service.py
    __init__.py
  gui/
    main_window.py
    composition/
      main_window_builder.py
      action_factory.py
    controllers/
      image_session_controller.py
      pair_controller.py
      detection_controller.py
      query_controller.py
      model_controller.py
      training_controller.py
      preferences_controller.py
    presenters/
      candidate_presenter.py
      status_presenter.py
      image_presenter.py
    dialogs/
      ...
    widgets/
      ...
```

其中：

- `services/` 负责应用级用例编排，不处理 Qt 控件细节
- `gui/controllers/` 负责把 Qt 事件转换成对 service 的调用，再把结果回写到 view
- `gui/presenters/` 负责把领域对象映射为表格、状态栏、viewer 所需显示格式
- `gui/composition/` 负责动作、菜单、信号绑定等装配逻辑
- `main_window.py` 只保留窗口骨架、组件持有和 controller 初始化

---

## 5. 文件级拆分方案

## 5.1 主窗口拆分

### 当前问题

`src/scann/gui/main_window.py` 当前包含以下混合职责：

- 应用启动后的配置加载与恢复
- 菜单、中央布局、状态栏、快捷键装配
- 闪烁、反色、拉伸等图像显示状态管理
- 文件夹扫描、配对列表维护、配对切换
- 批量对齐、批量处理、批量检测
- 查询路由、MPC 报告生成
- 模型加载、训练流程接入
- 配置保存、最近目录管理、关闭事件恢复

### 目标拆分

建议从 `main_window.py` 抽出以下文件：

#### 1. `src/scann/gui/composition/main_window_builder.py`

职责：

- 创建菜单栏、中央区域、状态栏
- 初始化 widgets 和 dialogs 的基础实例
- 返回结构化引用对象，例如 `MainWindowUiParts`

从主窗口迁移的方法：

- `_init_menu_bar`
- `_init_central_ui`
- `_init_status_bar`
- `_init_histogram_dock`
- `_init_shortcuts`

#### 2. `src/scann/gui/controllers/image_session_controller.py`

职责：

- 管理当前显示图像、新图旧图切换、闪烁、反色、拉伸、viewer 刷新
- 维护与 `BlinkService` 的协作
- 更新浮层标签、缩放状态、坐标状态等视图状态

从主窗口迁移的方法：

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

#### 3. `src/scann/gui/controllers/pair_controller.py`

职责：

- 管理文件夹打开、最近目录、配对扫描、配对切换、缓存加载
- 协调 `PairService` 和 viewer/session

从主窗口迁移的方法：

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

#### 4. `src/scann/gui/controllers/detection_controller.py`

职责：

- 构建检测参数
- 发起批量对齐、批量处理、批量检测
- 协调 `DetectionPipeline`、`PairService`、`CandidatePresenter`

从主窗口迁移的方法：

- `_on_batch_align`
- `_on_batch_process`
- `_run_batch_process`
- `_build_detection_params`
- `_on_batch_detect`
- `set_candidates`
- `_update_markers`
- `_on_candidate_selected`
- `_on_candidate_double_clicked`
- `_focus_candidate`
- `_on_mark_real`
- `_on_mark_bogus`
- `_on_next_candidate`

#### 5. `src/scann/gui/controllers/query_controller.py`

职责：

- 右键查询路由
- 菜单查询路由
- 报告生成与弹窗
- 坐标复制与上下文动作

从主窗口迁移的方法：

- `_on_image_clicked`
- `_on_image_right_click`
- `_do_query`
- `_on_menu_query`
- `_on_mpc_report`
- `_on_context_mpc_report`
- `_on_context_add_candidate`
- `_on_copy_wcs_coordinates`

#### 6. `src/scann/gui/controllers/model_controller.py`

职责：

- 模型加载、模型信息展示
- 推理引擎生命周期管理
- 为检测控制器提供模型状态

从主窗口迁移的方法：

- `_on_load_model`
- `_on_model_info`

#### 7. `src/scann/gui/controllers/training_controller.py`

职责：

- 打开训练对话框
- 启动训练 worker
- 监听训练进度、完成、异常、停止

从主窗口迁移的方法：

- `_on_open_training`
- `_on_training_started`
- `_on_training_progress`
- `_on_training_finished`
- `_on_training_error`
- `_on_training_stopped`

#### 8. `src/scann/gui/controllers/preferences_controller.py`

职责：

- 设置对话框打开、配置保存、MPCORB 路径更新、关闭前状态保存

从主窗口迁移的方法：

- `_on_open_preferences`
- `_on_select_mpcorb_file`
- `_save_runtime_state`
- `_restore_ui_state`
- `closeEvent`
- `resizeEvent`

#### 9. `src/scann/gui/presenters/candidate_presenter.py`

职责：

- 将 `Candidate` 列表映射为表格行、viewer 标记、状态栏文本
- 统一真实/伪目标标记后的视图刷新

#### 10. `src/scann/gui/presenters/status_presenter.py`

职责：

- 状态栏消息输出
- 日志级别与用户消息统一包装

### 主窗口保留内容

`main_window.py` 最终只保留：

- `MainWindow` 类定义
- `__init__`
- 基础 widget 成员持有
- controller 初始化与连接
- 少量必须依赖窗口本体的 Qt 生命周期代码

---

## 5.2 服务层拆分

## 5.2.1 DetectionPipeline 拆分

### 当前问题

`src/scann/services/detection_service.py` 同时包含：

- 流程编排
- 图像预处理
- 滑窗生成
- patch 提取
- AI 评分
- NMS
- v1/v2 模型兼容适配

### 目标文件

#### 1. `src/scann/services/detection_pipeline.py`

保留 `DetectionPipeline`，仅负责：

- 执行阶段顺序
- 调用协作者
- 汇总结果与错误

#### 2. `src/scann/services/detection_patch_extractor.py`

职责：

- `_extract_patch`
- `_prepare_triplet_patch`
- 图像 patch 裁剪和边界填充

#### 3. `src/scann/services/detection_window_scanner.py`

职责：

- `_sliding_window_detect`
- 滑窗采样规则
- 有效区域筛选

#### 4. `src/scann/services/detection_postprocess.py`

职责：

- `_nms_candidates`
- AI 分数过滤
- 排序规则

#### 5. `src/scann/services/detection_image_adapter.py`

职责：

- `_robust_to_uint8`
- v1 模型输入转换
- 归一化与输入形态适配

### 迁移原则

- 第一阶段先原样移动私有函数，不改逻辑
- 第二阶段再将协作者通过构造函数注入到 `DetectionPipeline`
- 第三阶段补充针对滑窗、patch、NMS 的纯单元测试

## 5.2.2 QueryService 拆分

### 当前问题

`src/scann/services/query_service.py` 将领域算法和外部 API 适配混在一起。

### 目标文件

#### 1. `src/scann/services/query_service.py`

只保留聚合接口：

- `query_vsx`
- `query_mpc`
- `query_simbad`
- `query_tns`
- `check_satellite`

实际实现转为委托子客户端。

#### 2. `src/scann/services/query_utils.py`

职责：

- HMS/DMS 解析
- 角距离计算

#### 3. `src/scann/services/query_clients/vsx_client.py`

#### 4. `src/scann/services/query_clients/mpc_client.py`

#### 5. `src/scann/services/query_clients/simbad_client.py`

#### 6. `src/scann/services/query_clients/tns_client.py`

#### 7. `src/scann/services/query_clients/satellite_client.py`

这样做的收益：

- 每个外部源可单独测试与限流
- API 变更时只影响对应 client
- 主服务只负责聚合和统一 `QueryResult`

## 5.2.3 新增应用服务

建议新增以下服务，承接 GUI 直接调用的跨层逻辑：

#### 1. `src/scann/services/pair_service.py`

职责：

- 调用 `scan_fits_folder`、`match_new_old_pairs`
- 读取配对图像
- 处理对齐产物路径与裁剪边界

#### 2. `src/scann/services/model_service.py`

职责：

- 封装 `InferenceEngine` 的创建、替换、关闭、信息读取
- 统一暴露 `is_ready`、`threshold`、`model_info`

#### 3. `src/scann/services/report_service.py`

职责：

- 从 `Candidate` 和 `FitsHeader` 生成 `Observation`
- 委托 `generate_mpc_report`

#### 4. `src/scann/services/config_service.py`

职责：

- 提供配置加载、保存、路径验证的应用接口
- 为后续将 `config.py` 持久化逻辑迁出 `core` 做过渡层

---

## 5.3 Core 与 Data 的边界收敛

## 5.3.1 config 拆分

当前 `src/scann/core/config.py` 包含两类职责：

- 配置模型使用与数据映射
- JSON 文件持久化和路径清洗

建议演进为：

- `src/scann/core/models.py` 继续持有 `AppConfig` 等数据模型
- `src/scann/services/config_service.py` 或 `src/scann/data/config_repository.py` 负责文件加载保存

推荐目标：

```text
src/scann/data/config_repository.py
src/scann/services/config_service.py
```

其中：

- `config_repository.py` 负责 JSON IO
- `config_service.py` 负责默认值、路径清洗、向 GUI 提供友好接口

## 5.3.2 ExclusionService 与 Astrometry 复用

当前 `ExclusionService` 自己实现 `_pixel_to_sky` 和 `_calculate_angular_distance`，建议逐步替换为：

- 优先复用 `core/astrometry.py` 中已有能力
- 将角距离计算统一收敛为一个共享工具函数

这样可以减少坐标逻辑在多个文件中漂移。

---

## 6. 分阶段实施计划

建议按以下 7 个阶段推进，每个阶段都应保证测试可通过并可独立回滚。

## 阶段 0：建立保护网

目标：在开始拆分前建立回归基线。

工作项：

- 盘点主窗口现有测试覆盖点
- 为高风险交互补充快照式或行为式测试
- 为 `DetectionPipeline` 和 `QueryService` 建立最小回归测试集
- 固化 import 方向检查规则，至少人工约束 GUI 不新增直接依赖

建议新增测试：

- `tests/test_main_window_pair_flow.py`
- `tests/test_main_window_query_flow.py`
- `tests/test_detection_pipeline_regression.py`
- `tests/test_query_service_regression.py`

退出条件：

- 主流程测试可稳定复现当前行为

## 阶段 1：抽出 Presenter 和 Message 层

目标：先削弱 `MainWindow` 中最容易抽离的展示职责。

工作项：

- 新增 `candidate_presenter.py`
- 新增 `status_presenter.py`
- 将 `_show_message`、表格刷新、marker 刷新逻辑迁出主窗口

收益：

- 低风险
- 对 UI 行为影响最小
- 为后续 controller 化打基础

## 阶段 2：抽出 PairController 和 ImageSessionController

目标：先拆最重的交互路径，即图像会话和配对加载。

工作项：

- 引入 `pair_service.py`
- 引入 `pair_controller.py`
- 引入 `image_session_controller.py`
- 主窗口改为委托 controller

退出条件：

- 打开文件夹、切换配对、闪烁、拉伸、查看器刷新行为不变

## 阶段 3：抽出 DetectionController 与 Detection 子模块

目标：把检测工作流从主窗口中完全隔离。

工作项：

- 新增 `detection_controller.py`
- 拆分 `detection_service.py` 为 pipeline + helpers
- 将批量对齐、批量处理、批量检测迁到控制器和服务层

退出条件：

- 检测结果数量、排序、阈值行为与现状一致

## 阶段 4：抽出 QueryController 与 Query Clients

目标：降低查询相关的外部依赖复杂度。

工作项：

- 新增 `query_controller.py`
- 新增 `query_utils.py`
- 新增 `query_clients/`
- 主窗口不再直接感知各查询源的实现细节

退出条件：

- 右键查询、菜单查询、结果弹窗行为一致

## 阶段 5：抽出 ModelController 和 TrainingController

目标：将 AI 生命周期与窗口事件解耦。

工作项：

- 新增 `model_service.py`
- 新增 `model_controller.py`
- 新增 `training_controller.py`
- 主窗口只保留菜单动作触发

退出条件：

- 模型加载、模型信息、训练流程、训练回调行为一致

## 阶段 6：配置与关闭流程归位

目标：把关闭恢复、配置持久化、MPCORB 配置等边界收紧。

工作项：

- 新增 `config_service.py`
- 可选新增 `data/config_repository.py`
- 新增 `preferences_controller.py`
- 主窗口只保留生命周期回调入口

退出条件：

- 配置文件读写与 UI 恢复行为一致

## 阶段 7：清理兼容层和文档

目标：删除过渡代码并更新架构文档。

工作项：

- 删除主窗口中残留的旧委托代码
- 更新 `docs/architecture.md`
- 更新测试说明和模块说明
- 如有必要，补充开发者约束文档

---

## 7. 推荐实施顺序与提交粒度

建议按“小步提交”推进，每次提交只做一类结构变化。

对应的逐提交执行版本见 [refactor_commit_checklist.md](refactor_commit_checklist.md)。

推荐提交序列：

1. 新增 presenter，不迁移业务逻辑
2. 主窗口接入 presenter
3. 新增 pair_service 和 pair_controller
4. 迁移配对与加载逻辑
5. 新增 image_session_controller
6. 迁移闪烁、反色、拉伸、显示逻辑
7. 新增 detection_controller
8. 拆分 detection_service 子模块
9. 新增 query_controller 和 query clients
10. 新增 model_service、model_controller、training_controller
11. 新增 config_service、preferences_controller
12. 清理兼容代码与更新文档

每一步都应满足：

- 测试通过
- 对外行为不变
- 主窗口 import 数量下降或至少不再上升

---

## 8. 目录与类设计建议

以下给出建议的核心类接口，用于统一拆分方向。

## 8.1 MainWindow 与 Controller 的关系

```python
class MainWindow(QMainWindow):
    def __init__(self):
        self.ui = build_main_window(self)
        self.services = self._create_services()
        self.controllers = self._create_controllers()
        self._connect_actions()
```

说明：

- `MainWindow` 不直接实现大段业务流程
- `services` 负责跨层协作
- `controllers` 负责把菜单、按钮、快捷键事件转为用例调用

## 8.2 PairController

```python
class PairController:
    def open_new_folder(self) -> None: ...
    def open_old_folder(self) -> None: ...
    def select_pair(self, index: int) -> None: ...
    def open_recent_folder(self, folder: str) -> None: ...
```

## 8.3 DetectionController

```python
class DetectionController:
    def batch_align(self) -> None: ...
    def batch_process(self, params: dict) -> None: ...
    def batch_detect(self) -> None: ...
    def mark_real(self) -> None: ...
    def mark_bogus(self) -> None: ...
    def next_candidate(self) -> None: ...
```

## 8.4 ModelService

```python
class ModelService:
    def load_model(self, model_path: str, config) -> ModelInfo: ...
    def unload_model(self) -> None: ...
    def get_engine(self) -> InferenceEngine | None: ...
    def get_model_info(self) -> ModelInfo | None: ...
```

重点：GUI 不再直接维护 `_inference_engine` 的生命周期细节。

---

## 9. 测试重构方案

重构成功与否，很大程度取决于测试是否随边界迁移。

### 9.1 测试目标

- 从“主窗口大集成测试”逐步演进到“controller/service 单测 + 少量 UI 集成测”
- 让配对、检测、查询、模型管理分别有独立测试入口

### 9.2 建议新增测试文件

- `tests/test_pair_service.py`
- `tests/test_pair_controller.py`
- `tests/test_image_session_controller.py`
- `tests/test_detection_controller.py`
- `tests/test_detection_postprocess.py`
- `tests/test_detection_window_scanner.py`
- `tests/test_model_service.py`
- `tests/test_query_utils.py`
- `tests/test_query_clients.py`
- `tests/test_preferences_controller.py`

### 9.3 现有测试迁移原则

- `test_main_window.py` 保留窗口级冒烟测试
- `test_main_window_features.py` 中与配对、检测、查询、模型加载相关的用例逐步迁往对应 controller 测试
- 原先必须 mock 大量 Qt 对象的测试，优先在 service 层重建

---

## 10. 风险与控制措施

## 风险 1：信号连接在迁移中断裂

控制措施：

- 为菜单动作、按钮点击、viewer 信号建立显式连接测试
- 新 controller 接入后保留旧入口一版，逐项切换

## 风险 2：状态分散后出现多处真相源

控制措施：

- 明确状态归属
- 图像显示状态归 `ImageSessionController`
- 配对状态归 `PairController`
- 模型状态归 `ModelService`
- 配置状态归 `ConfigService`

## 风险 3：重构过程中 import 循环增加

控制措施：

- controller 只依赖 service、presenter、window 接口
- presenter 不依赖 controller
- service 不依赖 Qt

## 风险 4：行为回归隐藏在 UI 层

控制措施：

- 每阶段保留回归测试
- 使用主流程人工 checklist 做冒烟验证

---

## 11. 人工回归检查清单

每个阶段完成后至少手动验证以下流程：

1. 打开新旧图文件夹并正确配对
2. 切换配对后图像、坐标、候选表同步刷新
3. 闪烁、反色、拉伸、缩放行为正常
4. 批量对齐与批量处理结果可落盘
5. 模型加载成功后可执行批量检测
6. 候选体可标记真/假并正确高亮
7. 右键查询和菜单查询均能弹出结果
8. MPC 报告可生成
9. 设置保存后重新打开程序能恢复关键状态

---

## 12. 第一批实际落地建议

如果只做第一轮重构，建议不要同时动所有模块，而是先完成以下最有性价比的 4 步：

1. 从 `main_window.py` 抽出 `status_presenter.py` 和 `candidate_presenter.py`
2. 新增 `pair_service.py` 和 `pair_controller.py`，迁移文件夹打开与配对加载
3. 新增 `image_session_controller.py`，迁移闪烁、反色、拉伸和图像显示逻辑
4. 将 `DetectionPipeline` 内的滑窗、patch、NMS 拆到 helper 文件

原因：

- 这四步对耦合降低最明显
- 对用户行为改动最小
- 能最快把主窗口从“全能类”降为“组装类”

---

## 13. 完成标准

当满足以下条件时，可以认为本轮架构重构完成：

- `MainWindow` 不再直接调用多数 `core`、`ai`、`data` 模块细节
- 主窗口代码主要由 UI 组装和 controller 初始化构成
- 检测、查询、模型管理、配置管理均具备独立 service/controller 边界
- 重构后主流程测试与人工回归通过
- `docs/architecture.md` 已更新为新的实际架构，而不是旧目标架构

---

## 14. 附录：建议优先创建的文件

建议第一轮先创建以下文件骨架：

- `src/scann/gui/composition/main_window_builder.py`
- `src/scann/gui/controllers/image_session_controller.py`
- `src/scann/gui/controllers/pair_controller.py`
- `src/scann/gui/controllers/detection_controller.py`
- `src/scann/gui/presenters/status_presenter.py`
- `src/scann/gui/presenters/candidate_presenter.py`
- `src/scann/services/pair_service.py`
- `src/scann/services/model_service.py`
- `src/scann/services/detection_pipeline.py`
- `src/scann/services/detection_postprocess.py`
- `src/scann/services/detection_patch_extractor.py`
- `src/scann/services/detection_window_scanner.py`

这组文件足以覆盖当前最主要的耦合热点。