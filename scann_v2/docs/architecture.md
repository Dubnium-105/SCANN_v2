# SCANN v2 架构概览

本文档只描述当前仍在维护的结构，不再保留历史重构计划或阶段性草案。

与数据集预处理和数据库细节直接相关的说明请看：

- `dataset_pipeline.md`

## 1. 系统组成

SCANN v2 当前由三部分组成：

1. 桌面应用
   - 位于 `src/scann/`
   - 以 PyQt5 为界面层
   - 提供 FITS 浏览、闪烁比对、图像处理、候选体检测、人工标注、外部查询和报告相关能力
2. 原生 FITS 标注平台
   - 后端位于 `src/scann/native_annotation/`
   - 前端位于 `frontend/`
   - 提供任务分配、在线标注、修订历史、回滚和数据集预处理能力
3. 运行与发布层
   - 位于 `docker/`
   - 负责容器构建、Linux 部署和 GitHub Actions 流水线

## 2. 代码目录

```text
src/scann/
|-- ai/                 # 模型、推理和训练相关代码
|-- core/               # 底层能力：FITS IO、对齐、图像处理、统一数据存储
|-- data/               # 本地文件与数据访问辅助
|-- gui/                # PyQt5 桌面界面、控制器、组件和对话框
|-- native_annotation/  # FastAPI 标注后端
|-- services/           # 桌面和标注平台复用的编排服务
|-- app.py              # 桌面应用入口
`-- logger_config.py    # 日志配置
```

## 3. 模块边界

### `core/`

`core/` 放与界面无关、可测试的底层能力，例如：

- `fits_io.py`：FITS 读写
- `image_aligner.py`：新旧图像对齐
- `image_processor.py`：图像显示与基础处理
- `candidate_detector.py`：候选体检测逻辑
- `fits_annotation_backend.py` / `fits_annotation_storage.py`：桌面标注数据访问
- `dataset_storage.py`：统一数据集数据库封装

### `services/`

`services/` 负责把底层能力串成完整业务流，例如：

- `pair_service.py`：图像配对与路径解析
- `blink_service.py`：闪烁比对
- `detection_pipeline.py`：检测主流程
- `query_service.py`：外部服务查询
- `dataset_preprocess_service.py`：数据集扫描、任务规划、对齐裁剪、任务清单汇总

### `gui/`

`gui/` 只处理桌面应用的展示和交互：

- `controllers/`：配对、检测、训练、查询、标注等控制器
- `widgets/`：图像查看器、表格、工具栏等控件
- `dialogs/`：设置、训练、报告等对话框

### `native_annotation/`

该目录是独立的 FastAPI 服务，不依赖 PyQt。

核心职责：

- 身份认证
- 任务列表与任务锁
- FITS 原图与 PNG 渲染输出
- 标注保存、历史查询、回滚
- 触发数据集预处理

## 4. 关键业务流

### 4.1 桌面检测流

1. `data/file_manager.py` 与 `services/pair_service.py` 扫描并配对图像
2. `core/image_aligner.py` 以新图为参考对齐旧图
3. `services/detection_pipeline.py` 运行检测
4. `services/query_service.py` 等服务补充查询和筛除逻辑
5. `gui/` 层负责候选体展示、标记和导出

### 4.2 数据集预处理流

1. 用户将原始文件放入 `dataset_raw/new`、`dataset_raw/old`、`dataset_raw/new_marked`
2. `services/dataset_preprocess_service.py` 扫描原始文件并写入 `raw_assets`
3. 服务以 `new` 为驱动生成 `tasks`
4. 多个任务可以共享同一个 `old_asset_id`
5. 预处理生成对齐裁剪产物并写入 `task_artifacts`
6. 兼容清单 `preprocessed_tasks.json` 由数据库导出

### 4.3 在线标注流

1. `native_annotation/dataset_service.py` 从数据库返回任务列表
2. `native_annotation/task_lock_service.py` 通过数据库维护领取状态
3. `native_annotation/fits_engine.py` 提供图像内容
4. `native_annotation/annotation_service.py` 写当前标注和修订历史
5. `frontend/` 提供登录、标注画布、检查面板和历史交互

## 5. 数据层设计

当前与标注和任务相关的数据层已经统一到数据集数据库：

- `scann_dataset.db`

主要表包括：

- `raw_assets`
- `tasks`
- `task_artifacts`
- `task_annotation_boxes_current`
- `annotation_revisions`
- `annotation_revision_boxes`

这一层的意义是：

- 用数据库取代“重新扫目录 + 按文件名猜任务”
- 允许多个任务复用同一张旧图
- 让桌面端、在线端、训练端共享同一份任务与路径真相

兼容输出仍然存在：

- `preprocessed_tasks.json`
- `annotations.json`

但它们已经不是主存储。

## 6. 维护约定

- 长期有效的结构说明放在当前文档
- 一次性的实现计划、检查单、backlog 和阶段总结不再进入 `docs/`
- 若任务流或数据库设计变化，应优先同步 `dataset_pipeline.md`
