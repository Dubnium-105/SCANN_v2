# SCANN v2 架构概览

本文档只描述当前仍在维护的结构，不再保留历史重构计划或阶段性草稿。

## 1. 系统组成

SCANN v2 目前由三部分组成：

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
   - 负责容器镜像构建、Linux 部署和 GitHub Actions 流水线

## 2. 代码目录

```text
src/scann/
|-- ai/                 # 模型、推理和训练相关代码
|-- core/               # 纯业务能力：FITS IO、对齐、图像处理、MPC 相关逻辑等
|-- data/               # 本地文件和数据库访问
|-- gui/                # PyQt5 桌面界面、控制器、组件和对话框
|-- native_annotation/  # FastAPI 标注后端
|-- services/           # 桌面应用编排层与任务服务
|-- app.py              # 桌面应用入口
`-- logger_config.py    # 日志配置
```

## 3. 模块边界

### `core/`

`core/` 放与界面无关、可测试的底层能力，例如：

- `fits_io.py`：FITS 读写
- `image_aligner.py`：新旧图像对齐
- `image_processor.py`：显示或预处理相关图像操作
- `candidate_detector.py`：候选体检测基础逻辑
- `astrometry.py`、`mpcorb.py`、`observation_report.py`：天体位置与报告相关能力
- `fits_annotation_backend.py`、`fits_annotation_storage.py`：桌面标注数据结构与存储辅助

### `services/`

`services/` 负责把多个底层能力串成完整流程，例如：

- `pair_service.py`：新旧图像配对
- `blink_service.py`：闪烁比对
- `detection_pipeline.py`：检测主流程
- `detection_service.py`：桌面检测服务入口
- `exclusion_service.py`：已知天体排除
- `query_service.py`：外部天文服务查询
- `dataset_preprocess_service.py`：供原生标注平台复用的数据集标准化和预处理

### `gui/`

`gui/` 只处理桌面应用的展示与交互：

- `main_window.py` 与 `composition/`：主窗口和装配逻辑
- `controllers/`：配对、检测、训练、查询、标注、帮助等控制器
- `widgets/`：图像查看器、表格、标注组件、工具栏等
- `dialogs/`：设置、训练、MPC 报告、快捷键帮助等对话框
- `presenters/`：面向 UI 的状态整理

### `native_annotation/`

该目录是独立的 FastAPI 服务，不依赖 PyQt。

核心职责：

- 身份认证
- 任务列表与任务锁
- FITS 原图与渲染图提供
- 标注保存、历史查询、回滚
- 数据集预处理触发

## 4. 关键业务流

### 桌面检测流

1. `data/file_manager.py` 与 `services/pair_service.py` 扫描并配对新旧 FITS
2. `core/image_aligner.py` 以新图为参考对齐旧图
3. `services/detection_pipeline.py` 运行检测
4. 检测模式支持 `patch`、`full_image`、`hybrid`
5. `services/exclusion_service.py` 与 `services/query_service.py` 用于排除已知目标或补充查询信息
6. `gui/` 层负责候选体展示、标记和导出

### 原生标注流

1. `services/dataset_preprocess_service.py` 统一原始文件命名并生成对齐产物
2. `native_annotation/dataset_service.py` 生成任务列表
3. `native_annotation/task_lock_service.py` 控制任务占用和心跳续租
4. `native_annotation/fits_engine.py` 提供 FITS 二进制与 PNG 渲染
5. `native_annotation/annotation_service.py` 将标注状态写入 SQLite，并保留 revision 历史
6. `frontend/` 前端负责登录、标注画布、面板和历史交互

## 5. 数据与存储

当前仓库里有两套与标注相关的存储面：

- 桌面应用侧：`core/` 中的标注后端与存储辅助
- 原生标注平台侧：`native_annotation/annotation_service.py`

原生标注平台目前的持久化重点是：

- `scann_native.db`：SQLite 主存储
- `annotation_revisions/`：按任务保存的 revision 日志
- `annotations.json`：兼容和快照用途的数据文件

## 6. 维护约定

- 长期有效的结构说明放在当前文档
- 一次性的实施计划、检查单、backlog 和阶段总结不再进入 `docs/`
- 这类内容应优先进入 issue、PR 描述或外部项目管理工具
