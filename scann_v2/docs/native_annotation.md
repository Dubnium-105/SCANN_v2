# 原生 FITS 标注平台

本文档描述当前仓库中的原生 FITS 标注平台实现，而不是早期设计草案。

更详细的数据集预处理和数据库设计请看：

- `dataset_pipeline.md`

## 1. 组成

平台由两部分组成：

- 后端：`src/scann/native_annotation/`
- 前端：`frontend/`

后端使用 FastAPI，前端使用 Vue 3 + Vite。

## 2. 后端职责

后端入口：

- `scann.native_annotation.app:create_app`
- `scann.native_annotation.app:app`

主要服务包括：

- `auth_service.py`
  - 注册、登录、JWT 校验
- `dataset_service.py`
  - 列出可标注任务
- `task_lock_service.py`
  - 任务领取、心跳续租、释放
- `fits_engine.py`
  - 返回 FITS 原文件或渲染后的 PNG
- `annotation_service.py`
  - 标注保存、历史查询、修订详情、回滚
- `routes.py`
  - 对外 API 路由

## 3. 当前 API 范围

当前已接入的主要接口包括：

- `/api/health`
- `/api/login`
- `/api/register`
- `/api/tasks`
- `/api/tasks/next`
- `/api/tasks/{task_id}/heartbeat`
- `/api/tasks/{task_id}/release`
- `/api/render/{file_path}`
- `/api/fits/{file_path}`
- `/api/annotations/{task_id}`
- `/api/annotations/{task_id}/history`
- `/api/annotations/{task_id}/history/{revision_id}`
- `/api/annotations/{task_id}/rollback/{revision_id}`
- `/api/dataset/preprocess`

## 4. 数据集与预处理

平台运行时依赖一个数据集根目录。代码默认从环境变量 `SCANN_NATIVE_DATASET_ROOT` 读取该路径。

当前推荐的输入目录是：

- `dataset_raw/new/`
- `dataset_raw/old/`
- `dataset_raw/new_marked/`

预处理后的输出目录是：

- `new/`
- `old/`
- `new_marked/`

预处理入口在 `scann.services.dataset_preprocess_service.DatasetPreprocessService`，负责：

- 扫描 `dataset_raw/*` 原始文件
- 根据 `field_key` 和 `capture_key` 生成任务
- 复用单旧图到多个任务，而不是复制旧图文件
- 生成对齐裁剪产物
- 将产物路径和任务状态写入数据库

如果用户仍把文件直接放进 `new/old/new_marked`，系统会做兼容迁移，但这不是主流程。

## 5. 数据库存储

当前原生标注平台不再以 `scann_native.db + annotations.json` 作为主存储。

主存储现在是数据集级数据库：

- `scann_dataset.db`

数据库中统一保存：

- 原始资产登记
- 任务主表
- 任务产物路径
- 当前标注
- 修订历史
- 任务锁
- 本地查看、本地标注、在线标注状态

对应实现：

- `src/scann/core/dataset_storage.py`

其中：

- `task_lock_service.py` 使用数据库字段维护领取状态
- `annotation_service.py` 将在线标注和修订历史写入同一个数据集库
- `dataset_service.py` 从数据库产物列表生成任务会话

兼容文件：

- `preprocessed_tasks.json`
- `annotations.json`

它们现在只用于兼容和导出，不再是系统内部真相来源。

## 6. 在线标注链路

当前在线标注链路如下：

1. `/api/dataset/preprocess` 触发预处理
2. `DatasetPreprocessService` 扫描原始输入、生成任务与产物
3. `/api/tasks` 返回已准备好的任务列表
4. `/api/tasks/next` 通过数据库领取任务
5. 前端通过 `/api/render/*` 或 `/api/fits/*` 获取图像
6. `/api/annotations/{task_id}` 保存当前标注并追加 revision
7. `/api/annotations/{task_id}/history*` 查询历史与差异
8. `/api/annotations/{task_id}/rollback/*` 通过追加 rollback revision 实现回滚

## 7. 本地查看与桌面端复用

桌面端与在线标注现在共享同一套任务和路径事实来源：

- 桌面端浏览任务时，从数据库中的预处理结果读取对齐产物
- 桌面端标注通过 `FitsAnnotationStorage` 写当前标注
- 训练和在线标注也优先读取数据库导出的精确路径

这避免了不同入口各自扫目录、再按文件名猜任务的漂移问题。

## 8. 本地启动

### 后端

从仓库根目录启动：

```powershell
pip install -r docker\backend\requirements.txt
$env:PYTHONPATH = (Resolve-Path .\scann_v2\src)
$env:SCANN_NATIVE_DATASET_ROOT = (Resolve-Path .\dataset)
python -m uvicorn scann.native_annotation.app:app --reload
```

### 前端

```powershell
cd scann_v2\frontend
npm ci
npm run dev
```

## 9. 文档维护说明

如果原生标注平台的任务流、数据库结构或预处理约定发生变化，应优先更新：

- `dataset_pipeline.md`
- 本文档中的概览说明
