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
- `annotation_sync_service.py`
  - 将当前标注与修订历史同步到 PostgreSQL，仅同步标注结构化数据，不同步 FITS 图像文件
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
- `/api/annotation-sync/status`
- `/api/annotation-sync/run`

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

## 8. PostgreSQL 标注备份与同步

平台支持把数据集的标注信息同步到指定 PostgreSQL 数据库，目标是远端备份与多环境共享标注结果。同步范围是：

- 新增的 revision 历史
- 新增 revision 的 bbox 明细
- 受新增 revision 影响的当前任务标注摘要
- 受新增 revision 影响的当前 bbox 标注
- 同步运行记录

不会同步 FITS 图像二进制，也不会同步 `dataset_raw/*` 原始文件路径。

同步默认借用标注平台的 revision 机制做增量备份。远端会在 `annotation_sync_state` 中记录当前数据集已经同步到的本地 `revision rowid`，后续定时或手动同步只上传这个游标之后新增的 revision。没有新增 revision 的定时同步不会反复写入全量标注数据。

### 8.1 环境变量

```powershell
$env:SCANN_ANNOTATION_SYNC_DATABASE_URL = "postgresql://user:password@host:5432/scann"
$env:SCANN_ANNOTATION_SYNC_DATASET_ID = "observatory-2026-04"
$env:SCANN_ANNOTATION_SYNC_SCHEMA = "scann_backup"
```

可选启用定时同步：

```powershell
$env:SCANN_ANNOTATION_SYNC_ENABLED = "true"
$env:SCANN_ANNOTATION_SYNC_INTERVAL_SECONDS = "300"
```

字段说明：

- `SCANN_ANNOTATION_SYNC_DATABASE_URL`：PostgreSQL DSN，也可用 `SCANN_ANNOTATION_SYNC_PG_DSN`
- `SCANN_ANNOTATION_SYNC_DATASET_ID`：远端数据集主键，生产部署建议显式设置
- `SCANN_ANNOTATION_SYNC_SCHEMA`：远端 schema，默认 `public`
- `SCANN_ANNOTATION_SYNC_ENABLED`：是否在后端启动时开启后台定时同步
- `SCANN_ANNOTATION_SYNC_INTERVAL_SECONDS`：定时同步间隔，必须大于 `0` 才会启动后台线程
- `SCANN_ANNOTATION_SYNC_CONNECT_TIMEOUT_SECONDS`：PG 连接超时，默认 `10`

### 8.2 Docker 部署说明

在常规 Docker 部署中：

- 主 `backend` 继续处理普通标注 API，不直接访问云 PostgreSQL
- 新增的 `sync-backend` 专门处理 `/api/annotation-sync/*`
- `sync-backend` 使用 `host` 网络访问云 PostgreSQL
- `frontend` 会把 `/api/annotation-sync/*` 代理到 `sync-backend` 的 Unix socket

这样可以在某些路由器或 OpenWrt 设备上，只让“同步到云 PG”这一条链路绕过 Docker bridge 出网限制，而不改变其余标注功能的部署方式。

首次成功同步前，远端 schema 中不会有 `annotation_sync_state` 等表；这些表会在第一次成功执行同步时自动创建。

如果远端 PostgreSQL 没有启用 SSL，请把 DSN 中的 `sslmode=require` 改成 `sslmode=disable`。如果启用了 SSL，建议继续使用 `sslmode=require`。

远端数据库用户至少需要：

```sql
GRANT CONNECT ON DATABASE scann_annotation TO scann_sync;
GRANT USAGE, CREATE ON SCHEMA scann_backup TO scann_sync;
ALTER SCHEMA scann_backup OWNER TO scann_sync;
```

### 8.3 手动同步

管理员可以调用：

```text
POST /api/annotation-sync/run
```

默认是增量同步。需要重新回填远端表时可显式执行：

```text
POST /api/annotation-sync/run?full=true
```

同步状态可以通过：

```text
GET /api/annotation-sync/status
```

远端写入使用幂等 upsert，重复执行不会重复生成相同 revision。前端管理员登录后，也可以通过顶部“标注同步”菜单触发增量或全量同步。

## 9. 本地启动

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

## 10. 文档维护说明

如果原生标注平台的任务流、数据库结构或预处理约定发生变化，应优先更新：

- `dataset_pipeline.md`
- 本文档中的概览说明
