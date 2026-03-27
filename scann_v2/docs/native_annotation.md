# 原生 FITS 标注平台

本文档描述当前仓库中的原生 FITS 标注平台，而不是早期设计草稿。

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
  - 任务认领、心跳续租、释放
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

## 4. 数据集约定

平台运行时依赖一个数据集根目录。代码默认从环境变量
`SCANN_NATIVE_DATASET_ROOT` 读取该路径。

当前预处理和任务生成围绕这些目录工作：

- `new/`
- `old/`
- `new_marked/`
- `dataset_raw/`：预处理时保存标准化前的原始文件

预处理入口在 `scann.services.dataset_preprocess_service.DatasetPreprocessService`，
会负责：

- 根据时间信息标准化文件名
- 复用或生成对齐后的成对文件
- 生成已标记裁剪图
- 汇总可分配任务

## 5. 标注存储

当前实现不是单纯的 JSON 文件方案，而是以 SQLite 为主：

- `scann_native.db`
  - revision 主表
  - 当前任务状态
  - 数据集快照
- `annotation_revisions/`
  - 每个任务的 JSONL 历史日志
- `annotations.json`
  - 兼容与快照用途

这意味着仓库中早期关于“未来如何升级存储”的设计文档已经不再是主参考。

## 6. 前端职责

前端位于 `frontend/src/`，当前核心模块包括：

- `views/AnnotationView.vue`
  - 标注主页面
- `components/CanvasPanel.vue`
  - 画布显示与交互
- `components/HeaderBar.vue`
  - 顶部操作区域
- `components/InspectorPanel.vue`
  - 侧边检查面板
- `services/*.js`
  - 认证、FITS 读取、标注、历史、任务接口封装
- `composables/`
  - 图像加载、闪烁控制、缓存池等复用逻辑

## 7. 本地启动

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

## 8. 文档维护说明

原生标注平台此前有多份设计稿、TDD 计划和存储升级草稿。随着代码落地，
这些文档已经被删去，后续请以当前实现和本说明为准。
