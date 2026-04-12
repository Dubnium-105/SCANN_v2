# 原生标注平台预标注架构

本文档描述原生 FITS 标注平台的长期预标注架构，以及当前仓库中的第一批实现边界。

## 1. 目标

预标注链路需要满足：

- 服务器长期运行但不依赖 GPU
- GPU 资源可以部署在内网或办公 PC 上
- 服务器仍然是唯一事实源
- AI 结果与人工正式标注分离，避免混淆
- worker 掉线、网络闪断、模型升级后可恢复

## 2. 总体架构

```text
frontend
   |
   v
native-annotation backend
   |- 任务列表 / 锁 / 正式标注 revision
   |- 预标注 job 队列
   |- 当前 AI draft 存储
   |- worker 心跳与结果回写
   |
   +--> dataset / FITS 文件
           ^
           |
      external GPU worker
      - 主动 claim job
      - 读取 FITS
      - 跑 DetectionPipeline + InferenceEngine
      - complete / fail / heartbeat
```

控制面始终在服务器，worker 只是执行器。

## 3. 数据设计

预标注不复用 `tasks.current_annotation_*` 或 `task_annotation_boxes_current`，而是单独存储。

### 3.1 `prelabel_jobs`

用于持久化异步任务队列。

关键字段：

- `job_id`
- `task_id`
- `requested_by`
- `model_version`
- `input_fingerprint`
- `status`
  - `queued`
  - `claimed`
  - `completed`
  - `failed`
  - `cancelled`
- `priority`
- `claim_worker_id`
- `claimed_at`
- `claim_expires_at`
- `last_heartbeat_at`
- `attempt_count`
- `error_message`
- `result_prelabel_id`

### 3.2 `task_ai_prelabels`

保存每个任务的 AI draft 记录，不作为正式人工 revision。

关键字段：

- `prelabel_id`
- `task_id`
- `job_id`
- `source_view`
- `ai_suggestion`
- `ai_confidence`
- `model_version`
- `input_fingerprint`
- `status`
  - `available`
  - `accepted`
  - `superseded`
  - 预留后续 `hidden`
- `box_count`
- `worker_id`
- `accepted_revision_id`
- `metadata_json`
- `created_at`
- `updated_at`
- `superseded_at`

### 3.3 `task_ai_prelabel_boxes`

保存 AI draft 的 bbox 明细，字段与正式 bbox 结构保持一致：

- `x`
- `y`
- `width`
- `height`
- `label`
- `detail_type`
- `confidence`

### 3.4 `worker_nodes`

记录长期运行的 GPU worker 节点。

关键字段：

- `worker_id`
- `display_name`
- `host_name`
- `device_label`
- `status`
- `capabilities_json`
- `last_seen_at`
- `last_claimed_at`

## 4. 核心原则

### 4.1 AI draft 与人工标注分离

AI 结果不直接写入：

- `annotation_revisions`
- `task_annotation_boxes_current`
- `tasks.current_annotation_count`

这样可以避免：

- 把预标注误判成“人工已完成”
- 干扰领取、统计和回退语义
- 让模型结果污染正式 revision 历史

### 4.2 worker 主动轮询

worker 从服务器主动 claim job，而不是服务器反向连接 PC。

这样更适合：

- NAT 后的办公电脑
- 不稳定公网链路
- 多 worker 扩容

### 4.3 输入幂等

每个 job 都绑定 `input_fingerprint`，由以下信息共同生成：

- `task_id`
- 当前任务文件路径
- 文件大小与修改时间
- `model_version`

只要输入或模型版本变化，就应重新 enqueue。

## 5. API 设计

### 5.1 管理端 / 标注端 API

- `POST /api/prelabels/enqueue`
  - 管理员提交预标注任务
- `GET /api/prelabels/{task_id}`
  - 查询任务当前可用 AI draft

### 5.2 worker API

worker 使用独立 token，不复用 annotator/admin JWT。

- `POST /api/prelabel-jobs/claim`
- `POST /api/prelabel-jobs/{job_id}/heartbeat`
- `POST /api/prelabel-jobs/{job_id}/complete`
- `POST /api/prelabel-jobs/{job_id}/fail`

## 6. 运行流程

### 6.1 enqueue

1. 管理员选择任务或全量任务
2. 后端计算 `input_fingerprint`
3. 若当前已有同输入、同模型的可用 draft，则跳过
4. 否则创建 `queued` job

### 6.2 worker claim

1. worker 上报节点信息
2. 后端回收超时 `claimed` job
3. worker 原子 claim 一个 `queued` job
4. 响应中返回任务路径和模型版本

### 6.3 complete

1. worker 上传 bbox、置信度和 metadata
2. 后端把旧的 `available` draft 标记为 `superseded`
3. 写入新的 `task_ai_prelabels` 和 `task_ai_prelabel_boxes`
4. 把 job 标记为 `completed`

### 6.4 人工接受

1. 前端把已导入的 AI draft 以 `metadata.applied_prelabel` 一并提交
2. 后端保存正式 `annotation_revision`
3. 若 `prelabel_id` 仍然有效，则把该 draft 标记为 `accepted`
4. 记录 `accepted_revision_id`
5. 同输入、同模型再次 enqueue 时默认跳过，避免重复生成同一份草稿

## 7. 前端集成约束

前端打开任务时应区分两层数据：

- 正式人工 revision
- AI draft overlay

当前阶段先提供 AI draft 查询 API 和任务列表摘要字段，后续再补：

- AI overlay 展示
- 一键应用 AI draft
- 选择性接受高置信框

## 8. 运维要求

- worker 进程需支持自动重启
- 队列需要可观测：
  - `queued`
  - `claimed`
  - `failed`
  - 平均等待时长
  - 平均推理时长
- 每条日志至少带：
  - `task_id`
  - `job_id`
  - `worker_id`
  - `model_version`

## 9. Worker 运行配置

当前实现提供独立 worker 入口：

```bash
scann-prelabel-worker
```

或：

```bash
python -m scann.native_annotation.prelabel_worker
```

关键环境变量：

- `SCANN_PRELABEL_SERVER_URL`
  - 标注后端地址，例如 `http://server:8000`
- `SCANN_PRELABEL_WORKER_TOKEN`
  - worker 专用 token
- `SCANN_PRELABEL_WORKER_ID`
  - worker 唯一标识
- `SCANN_PRELABEL_WORKER_NAME`
  - worker 展示名称
- `SCANN_PRELABEL_WORKER_DATASET_ROOT`
  - 可选。本地挂载的数据集根目录；设置后优先走本地读盘
- `SCANN_PRELABEL_WORKER_CONFIG_PATH`
  - 可选。读取现有 `scann_v2_config.json`
- `SCANN_PRELABEL_WORKER_MODEL_PATH`
  - 模型路径；未设置时回退到 `config.model_path`
- `SCANN_PRELABEL_WORKER_MODEL_VERSION`
  - worker 当前提供的模型版本
- `SCANN_PRELABEL_WORKER_MODEL_FORMAT`
- `SCANN_PRELABEL_WORKER_MODEL_BACKBONE`
- `SCANN_PRELABEL_WORKER_COMPUTE_DEVICE`
- `SCANN_PRELABEL_WORKER_IDLE_SECONDS`
- `SCANN_PRELABEL_WORKER_HEARTBEAT_SECONDS`
- `SCANN_PRELABEL_WORKER_REQUEST_TIMEOUT_SECONDS`

检测参数默认从现有配置文件继承，也可以通过环境变量覆盖，例如：

- `SCANN_PRELABEL_THRESH`
- `SCANN_PRELABEL_MIN_AREA`
- `SCANN_PRELABEL_MAX_AREA`
- `SCANN_PRELABEL_CONTRAST_MIN`
- `SCANN_PRELABEL_WORKER_PATCH_SIZE`
- `SCANN_PRELABEL_WORKER_DETECTION_MODE`

### 9.1 资产读取模式

worker 支持两种读取方式：

1. 共享目录模式
   - 配置 `SCANN_PRELABEL_WORKER_DATASET_ROOT`
   - worker 使用 claim 返回的相对路径直接读本地 FITS
2. 远端拉取模式
   - 不配置本地数据集目录
   - worker 通过 `GET /api/prelabel-jobs/{job_id}/fits/{view}` 拉取二进制 FITS

建议同局域网优先使用共享目录模式，跨公网再用远端拉取模式。

## 10. 当前实现边界

当前仓库先落第一批控制面能力：

- 预标注 job 队列表
- 当前 AI draft 存储
- worker claim / heartbeat / complete / fail API
- worker 常驻轮询进程
- worker 受控拉取 FITS 资产接口
- 任务列表中的预标注摘要字段
- 单任务 AI draft 查询 API
- 前端 AI draft 叠层显示
- 前端“应用 AI 草稿 / 移除 AI 导入”交互
- 管理员可在标注页对当前任务发起“重新生成 AI 草稿”
- 人工保存后把已应用 draft 回写为 `accepted`

后续再补：

- worker 本地 spool
- draft 接受率统计
- 模型版本回滚与批量重跑工具
