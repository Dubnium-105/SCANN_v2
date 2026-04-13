# 原生标注平台训练闭环

这份文档描述当前仓库里已经落地的长期运行版闭环能力，用来支撑：

`人工标注 -> 训练 -> 模型注册/推广 -> 预标注 -> 人工审核 -> 再训练`

它和 [native_annotation_prelabel.md](./native_annotation_prelabel.md) 的关系是：

- `native_annotation_prelabel.md` 侧重预标注控制面和 GPU 推理 worker
- 本文档侧重训练控制面、模型注册表和训练 worker

## 1. 已实现能力

当前仓库已经具备下面这条后端闭环：

1. 标注数据以当前正式 revision 为准，生成冻结的训练快照
2. 训练快照进入 `training_jobs` 队列
3. 外部 GPU 训练 worker 主动 claim 作业并下载快照
4. worker 调用现有 `TrainingWorker` 训练并上传 checkpoint
5. 后端写入 `training_runs` 和 `model_registry`
6. 可选自动推广新模型
7. 可选自动为指定任务重新 enqueue 预标注

## 2. 数据表

训练闭环新增了四张长期运行表：

- `dataset_snapshots`
  - 记录快照 ID、名称、冻结文档路径、任务数、标注数、元数据
- `training_jobs`
  - 记录训练任务队列状态、目标 backbone、模型版本、模型 ID、超参和 claim 状态
- `training_runs`
  - 记录一次实际训练运行的开始/结束时间、指标、产物路径、worker 信息
- `model_registry`
  - 记录模型主表，包含 `model_id`、`model_version`、`model_backbone`、`snapshot_id`、`training_run_id`、`artifact_path`、`is_promoted`

控制面文件默认落在数据集根目录下：

- 训练快照：`.scann_control/training_snapshots/<snapshot_id>.json`
- 模型产物：`.scann_control/models/<model_id>/<filename>`

## 3. 训练 API

管理员 API：

- `POST /api/training/snapshots`
  - 基于当前正式标注生成冻结快照
- `GET /api/training/snapshots`
  - 查看快照列表
- `POST /api/training/jobs`
  - 创建训练作业
  - 支持直接指定 `snapshot_id`
  - 也支持通过 `snapshot_name` + `snapshot_task_ids` 先生成快照再创建作业
- `GET /api/training/jobs`
  - 查看训练队列
- `GET /api/training/runs`
  - 查看历史训练运行与指标
- `GET /api/training/models`
  - 查看模型注册表
- `GET /api/training/models/promoted?task_type=classification`
  - 查看当前推广中的模型
- `POST /api/training/models/{model_id}/promote`
  - 手工推广模型
  - 支持 `enqueue_prelabels=true`，推广后直接重排预标注
- `GET /api/training/models/{model_id}/artifact`
  - 下载已注册模型的 checkpoint

训练 worker API：

- `POST /api/training-jobs/claim`
- `POST /api/training-jobs/{job_id}/heartbeat`
- `GET /api/training-jobs/{job_id}/snapshot`
- `POST /api/training-jobs/{job_id}/artifact`
- `POST /api/training-jobs/{job_id}/complete`
- `POST /api/training-jobs/{job_id}/fail`

训练 worker 使用独立 token，不复用 annotator/admin JWT。

## 4. 训练作业字段

`POST /api/training/jobs` 里最关键的字段：

- `task_type`
  - 当前支持 `classification` / `detection`
- `model_version`
  - 面向业务的版本号，例如 `cls-v3`
- `model_id`
  - 训练产物唯一 ID；不传时后端自动生成
- `model_backbone`
  - 例如 `ResNet18`、`ResNet34`、`ResNet50`、`ViT_B_16`
- `train_config`
  - 直接传给训练 worker 的超参字典
- `promote_on_success`
  - 训练成功后自动写成 promoted model
- `enqueue_prelabels_on_success`
  - 训练成功后自动重排预标注
- `prelabel_task_ids`
  - 可选。限制只为这批任务重跑预标注；为空时按预标注服务默认行为处理
- `force_prelabel`
  - 即使已有相同输入和相同模型身份的 draft，也允许强制重跑

## 5. 训练 Worker

当前仓库已经提供独立入口：

```bash
scann-training-worker
```

或：

```bash
python -m scann.native_annotation.training_job_worker
```

核心环境变量：

- `SCANN_TRAINING_SERVER_URL`
- `SCANN_TRAINING_WORKER_TOKEN`
- `SCANN_TRAINING_WORKER_DATASET_ROOT`
- `SCANN_TRAINING_WORKER_OUTPUT_ROOT`
- `SCANN_TRAINING_WORKER_ID`
- `SCANN_TRAINING_WORKER_NAME`
- `SCANN_TRAINING_WORKER_HOST_NAME`
- `SCANN_TRAINING_WORKER_DEVICE_LABEL`
- `SCANN_TRAINING_WORKER_TASK_TYPES`
- `SCANN_TRAINING_WORKER_MODEL_BACKBONES`
- `SCANN_TRAINING_WORKER_DEVICE`
- `SCANN_TRAINING_WORKER_IDLE_SECONDS`
- `SCANN_TRAINING_WORKER_HEARTBEAT_SECONDS`
- `SCANN_TRAINING_WORKER_REQUEST_TIMEOUT_SECONDS`

运行方式：

1. worker 周期性轮询 `/api/training-jobs/claim`
2. 认领成功后下载冻结快照
3. 本地调用 `scann.ai.training_worker.TrainingWorker`
4. 上传产物
5. 回写完成状态、训练指标和附加元数据

## 6. 快照与训练器对接

现有 `scann.ai.training_worker.TrainingWorker` 已支持从外部快照文档训练：

- 当提供 `annotations_document_path` 时，训练器优先读取该冻结快照
- 否则仍退回到当前数据集目录中的实时标注文档

这保证了训练过程不会被并发标注写入污染。

## 7. 当前推荐闭环

推荐把长期运行拆成两个 worker 面：

- 训练 worker
  - 负责训练、上传 checkpoint、注册模型
- 预标注 worker
  - 负责基于 promoted model 执行推理并回写 AI draft

推荐运行流程：

1. 标注员完成一批任务
2. 管理员创建训练快照
3. 管理员创建训练作业，并打开 `promote_on_success`
4. 如需立刻推到审核队列，再打开 `enqueue_prelabels_on_success`
5. 训练 worker 成功后，后端自动注册模型并可自动推广
6. 预标注 worker 使用新的 `model_id + model_backbone + model_version` 生成 AI draft
7. 标注员审核并保存正式 revision
8. 新一轮 accepted 数据再进入下一次训练快照

## 8. 当前边界

这一轮已经把“长期运行版”的核心控制面做到了可用，但仍有三块建议继续补：

- 审核反馈分析
  - 目前已记录 `accepted_revision_id`，但还没有独立统计“部分接受 / 全拒绝 / 人工新增删减框”
- 主动学习策略
  - 当前是否重跑预标注仍由管理员选择，还没有基于不确定性或长尾类别做自动重采样
- 管理端 UI
  - 当前已经补了 Web 管理入口和桌面端 worker 控制台，但还没有独立的数据看板、统计页和批量运营页

## 9. UI 入口

当前已经提供两类 UI：

- Web 前端
  - 管理员可在标注页头部打开“训练闭环”面板
  - 可以创建快照、创建训练作业、查看最近作业/运行/模型，并手工推广模型
  - 推广时支持附带限定任务范围的预标注重排
- 桌面端
  - AI 菜单新增“长期运行控制台...”
  - 可以在有 GPU 的 PC 上直接启动预标注 worker 和训练 worker
  - 适合作为长期常驻执行器使用，观察状态、处理数量和日志

## 10. 代码位置

训练闭环核心实现位置：

- [dataset_storage.py](../src/scann/core/dataset_storage.py)
- [training_lifecycle_service.py](../src/scann/native_annotation/training_lifecycle_service.py)
- [training_job_worker.py](../src/scann/native_annotation/training_job_worker.py)
- [routes.py](../src/scann/native_annotation/routes.py)
- [training_worker.py](../src/scann/ai/training_worker.py)
- [TrainingLoopMenu.vue](../frontend/src/components/TrainingLoopMenu.vue)
- [HeaderBar.vue](../frontend/src/components/HeaderBar.vue)
- [worker_console_dialog.py](../src/scann/gui/dialogs/worker_console_dialog.py)

对应测试：

- [test_training_lifecycle_api.py](../tests/test_training_lifecycle_api.py)
- [test_training_job_worker.py](../tests/test_training_job_worker.py)
- [test_worker_console_dialog.py](../tests/test_worker_console_dialog.py)
