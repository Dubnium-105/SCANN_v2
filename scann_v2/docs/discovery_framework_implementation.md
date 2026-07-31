# 候选发现与模型治理框架实现说明

## 1. 本轮边界

本轮先完成软件框架，不执行以下业务动作：

- 不激活新的 dataset partition；
- 不生成或改写 gold test 真值；
- 不回填历史标注；
- 不启动正式训练或 M4 A–E 实验；
- 不切换生产 detector；
- 不执行 shadow、canary 或模型推广。

数据库变更仅为新增表。生产应用迁移前仍须先做 SQLite online backup 到异盘，并在备份副本上验证迁移和业务表计数。

## 2. 已完成的软件框架

### 2.1 候选发现

- `DetectionTrace` 记录输入统计、配准、各阶段候选数、阈值、耗时、回退原因、错误和最终分数分布。
- prelabel metadata 保存完整 trace，便于离线回放和线上聚合。
- candidate evaluator 使用 IoU + 中心距离做确定性一对一匹配，输出：
  - 总体和细类召回；
  - precision；
  - raw/post-filter 候选量 P50/P95；
  - 单任务耗时 P50/P95；
  - immutable `manifest.json`、`metrics.json` 和 `per_task.jsonl`。
- 模拟源注入支持固定随机种子、正负极性、禁入区和 recovery curve，注入结果不写回正式数据集。
- detector 支持：
  - `legacy`：默认，保持现网兼容；
  - `significance_v1`：robust background/MAD、正负显著性、形态学过滤和资源安全上限。
- 新结构化特征包括 SNR、flux difference、FWHM、ellipticity、dipole、背景梯度、边缘距离、饱和比例和质心偏移等。

### 2.2 分层与多模态分类

- `hierarchical_v1` 使用共享冻结 encoder 和三个独立头：
  - `review_action`；
  - `phenomenon_family`；
  - `detail_type`。
- loss 支持未知标签 mask、class-balanced focal 和分头权重。
- validation 支持 temperature scaling、Brier score、ECE/reliability bins。
- checkpoint 固定记录 taxonomy、partition、partition hash、feature version、类别顺序、温度和 validation 指标。
- training worker 增加 `hierarchical_frozen` 模式；checkpoint 明确记录 `gold_test_used_for_selection=false`。
- 推理端兼容旧二分类、旧 11 类、frozen feature、`hierarchical_v1`。
- `multimodal_hierarchical_v1` 已具备：
  - new/old/signed-difference 三视图共享 encoder；
  - 同步旋转/翻转；
  - 结构化特征值 + missing mask；
  - 仅用训练集拟合的均值/标准差；
  - checkpoint 构建、装载和现有候选推理链路；
  - 缺失结构化特征的显式零值+mask 策略。

### 2.3 人工闭环和主动学习

- 保存采用过 AI prelabel 的人工 revision 后，系统自动生成幂等 review event。
- review event 固定记录匹配算法版本、全接受/部分接受/全拒绝、人工新增/删除、位置修正、改类、置信度和审核时长。
- 主动学习实现加权打分、同 group 上限、embedding 去重、固定 seed、OOD/高业务价值标记和双人复核抽样。
- AI 选择结果只写批次队列，不写成人工真值。

### 2.4 OOD、关联和监控

- OOD 首版包含 entropy、最大 softmax gap、Mahalanobis、ensemble disagreement 和跨模态分歧。
- 异常队列有固定 Top-K 和 artifact-risk gate，任何结果都带 `auto_reject_allowed=false`。
- 天球/时间关联受 WCS 有效率门控制；低于 95% 时只返回未启用原因。
- detection monitoring 聚合空结果率、候选量 P50/P95/P99、耗时、错误和回退原因。

### 2.5 发布治理

状态机为：

```text
registered
  -> offline_passed
  -> shadow
  -> canary
  -> promoted
  -> retired
```

任意有效阶段可以因 artifact 问题进入 `invalid_artifact`。关键约束：

- artifact 必须存在、非空；有登记 SHA256 时必须匹配；
- shadow 必须引用已完成 evaluation，且不影响可见 prelabel；
- canary 使用稳定 task hash，流量上限 50%，必须人工批准；
- promote 必须通过 artifact、taxonomy、partition、gold metrics、shadow、canary 和人工批准；
- 自动推广始终为 false；
- rollback 只能回到有历史有效 deployment 且 artifact 仍有效的模型。

## 3. 新增数据库表

schema migration 3 仅新增：

- `evaluation_runs`
- `annotation_review_events`
- `active_learning_batches`
- `active_learning_items`
- `model_deployments`

大量逐任务评价结果写入 `.scann_control/evaluations/<run_id>/`，SQLite 只保存摘要、路径和 manifest SHA256。

## 4. API

### 评价

```text
POST /api/evaluations
GET  /api/evaluations
GET  /api/evaluations/{run_id}
```

### 审核反馈和主动学习

```text
POST /api/review-feedback
GET  /api/review-feedback
POST /api/active-learning/batches
GET  /api/active-learning/batches
GET  /api/active-learning/batches/{batch_id}
```

### 发布治理

```text
GET  /api/training/model-deployments
POST /api/training/models/{model_id}/deployments/shadow
POST /api/training/models/{model_id}/deployments/canary
POST /api/training/models/{model_id}/deployments/promote
POST /api/training/models/{model_id}/deployments/rollback
```

### 监控

```text
POST /api/monitoring/detection/aggregate
```

除单任务 review feedback 创建外，上述控制面 API 均限制为 admin。

## 5. 前端

admin Header 新增“发现治理”入口，以只读方式展示：

- evaluation 数量和最近状态；
- active-learning 批次数；
- review event 数；
- deployment 数和最近阶段；
- 自动推广关闭提示。

发布动作的客户端函数已经提供，但当前面板不自动触发任何状态变化。

## 6. 配置与启用顺序

建议顺序：

1. 在备份副本验证 migration 3。
2. 部署代码但保持 detector=`legacy`。
3. 只采集 DetectionTrace。
4. 固定 gold partition 后运行 legacy candidate baseline。
5. 运行 injection evaluator。
6. 离线比较 `significance_v1`，通过 gate 后才切小流量。
7. 使用 `hierarchical_frozen` 训练 validation-only calibrated baseline。
8. M4 A–E 至少三个固定 seed；gold test 只生成最终报告。
9. 模型依次进入 shadow、canary 和人工推广。

## 7. 兼容说明

- 旧 checkpoint 和旧 prelabel response 字段保持兼容。
- 新字段均为 optional。
- detector 默认仍为 `legacy`。
- 现有旧模型手工 promote API 为兼容入口；新模型应统一走 `/deployments/*` 治理 API。
- 本轮没有修改或删除任何既有业务表、标注 revision、模型 artifact 或数据文件。
