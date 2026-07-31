# SCANN v2 分层发现与持续学习实现计划

> 状态：执行中（WP-01～WP-03 已完成，生产分区待审阅激活）
> 基线分支：`codex/refactor-data-integrity`
> 基线提交：`caf95d44`
> 目标：在不破坏现有标注、训练和线上推理闭环的前提下，把系统从“11 类框分类器”演进为“候选检测 → 可用性判断 → 科学分类 → 异常发现 → 人工反馈”的分层发现系统。

## 1. 实施原则

1. **数据完整性优先**
   - 不原地重写现有 16,840 个标注框。
   - 不删除现有训练快照、模型记录、预标注或标注 revision。
   - 新分类口径通过版本化映射派生；确需回填时只新增字段或新表，并保留来源。
   - 所有生产数据库迁移必须先执行 SQLite 在线备份、`integrity_check`、SHA256 和核心计数核对。

2. **先建立可重复评价，再改变算法**
   - 当前最新模型的 `macro F1=0.219`，但该值仍不足以回答“发现能力是否提升”，因为缺少固定的夜晚/视场隔离测试集和候选阶段指标。
   - 第一批算法改动不得直接以线上候选数量或随机 box 验证集准确率作为唯一依据。

3. **适度重构**
   - 复用现有 `DatasetStorage`、训练作业、模型注册、预标注 worker 和前端审核入口。
   - 只在数据契约、评价、候选检测、分层模型和主动学习等自然边界新增模块。
   - 不在第一阶段引入新的数据库服务、消息队列或分布式训练框架。

4. **兼容式发布**
   - 旧模型、旧快照和旧客户端继续可读。
   - 新模型继续输出现有 `ai_suggestion`、`ai_confidence` 和框列表，同时在 metadata 中增加分层概率、OOD 分数和推理追踪。
   - 新模型先离线评价，再 shadow，最后 canary；禁止训练成功后无门槛自动推广。

## 2. 当前基线

### 2.1 数据与模型

- 任务：1,722
- 已标注任务：1,311
- 待标注任务：411
- 当前标注框：16,840
- 最近训练快照样本：16,839
- 类别最高不平衡约：288:1
- `cmos_condensation`：训练支持为 0
- 当前训练目标：固定 11 个 `detail_type`
- 当前主模型：冻结 DINOv2 ViT-B/14 特征编码器 + 轻量分类头
- 最近模型：
  - `macro F1=0.219144`
  - `macro AP=0.370434`
  - `tail recall@1=0.569444`
  - `accuracy=0.308251`
  - 因类别覆盖不足未自动推广

### 2.2 已有能力

- 任务、原始文件、处理产物、当前标注和 revision 已统一进入 `scann_dataset.db`。
- 训练快照、训练 job/run、模型注册和预标注 job 已形成闭环。
- 训练代码已经支持：
  - 按 `task_id` 分组的训练/验证拆分；
  - class-balanced focal loss；
  - weighted sampler；
  - macro F1、macro AP、尾类 recall 和类别覆盖审计。
- 模型产物已有路径约束、原子上传、SHA256 和大小校验。
- 训练与预标注 worker 已部署并使用持久化模型缓存。

### 2.3 主要差距

1. 当前只有训练/验证拆分，没有独立、冻结、按夜晚/视场隔离的 gold test。
2. 当前粗标签 `label` 大量为空，训练实际依赖 `detail_type`；标签语义存在隐式契约。
3. `real/bogus` 是当前审核动作口径，不完全等价于物理现象：
   - 例如 `disappeared_asteroid` 是真实天体行为，但当前映射为 `bogus`。
4. 候选检测仍主要基于 8-bit 正差分、固定阈值、轮廓和规则过滤。
5. 标准参数与放宽参数之间跳变大，缺少每阶段可观测指标。
6. 训练指标尚未与“每幅图误报数、真实目标召回率、人工复核量、发现延迟”绑定。
7. 尚未实现主动学习、审核反馈分析、OOD/异常发现和模型 shadow/canary。
8. SQLite schema 由启动时 `CREATE TABLE IF NOT EXISTS` 和补列维护，缺少正式迁移版本。

## 3. 目标数据与模型契约

### 3.1 标签不再只使用一棵平铺分类树

保留现有 11 个 `detail_type`，新增三个派生维度：

| 维度 | 建议取值 | 用途 |
|---|---|---|
| `review_action` | `keep` / `reject` / `unknown` | 是否进入人工重点复核或后随流程 |
| `phenomenon_family` | `appearance` / `variability` / `moving` / `disappearance` / `persistent_mismatch` / `instrument_artifact` / `unknown` | 表达物理或成像现象 |
| `detail_type` | 保留现有 11 类 | 与现有 UI、历史模型和业务统计兼容 |

首版映射必须由项目负责人和天文领域人员签字确认，特别确认：

- `asteroid`、`disappeared_asteroid` 在不同业务目标下的审核动作；
- `corresponding` 是正常对应、配准问题还是亮度变化；
- `disappeared_star/galaxy` 是否全部视为天气或处理伪影；
- `supernova` 和 `variable_star` 的标注证据要求；
- 零样本类别是否暂时只能输出 `unknown`。

映射落在新模块：

```text
src/scann/ai/taxonomy.py
```

包含：

- `TAXONOMY_VERSION`
- 11 类到 `review_action`、`phenomenon_family` 的版本化映射
- 对缺失 `label` 的非破坏性派生函数
- 未知值和历史值兼容规则
- taxonomy 审计报告

### 3.2 快照文档升级

训练快照从 `version=2.3` 升级到 `3.0`，每个任务增加：

- `task_id`
- `field_key`
- `capture_key`
- `date_obs`
- `night_key`
- `group_key`
- `taxonomy_version`
- `partition_id`
- 每个框的原始标签和派生标签
- 派生来源：`human` / `derived_from_detail_type` / `legacy_default`

旧快照继续按 2.3 读取；新训练默认生成 3.0。

### 3.3 模型输出

新模型格式建议命名为 `hierarchical_classifier_v1`，输出：

```json
{
  "review_action_probs": {"keep": 0.91, "reject": 0.08, "unknown": 0.01},
  "phenomenon_family_probs": {"moving": 0.84, "appearance": 0.09},
  "detail_type_probs": {"asteroid": 0.80, "supernova": 0.05},
  "calibrated_confidence": 0.87,
  "ood_score": 0.12,
  "predicted_taxonomy_version": "scann-discovery-v1"
}
```

兼容输出：

- `ai_suggestion`：仍返回最高概率 `detail_type`
- `ai_confidence`：使用校准后的细类置信度
- `label`：由 `review_action` 映射回旧的 `real/bogus`

## 4. 目标流水线

```text
FITS 新旧图
  -> 校准/配准质量检查
  -> 浮点差分与显著性图
  -> 正负候选生成
  -> 规则质量门与安全上限
  -> 分层模型：keep/reject + family + detail
  -> 概率校准与 OOD
  -> 目录/历史上下文（条件具备后）
  -> 自动过滤 / 人工复核 / 异常队列
  -> 审核结果回流
  -> 主动学习批次
```

实时路径和慢速路径分开：

- **实时路径**：单次 FITS 对、图像 cutout、结构化特征，目标是快速预标注。
- **慢速路径**：同一视场的多次观测、目录交叉匹配和异常排序，目标是减少误报与发现未知对象。

## 5. 分阶段实施

## M0：版本化迁移与基线冻结

**预计：2–3 个工作日**

### 任务

- `M0-01` 新增 `schema_migrations` 表和迁移执行器。
- `M0-02` 把现有建表逻辑保留为基础 schema，将后续变更拆成有编号、有 checksum 的事务迁移。
- `M0-03` 增加只读数据审计命令：
  - 文件存在性；
  - SQLite integrity；
  - 任务、框、revision、prelabel、job、run、model 计数；
  - `label/detail_type` 一致性；
  - 无效路径和孤儿记录；
  - 模型产物存在性与哈希。
- `M0-04` 冻结一次基线训练快照、部署清单和指标报告。
- `M0-05` 给后续实验定义统一随机种子、软件 commit、CUDA/PyTorch 版本和配置归档格式。

### 代码位置

- 新增 `src/scann/core/schema_migrations.py`
- 调整 `src/scann/core/dataset_storage.py`
- 新增 `src/scann/scripts/audit_dataset.py`
- 新增 `tests/test_schema_migrations.py`
- 新增 `tests/test_dataset_audit.py`

### 验收

- 空库可从 0 迁移到最新版本。
- 现有生产数据库副本可迁移，核心计数不变。
- 同一迁移重复执行不产生变化。
- 任一迁移失败时事务回滚。
- 旧版本应用在仅包含增量表/列的新库上仍能启动。

## M1：标签契约、固定数据分区与 gold test

**预计：4–6 个工作日**

### 任务

- `M1-01` 形成 `scann-discovery-v1` 标签说明和映射表。
- `M1-02` 生成标签审计报告：
  - 11 类样本数；
  - 任务数、视场数和观测夜数；
  - 每类可独立分组数；
  - `label` 缺失、冲突和未知值；
  - 单任务大量重复框。
- `M1-03` 实现稳定的任务级三分区：
  - train 70%
  - validation 15%
  - gold test 15%
- `M1-04` 分组优先级：
  1. `night_key + field_key`
  2. 缺少日期时退回 `capture_key/field_key`
  3. 最后才退回 `task_id`
- `M1-05` 将分区写成不可变 manifest：
  - `.scann_control/partitions/<partition_id>.json`
  - 保存 task IDs、group keys、类支持、生成算法版本和 SHA256
- `M1-06` gold test 从训练 UI 隐藏，常规训练 job 不允许读取其标签。
- `M1-07` 快照文档升级到 3.0，但保留 2.3 兼容读取。

### 代码位置

- 新增 `src/scann/ai/taxonomy.py`
- 新增 `src/scann/ai/dataset_partition.py`
- 调整 `src/scann/ai/class_balance.py`
- 调整 `src/scann/native_annotation/training_lifecycle_service.py`
- 新增 `tests/test_taxonomy.py`
- 新增 `tests/test_dataset_partition.py`
- 扩展 `tests/test_training_lifecycle_api.py`

### 验收

- train/validation/test 的 `task_id` 零重叠。
- 可构造时，`night_key + field_key` 零重叠。
- 同一 partition 输入重复生成相同 manifest/hash。
- `cmos_condensation` 等无样本类明确标为 `unsupported`，不伪造指标。
- 原始 box 不被更新；派生标签只出现在快照或新 metadata 中。

## M2：候选检测可观测化与稳定化

**预计：7–10 个工作日**

### M2A：只加观测，不改变结果

- `M2-01` 为每次检测生成 `DetectionTrace`：
  - 图像统计；
  - 配准成功率和偏移；
  - standard/relaxed/sliding/dense 各阶段候选数；
  - AI 前后候选数；
  - 阈值、耗时、回退原因；
  - 最终候选分数分布。
- `M2-02` 将 trace 写入 prelabel metadata 和结构化日志。
- `M2-03` 建立候选阶段离线 evaluator，按 gold test 计算：
  - 标注框中心是否被候选覆盖；
  - recall@IoU/center-distance；
  - 每任务 raw candidates；
  - 每任务 post-filter candidates；
  - 每阶段漏检来源；
  - P50/P95 推理时间。

### M2B：建立注入恢复测试

- `M2-04` 从真实图像估计背景、噪声和近似 PSF。
- `M2-05` 在无标注区域注入不同 SNR、FWHM 和位置的模拟点源。
- `M2-06` 记录 recovery curve：
  - recall vs SNR；
  - recall vs FWHM；
  - 中心/边缘；
  - 不同夜晚和视场。
- `M2-07` 注入产物只进入临时目录或 evaluation artifact，不进入正式数据集。

### M2C：候选算法改进

- `M2-08` 保留浮点数据，减少过早转换为 8-bit。
- `M2-09` 使用 robust background/MAD 构建显著性图，分别检测正、负残差。
- `M2-10` 增加局部特征：
  - SNR、flux difference；
  - FWHM、sharpness、ellipticity；
  - 正负像素比、dipole score；
  - 背景梯度；
  - 边缘距离、饱和/坏像素占比；
  - new/old 质心偏移。
- `M2-11` standard 和 relaxed 改为连续参数策略，避免从 0 突然跳到上千。
- `M2-12` 设置 raw candidate 安全上限，但上限只用于保护资源，不作为提高指标的手段。
- `M2-13` 保留旧 detector，通过配置切换 `legacy` / `significance_v1`。

### 代码位置

- 调整 `src/scann/core/candidate_detector.py`
- 调整 `src/scann/services/detection_pipeline.py`
- 新增 `src/scann/services/candidate_feature_extractor.py`
- 新增 `src/scann/ai/candidate_evaluation.py`
- 新增 `src/scann/ai/source_injection.py`
- 调整 `src/scann/native_annotation/prelabel_worker.py`
- 新增对应单元和回归测试

### 验收门槛

- 固定 gold test 上，候选阶段真实类召回不低于旧流程。
- 目标门槛：真实类候选召回达到 95%；若数据质量不允许，至少较旧流程提升 10 个百分点。
- 放宽回退后的 raw candidate P95 控制在 200/任务以内。
- AI 后候选 P95 控制在 50/任务以内。
- standard 为 0、relaxed 超过 1,000 的任务比例显著下降。
- 注入恢复曲线、版本和随机种子可复现。

## M3：分层分类器基线

**预计：7–9 个工作日**

### 任务

- `M3-01` 保留冻结 DINO 编码器，先验证分层任务本身，不同时改变 backbone。
- `M3-02` 增加三个分类头：
  - `review_action_head`
  - `phenomenon_family_head`
  - `detail_type_head`
- `M3-03` 使用多任务损失：
  - action：class-balanced CE/focal
  - family：class-balanced CE/focal
  - detail：11 类 class-balanced focal
  - 对未知或不可靠标签使用 mask，不强行监督
- `M3-04` 指标增加：
  - keep recall/precision/AP
  - reject recall
  - family macro F1/AP
  - detail macro F1/AP
  - 每类 support
  - confusion matrix
  - Brier score、ECE 或 reliability bins
- `M3-05` 使用 validation 拟合 temperature scaling，gold test 只用于最终报告。
- `M3-06` 模型 checkpoint 写入 taxonomy、partition、模型输入、特征版本和校准参数。
- `M3-07` 推理端兼容旧 classifier 和新 hierarchical classifier。

### 代码位置

- 新增 `src/scann/ai/hierarchical_classifier.py`
- 调整 `src/scann/ai/training_worker.py`
- 调整 `src/scann/ai/inference.py`
- 调整 `src/scann/ai/model_format.py`
- 调整 `src/scann/native_annotation/prelabel_worker.py`
- 扩展 `tests/test_training_worker.py`
- 新增 `tests/test_hierarchical_classifier.py`
- 新增 `tests/test_hierarchical_inference.py`

### 验收门槛

- 相同 partition、相同 frozen encoder 下，分层模型 keep recall 不低于平铺 11 类基线。
- detail macro AP 至少不退化超过 1 个百分点。
- 低支持或零支持类不会被高置信度输出。
- gold test 指标只能由独立 evaluation 命令生成，不参与 epoch 选择。

## M4：三图 + 结构化特征的多模态实验

**预计：10–15 个工作日**

### 输入

- new/science patch
- old/reference patch
- signed difference patch
- M2 产生的结构化特征
- 可获得时加入 FITS 观测元数据

### 实验矩阵

| 实验 | 图像编码 | 结构化特征 | 微调策略 |
|---|---|---|---|
| A | 当前单路 frozen DINO | 否 | 全冻结 |
| B | new/old/diff 共享编码器 | 否 | 全冻结 |
| C | new/old/diff 共享编码器 | 是 | 全冻结 |
| D | new/old/diff 共享编码器 | 是 | 最后 1–2 blocks |
| E | 轻量 CNN 三通道基线 | 是 | 全训练 |

每项至少运行 3 个固定 seed。比较时必须使用同一 partition 和相同候选输入。

### 实现约束

- 先采用 shared encoder + late fusion，避免三套完整 ViT 占用过多显存。
- 结构化特征先经过标准化 MLP，再和图像 embedding 拼接。
- 保存训练集均值/方差；推理端使用 checkpoint 内相同参数。
- 对缺失特征提供 mask，不使用任意常数冒充真实观测。
- 数据增强仅使用符合天文语义的变换：
  - 旋转/翻转；
  - 小幅平移；
  - 噪声和背景变化；
  - PSF/模糊变化；
  - 禁止破坏 new/old/diff 对应关系。

### 代码位置

- 新增 `src/scann/ai/multimodal_classifier.py`
- 调整 `src/scann/ai/dataset.py`
- 调整 `src/scann/ai/training_worker.py`
- 调整 `src/scann/ai/inference.py`
- 新增 `tests/test_multimodal_dataset.py`
- 新增 `tests/test_multimodal_classifier.py`

### 晋升门槛

候选模型同时满足：

- keep recall 不下降超过 1 个百分点；
- detail macro AP 相对基线提升至少 10%，或绝对提升至少 3 个百分点；
- post-filter 人工复核量下降至少 20%；
- 三个 seed 的指标方差在可接受范围；
- 单任务 GPU 推理延迟满足现有 worker 吞吐要求；
- 不出现特定夜晚/视场显著崩溃。

若 D/E 明显不优于 C，则保持 frozen encoder，优先积累数据，不继续增加训练复杂度。

## M5：审核反馈与主动学习

**预计：7–10 个工作日**

### 审核反馈

对 AI prelabel 和最终 revision 做匹配，记录：

- 全接受；
- 部分接受；
- 全拒绝；
- 人工新增框；
- 人工删除框；
- 框位置修正；
- detail/action 改类；
- AI 置信度和最终结果；
- 审核耗时。

匹配建议使用 IoU + 中心距离，并保存匹配算法版本，避免以后算法变化导致历史统计漂移。

### 主动学习

为 411 个待标任务生成优先级：

```text
score =
  0.40 * uncertainty
  + 0.20 * model_disagreement
  + 0.20 * embedding_diversity
  + 0.15 * rare_class_value
  + 0.05 * recency_or_domain_shift
```

约束：

- 同一 field/night 每批最多选 3 个任务；
- 明显重复 embedding 只保留代表样本；
- 每批保留 10% 双人复核；
- 高 OOD 与高业务价值样本单独标记；
- 不自动把 AI 预测写成人工真值。

### 批次

- 第一轮：50 个，用于验证选择策略和 UI。
- 第二轮：75–100 个，根据第一轮收益调整权重。
- 后续：每轮训练后重新计算，不一次性锁定剩余全部任务。

### 代码位置

- 新增 `src/scann/ai/active_learning.py`
- 新增 `src/scann/native_annotation/review_feedback_service.py`
- 新增 `src/scann/native_annotation/active_learning_service.py`
- 调整 `src/scann/native_annotation/routes.py`
- 前端新增审核反馈统计和“主动学习批次”入口
- 新增对应 API、服务和 UI 测试

### 验收

- 相同预算下，主动学习批次的低置信度、稀有类或模型错误发现率高于随机批次。
- 批内同视场重复率受控。
- 反馈事件可追溯到 prelabel、模型、任务和最终 revision。
- 双人复核可计算类别一致率和框匹配一致率。

## M6：异常发现、时间关联与目录上下文

**预计：10–15 个工作日；在 M4/M5 达标后启动**

### OOD/异常通道

首版不直接上复杂生成模型，使用：

- embedding kNN/Mahalanobis 距离；
- 模型集成或多 seed disagreement；
- 最大 softmax、entropy；
- 图像模型与结构化特征模型的分歧。

每天或每批输出 Top-K：

- 高异常、低 artifact 风险；
- 高 keep 概率但细类不确定；
- 与训练分布距离远；
- 不能自动进入 reject。

### 时间和对象关联

只有在 WCS 质量审计通过后实施：

- 从 task/FITS 提取天球坐标；
- 在角距离和时间窗口内关联同一对象；
- 建立简化 light-curve/appearance history；
- 对移动目标保留速度和方向特征；
- 对目录匹配保存目录版本、匹配半径和时间。

首批上下文优先复用已有 MPC/排除服务，再考虑 Gaia、SIMBAD 等目录。

### 验收

- OOD 队列中 artifact 占比低于纯 entropy 排序基线。
- 专家每批只需复核固定 Top-K，不产生无上限队列。
- WCS 有效率低于 95% 时不启用自动对象关联，只记录待改进项。
- 任何目录匹配都可重放、可说明版本和匹配参数。

## M7：shadow、canary、生产推广与监控

**预计：5–8 个工作日**

### 发布状态

新增模型部署状态：

- `registered`
- `offline_passed`
- `shadow`
- `canary`
- `promoted`
- `retired`
- `invalid_artifact`

### 流程

1. 离线 gold test 通过。
2. shadow 运行 3–7 天：
   - 不改变用户看到的预标注；
   - 只记录候选和预测摘要。
3. canary：
   - 按 task hash 固定选择 10%；
   - 可随时恢复到旧模型；
   - 观察审核接受率、人工增删框和候选量。
4. 全量 promoted。
5. 保留前一个可用 artifact 和配置作为一键回滚目标。

### 线上指标

- 输入：
  - 图像像素统计、背景、噪声、FWHM；
  - 配准成功率和偏移；
  - 数据缺失率。
- 候选：
  - standard/relaxed/final 候选数；
  - 空结果率；
  - 候选数 P50/P95/P99。
- 模型：
  - 预测类别分布；
  - entropy/OOD；
  - 置信度分布；
  - 推理错误和耗时。
- 人工闭环：
  - 接受/部分接受/拒绝；
  - 人工新增和删除框；
  - 每任务审核时长。
- 系统：
  - job 延迟、失败率；
  - worker heartbeat；
  - GPU/内存；
  - 数据盘、备份盘和系统盘使用率。

### 自动推广规则

首版关闭 `promote_on_success` 默认值。模型必须同时满足：

- artifact 校验通过；
- taxonomy/partition 可识别；
- 所有必需 gold 指标通过；
- 无不可验证的关键目标类；
- shadow/canary 无明显漂移；
- 人工确认推广。

## 6. 数据库与文件变更

### 6.1 建议新增表

| 表 | 作用 |
|---|---|
| `schema_migrations` | 记录迁移 ID、checksum、应用时间 |
| `dataset_partitions` | 记录不可变分区 manifest、hash、taxonomy |
| `evaluation_runs` | 记录 gold/injection/candidate/model 评价摘要和 artifact |
| `annotation_review_events` | 记录 AI 预标注与人工 revision 的差异 |
| `active_learning_batches` | 记录选择策略、模型、预算和状态 |
| `active_learning_items` | 记录任务得分、原因、排序和审核结果 |
| `model_deployments` | 记录 shadow/canary/promoted/rollback 历史 |

大量逐候选结果不直接塞进 SQLite，写到：

```text
.scann_control/evaluations/<evaluation_run_id>/
  manifest.json
  per_task.jsonl
  metrics.json
  plots/
```

数据库只保存摘要、相对路径和 SHA256。

### 6.2 迁移策略

- M0/M1 只新增表，不改旧表语义。
- M2/M3 优先把新信息放入 metadata；稳定后再决定是否升为列。
- 所有 backfill：
  - 支持 `--dry-run`
  - 支持 task 范围
  - 幂等
  - 输出前后计数和冲突清单
- 禁止用批量 SQL 把空 `label` 直接覆盖成派生值。

## 7. API 计划

建议新增：

```text
POST /api/evaluations
GET  /api/evaluations
GET  /api/evaluations/{run_id}

POST /api/dataset-partitions
GET  /api/dataset-partitions

POST /api/active-learning/batches
GET  /api/active-learning/batches
GET  /api/active-learning/batches/{batch_id}

POST /api/training/models/{model_id}/deployments/shadow
POST /api/training/models/{model_id}/deployments/canary
POST /api/training/models/{model_id}/deployments/promote
POST /api/training/models/{model_id}/deployments/rollback
```

现有 API 保持兼容。任何需要返回新字段的响应均采用 optional 字段，避免旧前端解析失败。

## 8. 测试计划

### 单元测试

- taxonomy 映射、未知值和版本兼容；
- 夜晚/视场分组与分区不重叠；
- schema migration checksum、重复执行和回滚；
- 显著性图、正负检测、dipole/边缘特征；
- 注入源生成和恢复匹配；
- 分层 loss、mask 和指标；
- 概率校准；
- active-learning 各分量和多样性约束；
- OOD 分数。

### 集成测试

```text
正式 revision
  -> snapshot 3.0
  -> training job
  -> hierarchical artifact
  -> registry
  -> shadow prelabel
  -> human revision
  -> review event
  -> active-learning batch
```

### 回归测试

- 旧 2.3 快照仍可训练。
- 旧 11 类模型仍可加载和推理。
- 当前桌面端和 Web 前端仍能显示旧字段。
- candidate detector `legacy` 模式输出与固定 fixture 一致。
- 生产数据库副本迁移后计数和 revision 内容不变。

### 性能测试

- 单 task 推理耗时；
- worker 连续处理 100 个 task 的显存稳定性；
- candidate 上限场景；
- snapshot 和 evaluation artifact 的磁盘增长；
- SQLite 写入和标注并发。

## 9. 生产数据安全与回滚

### 每次数据库迁移前

1. 确认当前容器、commit、image ID 和数据库路径。
2. 对 `scann_dataset.db` 和 `scann_native.db` 执行 SQLite online backup 到 `/mnt/disk4`。
3. 对备份执行：
   - `PRAGMA integrity_check`
   - SHA256
   - 核心表计数
4. 记录备份 manifest、迁移 ID 和预期变更。
5. 先在备份副本运行迁移和测试。
6. 生产迁移只在副本验证通过后执行。

当前可用基线：

- 全量异盘备份：`/mnt/disk4/scann-backups/SCANN-20260731T004200-HKT`
- 部署后数据库快照：`/mnt/disk4/scann-backups/SCANN-postdeploy-20260731T0128-HKT`

### 回滚

- 代码回滚：保留上一版本容器 image ID。
- 模型回滚：`model_deployments` 指回上一有效 artifact。
- 数据库回滚：
  - 增量表/列优先保持不动，由旧代码忽略；
  - 若迁移改变了旧路径，则停写后恢复迁移前 online backup；
  - 恢复后再次执行完整性和计数核对。
- 标注数据回滚继续使用已有 revision 机制，不直接覆盖历史。

### 磁盘约束

- 模型、evaluation artifact、embedding cache 放 `/mnt/disk1`。
- 数据库备份和长期报告放 `/mnt/disk4`。
- 不把大型 checkpoint、DINO cache 或 evaluation cutout 放系统根盘；当前根盘使用率已约 89%。

## 10. 推荐排期与人员

假设：

- 1 名后端/ML 工程师全职；
- 1 名领域标注人员投入约 30%；
- 复用当前 GPU worker；
- 不同时开发新的桌面端大功能。

| 周 | 里程碑 | 主要产物 |
|---|---|---|
| 第 1 周 | M0 + M1 前半 | 迁移框架、数据审计、taxonomy 草案 |
| 第 2 周 | M1 完成 + M2A | 固定 partition/gold test、DetectionTrace |
| 第 3 周 | M2B/M2C | 注入测试、显著性候选 detector |
| 第 4 周 | M2 完成 + M3 | 候选评价报告、分层 frozen baseline |
| 第 5 周 | M4 前半 | 三图与结构化特征数据管线 |
| 第 6 周 | M4 完成 | A–E 实验对比与候选模型 |
| 第 7 周 | M5 | 审核反馈、主动学习首批 50 个 |
| 第 8 周 | M7 前半 | shadow、监控与 canary |
| 第 9–10 周 | M6/M7 | OOD、条件允许时对象关联、生产推广 |

核心 MVP 是 M0–M3，预计 4 周；完成后系统即具备可靠评价、稳定候选和分层模型。
M4–M7 属于收益驱动阶段，只有前一阶段指标证明值得继续时才投入，完整周期约 8–10 周。

## 11. 阶段决策门

### Gate A：是否进入模型重构

M1/M2 完成，且：

- gold test 可复现；
- 候选召回和误报可量化；
- 标签映射已确认。

否则继续修数据和候选层，不进入 M3/M4。

### Gate B：是否微调 DINO

M4 的 frozen 多模态模型先于 partial fine-tune。只有以下情况才微调：

- frozen 模型稳定受限；
- gold test 有足够独立 group；
- 三个 seed 的改善一致；
- GPU/推理成本可接受。

### Gate C：是否全量推广

- gold 指标过门槛；
- shadow 3–7 天无明显漂移；
- canary 审核接受率不低于旧模型；
- 模型 artifact、taxonomy、partition 和代码 commit 全部可追溯；
- 已验证回滚。

## 12. 第一批可直接执行的工作包

建议下一轮开发只领取以下工作包，不同时展开 M4–M7：

1. `WP-01`：schema migration + 数据审计命令。
2. `WP-02`：taxonomy v1 文档和非破坏性标签派生。
3. `WP-03`：按 night/field/task 分组的固定 partition 和 gold test。
4. `WP-04`：候选 DetectionTrace 与离线 evaluator。
5. `WP-05`：模拟源注入和 legacy detector 基线报告。
6. `WP-06`：`significance_v1` detector，保留 legacy 配置回退。
7. `WP-07`：冻结 DINO 的分层分类器基线。

完成 WP-01～WP-07 后召开一次指标评审，再决定是否进入三图多模态、主动学习和 OOD。
