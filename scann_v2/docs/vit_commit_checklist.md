# SCANN v2 ViT 全图检测提交检查单

> 目标：把实现计划拆成“可提交、可回滚、可验收”的 commit 单元。
> 建议每个提交控制在单一主题，避免跨层大改。

---

## 0. 提交通用约束（每个 commit 都要满足）

- [ ] 单一主题：仅做一个明确目标
- [ ] 代码可运行：无语法错误、无导入错误
- [ ] 测试可验证：至少包含与改动直接相关的测试
- [ ] 向后兼容：未显式切换模式时不影响现有 `patch` 行为
- [ ] 文档同步：公共接口/配置变化必须更新文档

---

## Commit 1：dense 推理接口骨架

**目标**：在 `InferenceEngine` 中引入全图 dense 推理入口。

**状态**：✅ 已完成（2026-03-13）

- [x] 新增 `detect_dense_full_image(...)` 方法签名与基础流程
- [x] 输入构造支持 `diff/new/old` 三通道
- [x] 检测模型不存在时返回空结果并给出日志
- [x] 最小单测：空模型、空输入、基本返回类型

**建议验证命令**

- `pytest tests/test_model.py -q`
- `pytest tests/test_detection_service_ai.py -q`

---

## Commit 2：dense 输出解码 + NMS

**目标**：把 `forward_dense` 张量转为 `Detection` 列表。

**状态**：✅ 已完成（2026-03-13）

- [x] 实现 heatmap 阈值筛选
- [x] 实现 top-k 选择
- [x] 实现 bbox 解码（坐标映射回原图）
- [x] 实现或复用 NMS，避免重复框
- [x] 单测覆盖：阈值行为、top-k 行为、NMS 合并

**建议验证命令**

- `pytest tests/test_detection_postprocess.py -q`
- `pytest tests/test_detection_pipeline.py -q`

---

## Commit 3：配置项 detection_mode 接入

**目标**：增加 `patch | full_image | hybrid` 运行模式。

**状态**：✅ 已完成（2026-03-13）

- [x] 配置模型新增字段 `detection_mode`
- [x] 默认值为 `patch`
- [x] 配置读写兼容旧配置文件
- [x] 设置 UI 映射正确（若已接线）
- [x] 单测覆盖配置序列化/反序列化

**建议验证命令**

- `pytest tests/test_config.py -q`
- `pytest tests/test_settings_dialog.py -q`

---

## Commit 4：DetectionPipeline full_image 分支

**目标**：服务层接入 full_image 主链路。

**状态**：✅ 已完成（2026-03-13）

- [x] `full_image` 分支调用 dense 接口
- [x] 结果映射为 `Candidate` 并保持排序规则一致
- [x] exclusion 流程在新分支仍生效
- [x] 单测覆盖分支选择和输出格式

**建议验证命令**

- `pytest tests/test_detection_pipeline.py -q`
- `pytest tests/test_detection_pipeline_regression.py -q`

---

## Commit 5：hybrid 分支与失败回退

**目标**：先 full_image，失败/低置信时回退 patch 或补召回。

**状态**：✅ 已完成（2026-03-13）

- [x] `hybrid` 分支执行顺序明确且可配置
- [x] full_image 异常时自动回退 patch
- [x] 记录回退原因日志（异常/空结果/低分）
- [x] 单测覆盖回退触发条件

**建议验证命令**

- `pytest tests/test_detection_service_ai.py -q`
- `pytest tests/test_detection_controller.py -q`

---

## Commit 6：v2 dense 训练数据适配

**目标**：从 `annotations.json` 生成 dense 监督目标。

**状态**：✅ 已完成（2026-03-13）

- [x] 数据集输出包含 input + heatmap + bbox target
- [x] target 与模型输出空间维度一致
- [x] 标签缺失/异常样本可跳过并记录日志
- [x] 单测覆盖 target 维度与边界框映射

**建议验证命令**

- `pytest tests/test_dataset.py -q`
- `pytest tests/test_annotation_models.py -q`

---

## Commit 7：训练流程接入 dense 模式

**目标**：训练 worker 支持 dense 检测训练闭环。

**状态**：✅ 已完成（2026-03-13）

- [x] 训练参数支持选择 `task_type=classification|detection`
- [x] 损失组合（focal + bbox）可运行
- [x] checkpoint 元数据包含任务类型与关键阈值
- [x] 能完成最小 epoch 并保存模型

**建议验证命令**

- `pytest tests/test_training_dialog.py -q`
- `pytest tests/test_model_format.py -q`

---

## Commit 8：文档收口与发布前检查

**目标**：确保设计、实现、配置说明一致。

**状态**：✅ 已完成（2026-03-13）

- [x] 更新 `architecture.md` 中 AI/Service 章节
- [x] 更新 `vit_full_image_detection_design.md` 状态说明
- [x] 补充运行示例与已知限制
- [x] 清理过期 TODO 或临时日志

**建议验证命令**

- `pytest -q`（若耗时过长可先跑 detection + ai 子集）

---

## 发布前总检查（Release Checklist）

- [x] 新增模式默认关闭（默认 patch）
- [x] 配置切换后行为符合预期
- [x] 主流程异常可回退且不会中断 UI
- [x] 核心日志可追踪：模式、阈值、候选数、回退原因
- [x] 文档、测试、代码三者一致
