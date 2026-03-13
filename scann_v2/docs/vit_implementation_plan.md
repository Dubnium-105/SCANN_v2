# SCANN v2 ViT 全图检测实现计划

> 基于 `vit_full_image_detection_design.md`，将“架构设计”落地为可执行开发计划。
> 目标：在不破坏现有 v2 patch 分类流程的前提下，增量接入 ViT 全图检测主链路。

---

## 1. 范围与非范围

### 1.1 本次范围（In Scope）

- 新增全图检测推理主链路：`InferenceEngine.detect_dense_full_image()`
- 在 `DetectionPipeline` 增加模式分支：`patch | full_image | hybrid`
- 新增全图检测训练数据适配与最小可用训练入口（可先单类目标）
- 增加回归测试：输入输出格式、解码/NMS、fallback 逻辑
- 增加运行期日志与错误回退（模型不可用时自动降级）

### 1.2 非范围（Out of Scope）

- 不改 UI 布局与交互（仅补必要状态文案）
- 不做多类别复杂检测头（先单类 real 候选）
- 不引入外部大型检测框架（如完整 YOLO 训练栈）

---

## 2. 交付目标（DoD）

- 在同一组 new/old FITS 输入上，`full_image` 模式可输出 `list[Candidate]`
- `hybrid` 模式可在 `full_image` 失败时自动回退到现有 `patch` 流程
- 配置项可控制检测模式，默认保持现有行为（`patch`）
- 核心路径有单测覆盖：
  - dense 输出解码
  - NMS 合并
  - fallback 分支
  - 配置解析
- 文档包含：运行方式、参数解释、已知限制

---

## 3. 分阶段计划

## Phase A：推理链路打通（最小可用）

### A1. InferenceEngine 新增 dense 接口

- 文件：`src/scann/ai/inference.py`
- 任务：
  - 新增 `detect_dense_full_image(new_image, old_image, ...)`
  - 输入构造为 3 通道（`diff/new/old`）
  - 调用 `SCANNDetector.forward_dense()`
  - 解码为 `list[Detection]`
- 验收：输出 `Detection(x,y,w,h,confidence)` 坐标在图像范围内

### A2. Dense 解码与 NMS

状态：✅ 已完成（2026-03-13）

- 文件：`src/scann/ai/inference.py`（或独立 helper）
- 任务：
  - 实现 heatmap 阈值、top-k、bbox 解码
  - 复用/统一 NMS 逻辑，避免重复实现
- 验收：构造假输出张量时，解码结果数量与置信度符合预期

### A3. 检测模式配置

状态：✅ 已完成（2026-03-13）

- 文件：`src/scann/core/config.py`、`src/scann/core/models.py`、设置对话框映射处
- 任务：
  - 新增 `detection_mode`: `patch | full_image | hybrid`
  - 默认 `patch`
- 验收：配置序列化/反序列化无破坏性变更

## Phase B：服务层接入与回退

### B1. DetectionPipeline 分支化

- 文件：`src/scann/services/detection_pipeline.py`
- 任务：
  - 按模式分流：
    - `patch`: 现有逻辑
    - `full_image`: 走 dense 检测
    - `hybrid`: 先 dense，再 patch 补召回或失败回退
- 验收：三种模式均可执行，且 `patch` 结果与当前行为一致

### B2. 日志与可观测性

- 文件：`src/scann/services/detection_pipeline.py`、`src/scann/ai/inference.py`
- 任务：
  - 记录模式、阈值、候选数量、回退原因
- 验收：日志足以定位“为什么回退/为什么无结果”

## Phase C：训练链路对齐（最小训练闭环）

### C1. v2 标注到 dense 监督目标转换

状态：✅ 已完成（2026-03-13）

- 文件：`src/scann/ai/dataset.py`（新增/扩展）
- 任务：
  - 从 `annotations.json` 生成 heatmap + bbox target
  - 支持基础增强与归一化
- 验收：单样本可视化检查通过；target 维度与模型输出对齐

### C2. 训练入口与损失组合

状态：✅ 已完成（2026-03-13）

- 文件：`src/scann/ai/trainer.py`、`src/scann/ai/training_worker.py`
- 任务：
  - 新增 dense 检测训练模式（可与分类训练并存）
  - 实装 `focal + bbox` 组合损失
- 验收：能跑通 1 epoch 并输出可保存 checkpoint

## Phase D：回归测试与发布准备

### D1. 单测与集成测试

- 新增测试建议：
  - `tests/test_inference_dense_decode.py`
  - `tests/test_detection_pipeline_modes.py`
  - `tests/test_detection_mode_config.py`
- 验收：新增测试稳定，既有关键测试不回退

### D2. 文档与参数说明

- 更新：
  - `docs/architecture.md`
  - `docs/vit_full_image_detection_design.md`
- 验收：开发与使用说明一致、无冲突

---

## 4. 风险与缓冲

- 小目标热力图监督稀疏，初期召回可能不稳定
  - 缓冲：`hybrid` 保底，先上线可回退版本
- 大图 token 成本高，推理时延波动
  - 缓冲：tile 推理 + top-k 裁剪
- 标注质量不齐影响收敛
  - 缓冲：先过滤低置信度伪标注，优先人工标注样本

---

## 5. 建议开发节奏（1~2 周）

- Day 1-2：Phase A
- Day 3-4：Phase B
- Day 5-7：Phase C
- Day 8-10：Phase D + 文档收口

---

## 6. 最终验收标准（Release Gate）

- 功能：`full_image/hybrid` 在真实数据可运行
- 质量：新增测试通过，关键旧测试不回归
- 稳定：无未捕获异常导致主流程中断
- 可运维：日志可定位模式/阈值/回退原因
- 可回滚：配置切回 `patch` 即恢复旧行为
