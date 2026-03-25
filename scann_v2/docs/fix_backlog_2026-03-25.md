# SCANN v2 待修复清单

最后更新: 2026-03-25

本文档整理了本轮代码排查中确认存在的未完成、功能不全或占位实现，目标是作为接下来修复工作的直接输入。

## 修复顺序建议

1. 先修服务层核心正确性问题
2. 再修已接入但空实现的 GUI 入口
3. 最后补齐低风险的展示型缺口

---

## P0: 已知天体排除使用了简化逻辑

- 状态: 已完成
- 影响: 高
- 模块: `ExclusionService`
- 文件:
  - `scann_v2/src/scann/services/exclusion_service.py`
  - `scann_v2/tests/test_exclusion_service.py`
- 完成说明:
  - `ExclusionService` 已接入 `compute_apparent_positions()`，不再只依赖静态 `ra/dec`
  - 当 `EXPTIME` 可用时，观测时刻改为曝光中点
  - 传播失败时保留静态 `ra/dec` 回退，避免旧路径直接失效
  - 已补充并通过相关测试
- 问题描述:
  - 当前已知天体排除流程没有在观测时刻传播轨道位置。
  - 代码注释明确说明“真实场景中需要计算小行星在观测时刻的位置”。
  - 当前实现简化为直接使用小行星的 epoch 位置。
  - 对没有 `ra/dec` 属性的轨道对象，注释直接标明“需要实现轨道计算（TODO）”。
- 风险:
  - 已知天体误判为未知目标。
  - 已知天体排除结果与观测时刻不一致。
  - 后续查询、标记、报告链路都可能建立在错误候选体上。
- 建议修复:
  - 梳理 `scann_v2/src/scann/core/mpcorb.py` 中现有轨道计算能力，优先复用，不要在 `ExclusionService` 内重复造轮子。
  - 将 `header` 中的观测时间、台站信息接入轨道传播流程。
  - 在 `check_candidates()` 中统一得到“观测时刻的已知天体位置列表”后再做角距离匹配。
  - 为无法计算位置的对象增加显式日志，不要静默跳过。
- 最小验收标准:
  - 给定包含观测时刻的 header，已知天体位置不再直接取 epoch 常量值。
  - 对有轨道参数但无预先 `ra/dec` 的对象，仍能参与排除流程。
  - 新增回归测试覆盖“epoch 位置”和“观测时刻位置”的区别。
- 建议测试:
  - 新增 `tests/test_exclusion_service.py` 用例，覆盖轨道传播后命中/未命中场景。
  - 如果 `mpcorb.py` 已有稳定接口，补一组集成型测试验证 `ExclusionService` 调用链。

## P1: 标注工具显式保存入口为空实现

- 状态: 已完成
- 影响: 中
- 模块: `AnnotationDialog` / `FitsAnnotationBackend`
- 文件:
  - `scann_v2/src/scann/gui/dialogs/annotation_dialog.py`
  - `scann_v2/src/scann/core/fits_annotation_backend.py`
  - `scann_v2/src/scann/core/annotation_backend.py`
  - `scann_v2/tests/test_annotation_dialog.py`
  - `scann_v2/tests/test_fits_annotation_backend.py`
- 完成说明:
  - `_save_annotations()` 已实现明确行为: 未加载数据时告警、v2 显式 flush 到 SQLite、v1 明确提示“即时保存”。
  - v2 模式下的框标签修改和删框已改为通过 `FitsAnnotationBackend` 持久化，不再只改内存。
  - `FitsAnnotationBackend.undo()/redo()` 现在会把回滚后的样本状态重新写回 SQLite。
  - 基础 undo/redo 状态恢复已覆盖 `ai_suggestion` / `ai_confidence`，避免 AI 预标注回滚只改内存不改持久层。
- 验收结果:
  - `Ctrl+S` 触发后已有明确反馈，且 v2 会执行显式全量落盘。
  - 新增回归测试覆盖 bbox 编辑持久化、删框持久化、undo/redo 持久化，以及对话框保存提示。

## P1: “计划任务”菜单仍是占位实现

- 状态: 已完成
- 影响: 中
- 模块: 主窗口帮助/设置入口
- 文件:
  - `scann_v2/src/scann/gui/composition/main_window_builder.py`
  - `scann_v2/src/scann/gui/composition/main_window_wiring.py`
  - `scann_v2/src/scann/gui/main_window.py`
  - `scann_v2/src/scann/gui/controllers/help_controller.py`
- 完成说明:
  - 设置菜单中的 `act_scheduler` 已改为默认隐藏且禁用，不再作为正式 UI 入口展示。
  - 内部挂点和 wiring 保留，避免后续真正落地该功能时还要重新穿线。
  - `open_scheduler()` 的提示文案已从“开发中，敬请期待”改为“当前版本未提供计划任务功能”，避免继续使用占位措辞。
- 验收结果:
  - 正式 UI 中不再出现“计划任务...”这一可点击占位入口。
  - 若通过程序调用该动作，反馈也变成明确的产品状态说明，而不是开发中占位文案。

## P2: v1 三联图 AI 建议展示逻辑未完成

- 状态: 已完成
- 影响: 低到中
- 模块: `TripletPreviewPanel`
- 文件:
  - `scann_v2/src/scann/gui/widgets/triplet_preview.py`
  - `scann_v2/src/scann/gui/dialogs/annotation_dialog.py`
  - `scann_v2/tests/test_triplet_preview.py`
- 完成说明:
  - `TripletPreviewPanel` 已新增可见的 AI 建议提示条，显示建议类别与置信度，不再只写 tooltip。
  - 建议条会根据 `real/bogus` 使用不同配色，作为最小可视化方案落地。
  - `AnnotationDialog._update_v1_display()` 已在样本没有 AI 建议时主动清空提示，避免跨样本残留旧结果。
- 验收结果:
  - `set_ai_suggestion()` 已有真实 UI 行为。
  - 用户在 v1 标注界面可以直接看到 AI 建议，而不必依赖 tooltip。
  - 新增测试覆盖了提示显示、清空，以及对话框跨样本切换时的残留清理。

---

## 可疑项

以下问题看起来像“功能不全”，但还需要结合前端或真实使用流再确认，因此暂不列为本轮必修项。

### Native Annotation 任务锁缺少独立释放/续租接口

- 模块: `native_annotation`
- 文件:
  - `scann_v2/src/scann/native_annotation/routes.py`
  - `scann_v2/src/scann/native_annotation/task_lock_service.py`
- 现状:
  - 有 claim。
  - 有服务层 release。
  - 路由层只有“保存标注时顺带 release”，没有独立 unlock/heartbeat/renew API。
- 风险:
  - 标注时间过长时可能依赖超时回收。
  - 用户主动离开任务时无法显式释放。
- 建议:
  - 在开始修复前，先确认产品预期。
  - 如果前端存在长时间标注场景，应补 heartbeat 或 renew。
  - 如果支持“跳过当前任务”或“退出任务”，建议补独立 release 接口。

---

## 建议开工顺序

1. `ExclusionService` 轨道传播接入
2. `AnnotationDialog._save_annotations()` 补成有行为的显式保存入口
3. 决定“计划任务”是下线还是最小实现
4. 补 `TripletPreviewPanel` 的 AI 建议可视化
5. 评估 `native_annotation` 任务锁是否需要补 release/heartbeat

## 开工前检查

- [ ] 确认 `mpcorb.py` 对外可复用接口，避免重复实现轨道传播
- [ ] 先补测试再改核心逻辑，尤其是 `ExclusionService`
- [ ] 决定“计划任务”入口的产品策略: 下线 or 实现
- [ ] 明确 v1/v2 标注保存语义，统一 `Ctrl+S` 行为
- [ ] 若任务锁要增强，先补接口契约和前端使用时序
