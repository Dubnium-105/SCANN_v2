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

- 状态: 已确认未完成
- 影响: 中
- 模块: `AnnotationDialog`
- 文件:
  - `scann_v2/src/scann/gui/dialogs/annotation_dialog.py`
- 问题描述:
  - `Ctrl+S` 已绑定到 `_save_annotations()`。
  - `_save_annotations()` 当前函数体为 `pass`。
  - 注释说明 v2 会自动持久化，但从用户视角看，显式保存入口仍然是空实现。
- 风险:
  - 用户按 `Ctrl+S` 没有反馈，容易误以为未保存。
  - v1/v2 行为语义不一致，后续维护容易出错。
  - 若以后后端持久化策略变化，这个空方法会成为隐蔽故障点。
- 建议修复:
  - 至少补齐显式反馈逻辑。
  - 如果 v2 的确自动保存，应在该方法中显示“已保存”或“当前模式为自动保存”。
  - 如果 v1 仍需要显式保存，应在这里根据 backend 类型分别处理。
  - 给 `Ctrl+S` 增加可观测行为，不要保留空函数。
- 最小验收标准:
  - 触发 `Ctrl+S` 后一定有明确结果: 真正保存、同步刷新、或显示自动保存提示。
  - 对不同 backend 的行为是确定且可测试的。
- 建议测试:
  - 在 `tests/test_annotation_dialog.py` 中增加快捷键或直接调用 `_save_annotations()` 的测试。
  - 覆盖 v1 backend、v2 backend、无 backend 三种情况。

## P1: “计划任务”菜单仍是占位实现

- 状态: 已确认未完成
- 影响: 中
- 模块: 主窗口帮助/设置入口
- 文件:
  - `scann_v2/src/scann/gui/composition/main_window_builder.py`
  - `scann_v2/src/scann/gui/composition/main_window_wiring.py`
  - `scann_v2/src/scann/gui/main_window.py`
  - `scann_v2/src/scann/gui/controllers/help_controller.py`
- 问题描述:
  - 菜单项“计划任务...”已经出现在 UI 中并完成接线。
  - 实际点击后只显示“计划任务功能开发中，敬请期待”。
- 风险:
  - 用户看到的是一个已经发布的功能入口，但实际不可用。
  - 会干扰功能认知，也容易导致测试/文档与真实能力不一致。
- 可选修复方案:
  - 方案 A: 短期下线入口，先从菜单中移除。
  - 方案 B: 保留入口，但改成明确的“未提供此功能”对话框，并补文档说明。
  - 方案 C: 直接落地一个最小可用版本，比如仅提供本地计划任务配置窗口。
- 建议:
  - 若近期不会开发，优先选方案 A，减少假入口。
  - 若近期会开发，保留入口但要补齐设计文档和任务拆分。
- 最小验收标准:
  - UI 中不再出现“看起来可用但其实只有占位提示”的正式入口。
  - 文档和行为保持一致。
- 建议测试:
  - 若下线，更新主窗口 wiring/builder 相关测试。
  - 若保留，测试中应断言展示的是明确的产品策略，而不是开发中占位文案。

## P2: v1 三联图 AI 建议展示逻辑未完成

- 状态: 已确认未完成
- 影响: 低到中
- 模块: `TripletPreviewPanel`
- 文件:
  - `scann_v2/src/scann/gui/widgets/triplet_preview.py`
  - `scann_v2/src/scann/gui/dialogs/annotation_dialog.py`
- 问题描述:
  - 上层会在 v1 模式中调用 `set_ai_suggestion()`。
  - 该方法内部注释写的是“在面板上叠加 AI 建议信息”。
  - 但实际循环体为 `pass`，当前只更新了 tooltip。
- 风险:
  - 用户无法在界面上看到设计中的 AI 建议提示。
  - 名义上支持的展示能力实际上没有完成。
- 建议修复:
  - 先定义最小展示方式，避免过度设计。
  - 可以先用面板角标、顶部状态条或标签文本展示建议和置信度。
  - 若暂不做叠加 UI，至少把注释改成真实行为，避免误导。
- 最小验收标准:
  - `set_ai_suggestion()` 不再是空循环。
  - 用户能在界面上看到 AI 建议，而不只是 tooltip。
- 建议测试:
  - 在 `tests/test_annotation_dialog.py` 或新增 widget 测试中验证建议文本已显示到控件上。

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
