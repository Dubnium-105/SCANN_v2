# Label Studio 原生 FITS 标注（Phase B）逐提交实施清单

> 版本：v1.0（commit-by-commit）
> 日期：2026-03-19
> 状态：planned

---

## 使用说明

- 状态字段：`planned` / `in_progress` / `done`
- 每完成一个提交，更新“状态、验证结果、备注”。
- 不允许跳过依赖提交。

---

## 提交清单

## C01 - Region 数据模型与转换工具

- 状态：done
- 目标：定义 `JS9RegionRecord`，实现 `region -> bbox`、schema 校验。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_js9_region_schema.py`（新增）
  - `tests/bridge/test_js9_region_to_bbox.py`（新增）
- 验证：
  - `pytest tests/bridge -q` ✅ 41 passed
- 备注：
  - 实现 `JS9RegionRecord` Pydantic 模型，支持 box/circle/polygon 三种形状
  - 实现 `js9_region_to_bbox` 函数，支持坐标裁剪和边界检查
  - 创建 18 个 schema 校验测试，覆盖有效/无效数据、边界情况
  - 创建 23 个转换测试，覆盖各种形状转换和边界处理

## C02 - webhook 支持 js9_regions_json（优先）

- 状态：done
- 目标：`/webhook/labelstudio` 优先解析 `js9_regions_json`，保留旧格式兜底。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_webhook_js9_regions.py`（新增）
  - `tests/bridge/test_webhook_backward_compat.py`（新增）
- 验证：
  - `pytest tests/bridge -q` ✅ 63 passed
- 备注：
  - 新增 `_extract_js9_regions_from_task_data()` 函数，支持从 task data 中提取 `js9_regions_json` 字段
  - 新增 `_convert_js9_regions_to_bboxes()` 函数，将 JS9 regions 转换为 bbox 格式
  - 修改 `labelstudio_webhook()` 路由，优先解析 `js9_regions_json`，回退到 `rectanglelabels`
  - 创建 11 个 JS9 region 解析测试，覆盖有效/无效 JSON、列表/字符串格式、各种形状转换
  - 创建 5 个向后兼容性测试，验证回退逻辑和优先级

## C03 - viewer 增加 Region 同步协议

- 状态：done
- 目标：viewer 增加 `collectRegions()/applyRegions()`，通过 `postMessage` 与宿主页面通信。
- 主要文件：
  - `bridge/app.py`（viewer HTML 生成逻辑）
  - `tests/bridge/test_viewer_region_protocol.py`（新增）
- 验证：
  - `pytest tests/bridge -q` ✅ 78 passed
- 备注：
  - 在 viewer HTML 中添加 `regionsState` 变量存储当前 regions
  - 实现 `collectRegions()` 函数，从 JS9 获取当前 regions 并规范化格式
  - 实现 `applyRegions(regions)` 函数，将 regions 应用到 JS9，支持 box/circle/polygon 三种形状
  - 添加 `postViewerMessage(type, payload)` 函数，向宿主页面发送消息
  - 添加 `window.addEventListener('message', ...)` 监听器，处理来自宿主页面的消息
  - 支持三种消息动作：`collectRegions`、`applyRegions`、`getRegions`
  - 保留 region 的 label、detail_type、confidence 属性
  - 添加完善的错误处理（try-catch、条件检查、错误日志）
  - 创建 15 个测试，覆盖 region 同步协议的所有功能

## C04 - tasks/pull 下发 annotation_mode 与 region 字段骨架

- 状态：done
- 目标：在任务数据中显式声明 `annotation_mode=js9_region_primary`，预留 `js9_regions_json`。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_pull_tasks_region_payload.py`（新增）
- 验证：
  - `pytest tests/bridge -q` ✅ 85 passed
- 备注：
  - 修改 `TaskRecord` 模型，新增 `annotation_mode` 和 `js9_regions_json` 字段
  - `annotation_mode` 默认值设为 `js9_region_primary`
  - `js9_regions_json` 预留为 Optional[str]，初始值为 None
  - 创建 7 个新测试，覆盖：
    - annotation_mode 字段正确下发
    - js9_regions_json 字段存在且为 None
    - 新增字段与现有字段向后兼容
    - 多个任务都包含 region 字段
    - 不导入 LS 时任务构建仍包含字段
    - TaskRecord 默认值和显式设置验证
  - 完整测试套件通过（85 passed），无回归

## C05 - Label Studio 配置/模板联动

- 状态：done
- 目标：将 `js9_regions_json` 纳入 LS 可提交结果字段（保留旧框工具作为降级）。
- 主要文件：
  - `bridge/app.py`
  - `docs/label_studio_phaseb_config.md`（新增）
  - `tests/bridge/test_ls_payload_contract.py`（新增）
- 验证：
  - `pytest tests/bridge -q` ✅ 90 passed
- 备注：
  - 新增 `get_label_studio_phaseb_label_config()`，统一输出 Phase B 推荐 LS 模板（`js9_iframe` + `preview_png` + `RectangleLabels` + `js9_regions_json`）。
  - webhook 新增 annotation result 解析能力：优先读取 `from_name=js9_regions_json` 的 TextArea 结果，回退到 task data，再回退 `rectanglelabels`。
  - 修复尺寸解析边界：当首个 result 为 TextArea 时，改为遍历 result 列表解析 `original_width/original_height`，避免误判“缺少图像尺寸”。
  - 合约覆盖：模板字段存在性、result 优先级、`[]` 清空语义（不回退旧框）。

## C06 - 数据库入库增强与审计字段

- 状态：done
- 目标：在不破坏现有表结构前提下，记录 region 来源与版本（可放扩展字段或日志）。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_region_storage_metadata.py`（新增）
- 验证：
  - `pytest tests/bridge/test_region_storage_metadata.py -q` ✅ 2 passed
  - `pytest tests/bridge -q` ✅ 92 passed
- 备注：
  - 新增 region 入库审计常量：`REGION_STORAGE_AUDIT_SCHEMA_VERSION`、`JS9_REGION_SCHEMA_VERSION`、`RECTANGLELABELS_SCHEMA_VERSION`。
  - 新增 `_record_region_storage_metadata()`：将入库审计信息写入 `dataset/.audit/region_storage_audit.jsonl`（JSONL 追加模式）。
  - webhook 写入路径增强：按优先级标记 `region_source`（`annotation_result.js9_regions_json` / `task_data.js9_regions_json` / `rectanglelabels_fallback`）并记录 schema version、annotation_mode、region/bbox 数量。
  - 保持 `images`/`bboxes` 表结构不变，通过外部审计日志实现可追踪性。

## C07 - 端到端回归（pull -> 标注 -> webhook）

- 状态：done
- 目标：新增最小 E2E 回归，覆盖 region 主链路。
- 主要文件：
  - `tests/bridge/test_pull_webhook_region_e2e.py`（新增）
- 验证：
  - `pytest tests/bridge/test_pull_webhook_region_e2e.py -q` ✅ 1 passed
  - `pytest tests/bridge -q` ✅ 93 passed
- 备注：
  - 新增最小 E2E 用例，覆盖链路：`/tasks/pull` 任务下发 -> `js9_regions_json` 提交 -> `/webhook/labelstudio` 入库。
  - 断言 SQLite `images`/`bboxes` 回写、region 审计日志写入与 `annotations.json` manifest 生成。
  - 验证 region 主链路使用 `annotation_result.js9_regions_json` 作为优先来源，且与既有 bridge 用例无回归。

## C08 - 手工验收与文档封版

- 状态：planned
- 目标：完成 MANUAL-01~MANUAL-08，更新使用说明/排障文档。
- 主要文件：
  - `docs/label_studio_inline_fits_tdd_plan.md`
  - `docs/label_studio_native_fits_annotation_master_plan.md`
  - `docs/label_studio_native_fits_annotation_commit_plan.md`
- 验证：
  - 20+ 样本试跑记录
- 备注：

---

## 提交规范建议

- 分支：`feature/phaseb-native-fits-annotation`
- commit message：
  - `feat(bridge): C01 add js9 region schema and converter`
  - `test(bridge): C01 add region schema tests`
- 每个提交必须附带：
  1. 变更说明
  2. 测试结果
  3. 回滚点说明
