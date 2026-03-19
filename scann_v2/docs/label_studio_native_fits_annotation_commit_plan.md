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

- 状态：planned
- 目标：viewer 增加 `collectRegions()/applyRegions()`，通过 `postMessage` 与宿主页面通信。
- 主要文件：
  - `bridge/app.py`（viewer HTML 生成逻辑）
  - `tests/bridge/test_viewer_region_protocol.py`（新增）
- 验证：
  - `pytest tests/bridge -q`
- 备注：

## C04 - tasks/pull 下发 annotation_mode 与 region 字段骨架

- 状态：planned
- 目标：在任务数据中显式声明 `annotation_mode=js9_region_primary`，预留 `js9_regions_json`。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_pull_tasks_region_payload.py`（新增）
- 验证：
  - `pytest tests/bridge -q`
- 备注：

## C05 - Label Studio 配置/模板联动

- 状态：planned
- 目标：将 `js9_regions_json` 纳入 LS 可提交结果字段（保留旧框工具作为降级）。
- 主要文件：
  - `docs/` 下 LS 配置文档（新增或更新）
  - `tests/bridge/test_ls_payload_contract.py`（新增）
- 验证：
  - 合约测试 + 手工提交一次
- 备注：

## C06 - 数据库入库增强与审计字段

- 状态：planned
- 目标：在不破坏现有表结构前提下，记录 region 来源与版本（可放扩展字段或日志）。
- 主要文件：
  - `bridge/app.py`
  - `tests/bridge/test_region_storage_metadata.py`（新增）
- 验证：
  - `pytest tests/bridge -q`
- 备注：

## C07 - 端到端回归（pull -> 标注 -> webhook）

- 状态：planned
- 目标：新增最小 E2E 回归，覆盖 region 主链路。
- 主要文件：
  - `tests/bridge/test_pull_webhook_region_e2e.py`（新增）
- 验证：
  - `pytest tests/bridge -q`
- 备注：

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
