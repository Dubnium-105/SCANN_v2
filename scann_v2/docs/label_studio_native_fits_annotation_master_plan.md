# Label Studio 原生 FITS 标注（Phase B）总体实施计划

> 版本：v1.0（规划）
> 日期：2026-03-19
> 状态：planned

---

## 1. 目标定义

将当前“JS9 仅用于判读、LS 画板用于标注”的模式，升级为：

1. **JS9 Region 作为主标注输入**（框/圆/多边形优先框）。
2. Label Studio 继续作为任务流转、审核与导出入口。
3. bridge 后端负责 JS9 Region 与现有入库结构（`images`/`bboxes`）互转。

最终效果：标注员直接在 FITS 视图中标注，避免 PNG 语义偏差。

---

## 2. 范围与非范围

### 2.1 本期范围

- JS9 Region 创建/编辑/删除/分类。
- Region 序列化并提交到 Label Studio。
- webhook 解析 Region 并写回 `annotations.db`。
- UI 上新增“Region 同步状态”与错误提示。
- 完整自动化测试 + 手工验收脚本。

### 2.2 非范围

- 天球坐标（WCS）级别标注与导出（本期仅像素坐标）。
- 桌面端 GUI 深度改造。
- 多用户并发冲突解决（仅保证最后提交可追踪）。

---

## 3. 目标架构

## 3.1 数据流

1. `/tasks/pull` 下发 FITS URL + viewer 页面。
2. viewer(JS9) 维护 `regionsState`。
3. 在任务提交前，把 `regionsState` 写入 LS 任务结果字段（建议字段：`js9_regions_json`）。
4. `/webhook/labelstudio` 优先解析 `js9_regions_json`，转换为 `bboxes`。
5. 保留现有 `rectanglelabels` 解析作为降级兼容。

## 3.2 关键对象

- `JS9RegionRecord`
  - `shape`: `box|circle|polygon`
  - `x`,`y`,`width`,`height`（先保证 box）
  - `label`,`detail_type`,`confidence`
- `TaskRecord` 增量字段（若需要）
  - `annotation_mode`: `js9_region_primary`

---

## 4. TDD 分层

### L1 单元测试

- Region JSON schema 校验。
- region -> bbox 转换（边界、负值、越界裁剪）。
- detail_type 与 label 映射。

### L2 API 测试

- viewer 页面包含 region 同步控件。
- webhook 接收 `js9_regions_json` 正常入库。
- 旧 `rectanglelabels` 仍可回写。

### L3 集成测试

- pull -> 标注 -> webhook -> sqlite 全链路。
- 混合任务（有些是旧格式、有些是 region）兼容性。

### L4 手工验收

- 在 FITS 上画框并提交可回看。
- 修改/删除 region 后提交结果正确。
- 无 JS9 时降级策略可用。

---

## 5. 里程碑

- M1：Region 数据模型与后端转换能力（1天）
- M2：viewer 中 Region 编辑与同步（1~2天）
- M3：webhook 主链路切换 + 回归（1天）
- M4：试运行（20+样本）与文档封版（1天）

---

## 6. 风险与缓解

1. **CSP/跨域限制 JS9 脚本与 FITS 访问**
   - 缓解：统一同源域名；保留 fallback iframe。
2. **Region 与 LS 提交时机错位**
   - 缓解：显式“同步 Region”按钮 + 提交前校验。
3. **历史标注兼容性**
   - 缓解：webhook 双通道解析（region 优先，rectanglelabels 兜底）。

---

## 7. 完成定义（DoD）

1. 默认标注路径为 JS9 Region。
2. 自动化测试全部通过。
3. 20 个真实样本回写无回归。
4. 文档补齐：部署、使用、故障排查。

---

## 8. 执行策略

执行以逐提交计划为准：

- [docs/label_studio_native_fits_annotation_commit_plan.md](docs/label_studio_native_fits_annotation_commit_plan.md)

每提交均要求：

1. 代码 + 测试同提交。
2. CI（至少本地 pytest）通过。
3. 文档状态同步更新。
