# SCANN v2 项目待完成内容清单

> 最后更新：2026年3月13日（第四次更新：ViT 全图检测文档收口）

本文档汇总了项目中所有标记为 TODO 的功能项，按模块分类整理。
带 ✅ 标记的项目已实现并通过测试。

## 2026-03-13 进度核查摘要

- 源码目录 `src/` 中检索到 TODO 共 **1** 处
   - **实质未完成**：1 处（`ExclusionService` 轨道计算接入）
   - **注释遗留**：0 处（`main_window.py` 历史 TODO 注释已清理）
- `tests/` 中测试函数静态统计：**716** 个（按 `def test_` 计数）
- 帮助菜单文档入口已接入项目 Wiki（不再是 placeholder URL）

---

## 目录

- [AI 模块](#ai-模块)
- [GUI 主窗口](#gui-主窗口)
- [服务层](#服务层)
- [优先级说明](#优先级说明)

---

## AI 模块

### 1. SCANNDetector 架构设计 ✅

**当前状态**: 已实现 ViT 编码器 + dense 检测头，并接入 `full_image/hybrid` 推理链路。

**文档入口**:
- `docs/vit_full_image_detection_design.md`
- `docs/vit_implementation_plan.md`

---

## GUI 主窗口

### 2. 图像显示与处理 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~线性拉伸图像显示~~ | `main_window.py:780` | 高 | ✅ | ImageProcessor |
| ~~WCS 坐标同步更新~~ | `main_window.py:1244` | 中 | ✅ | Astrometry 模块 |

**实现摘要**:
- `_on_stretch_changed(black, white)`: 根据闪烁状态获取当前图像，调用 `histogram_stretch(data, black_point, white_point)` 执行线性拉伸，刷新 `image_viewer`
- `_on_mouse_moved(x, y)`: 检查 `_new_fits_header`，调用 `pixel_to_wcs()` → `format_ra_hms()`/`format_dec_dms()` → 更新状态栏
- **测试**: `TestStretchChanged` (3 用例), `TestWCSSync` (2 用例)

---

### 3. 文件菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~打开新图文件夹~~ | `main_window.py:882` | 高 | ✅ | FitsIO, FileManager |
| ~~打开旧图文件夹~~ | `main_window.py:914` | 高 | ✅ | FitsIO, FileManager |
| ~~保存当前图像~~ | `main_window.py:948` | 中 | ✅ | FitsIO |
| ~~另存为带标记图像~~ | `main_window.py:966` | 中 | ✅ | FitsIO, 图像渲染 |
| ~~更新最近打开菜单~~ | `main_window.py:985` | 低 | ✅ | AppConfig |

**实现摘要**:
- `_on_open_new_folder`: `scan_fits_folder()` → 填充 `file_list` → `read_fits()` 加载首张 → 保存 `_new_fits_header`
- `_on_open_old_folder`: 扫描 → `match_new_old_pairs()` 自动配对 → 列表显示配对状态 (✅/🆕/📁) → 加载首对
- `_on_save_image`: `QFileDialog` → `write_fits()`
- `_on_save_marked_image`: `image_viewer.grab()` → `pixmap.save()` 导出 PNG
- `_on_update_recent_menu`: 从 `AppConfig.recent_folders` 填充菜单项
- **测试**: `TestOpenNewFolder` (4), `TestOpenOldFolder` (3), `TestSaveImage` (3), `TestRecentMenu` (2)

---

### 4. 处理菜单功能

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~批量对齐~~ | `main_window.py:1003` | 高 | ✅ | ImageAligner |
| ~~批量处理参数集成~~ | `main_window.py:1082` | 中 | ✅ | BatchProcessDialog, ImageProcessor |

**已完成**:
- `_on_batch_align`: 遍历 `_image_pairs`，逐对 `read_fits()` + `align()` + `write_fits()` 回写对齐结果，统计成功/失败，重新加载当前配对
- `_on_batch_process`: 打开对话框，连接 `process_started` 信号到 `_run_batch_process()`
- `_run_batch_process(params)`: 扫描 FITS 文件 → 按参数执行 `denoise()`/`pseudo_flat_field()` → `write_fits()` 保存到输出文件夹
- **测试**: `TestBatchAlign` (2), `TestBatchProcess` (3)

---

### 5. AI 菜单功能

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~批量检测~~ | `main_window.py:1042` | 高 | ✅ | DetectionService |
| ~~训练对话框信号集成~~ | `main_window.py:1073` | 高 | ✅ | TrainingDialog, Trainer |
| ~~加载 AI 模型~~ | `main_window.py:1095` | 高 | ✅ | InferenceEngine |
| ~~显示模型信息~~ | `main_window.py:1126` | 中 | ✅ | InferenceEngine |

**已完成**:
- `_on_batch_detect`: `DetectionPipeline(inference_engine=...)` → `process_pair()` → `set_candidates()`
- `_on_open_training`: 连接 `training_started` → `_on_training_started(params)`, `training_stopped` → `_on_training_stopped()`
- `_on_training_started(params)`: 接收超参数字典并更新状态栏
- `_on_training_stopped()`: 清理训练线程并更新状态栏
- `_on_load_model`: `InferenceEngine(model_path=path)` → 状态栏显示阈值 → 异常时清空并提示
- `_on_model_info`: 计算 `sum(p.numel())` → `QMessageBox` 显示架构/参数量/阈值/设备
- **测试**: `TestBatchDetect` (3), `TestLoadModel` (3), `TestModelInfo` (2), `TestTrainingIntegration` (3)

---

### 6. 查询菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~远程查询集成~~ | `main_window.py:842` | 高 | ✅ | QueryService, QueryResultPopup |
| ~~菜单触发查询路由~~ | `main_window.py:1142` | 中 | ✅ | 候选体管理 |
| ~~MPC 报告候选传入~~ | `main_window.py:1153` | 中 | ✅ | MpcReportDialog, ObservationReport |

**实现摘要**:
- `_do_query(query_type, x, y)`: WCS 坐标转换 → `QueryService().query_xxx(ra, dec)` → `QueryResultPopup` 显示结果列表
- `_on_menu_query`: 有候选体则用其坐标调 `_do_query()`，否则提示
- `_on_mpc_report`: 遍历候选体 → `pixel_to_wcs()` 转换坐标 → 构造 `Observation` 列表 → `generate_mpc_report()` → `dlg.set_report()`
- **测试**: `TestQueryIntegration` (5), `TestMpcReportIntegration` (3)

---

### 7. 视图菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~MPCORB 轨道叠加~~ | `main_window.py:1167` | 中 | ✅ | ImageViewer |
| ~~已知天体标记~~ | `main_window.py:1173` | 中 | ✅ | ImageViewer |

**实现**: 委托 `image_viewer.set_mpcorb_visible(checked)` / `set_known_objects_visible(checked)`

---

### 8. 配置菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~加载 MPCORB 文件~~ | `main_window.py:1199` | 中 | ✅ | MpcorbParser |
| ~~保存配置~~ | `main_window.py:1183` | 低 | ✅ | Config |

**实现**: `_on_open_preferences` → `SettingsDialog` → `save_config()`; `_on_select_mpcorb_file` → `MpcorbParser(path).load()`

---

### 9. 计划任务

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| 计划任务管理界面 | `main_window.py:1212` | 低 | 🔲 | TaskScheduler |

**备注**: 功能设计待定，当前仅显示占位提示

---

### 10. 帮助菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~在线文档 URL~~ | `main_window.py:1720` | 低 | ✅ | GitHub Wiki |

**实现**: `_on_open_docs()` 已调用 `webbrowser.open("https://github.com/Dubnium-105/SCANN_v2/wiki")`

---

### 11. 右键菜单功能 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~定位候选体~~ | `main_window.py:1267` | 高 | ✅ | Candidate 管理 |
| ~~添加手动候选体~~ | `main_window.py:1284` | 中 | ✅ | 候选体数据结构 |
| ~~复制天球坐标~~ | `main_window.py:1297` | 中 | ✅ | Astrometry, 剪贴板 |

**实现摘要**:
- `_on_context_mpc_report(x,y)`: 搜索 50px 范围内最近候选体并聚焦
- `_on_context_add_candidate(x,y)`: `Candidate(x,y,is_manual=True)` → 追加到列表 → 刷新表格和标记
- `_on_copy_wcs_coordinates(x,y)`: `pixel_to_wcs()` → `HH MM SS.ss ±DD MM SS.s` → 剪贴板
- **测试**: `TestContextAddCandidate` (3), `TestCopyWCSCoordinates` (2)

---

### 12. 图像配对加载 ✅

| 功能 | 文件位置 | 优先级 | 状态 | 依赖 |
|------|---------|--------|------|------|
| ~~配对选择加载~~ | `main_window.py:1319` | 高 | ✅ | FitsIO |

**实现**: `_load_pair(index)` + `_on_pair_selected(index)` 通过 `file_list.currentRowChanged` 信号连接
- **测试**: `TestPairListSelection` (2 用例)

---

## 服务层

### 13. 轨道计算实现

**文件**: `src/scann/services/exclusion_service.py` (第146行)

**当前状态**:
```python
# 如果没有，说明需要实现轨道计算（TODO）
```

**备注**: `mpcorb.py` 已有完整二体轨道计算管线（开普勒方程求解 + 轨道传播），`exclusion_service.py` 需集成调用。

---

## 优先级说明

### 高优先级 (核心工作流) ✅
1. ~~文件加载功能（新图/旧图）~~ ✅
2. ~~批量检测功能~~ ✅
3. ~~模型加载~~ ✅
4. ~~图像对齐~~ ✅
5. ~~线性拉伸显示~~ ✅
6. ~~训练对话框信号集成~~ ✅
7. ~~远程查询功能集成~~ ✅

### 中优先级 (体验增强) ✅
1. ~~保存和导出功能~~ ✅
2. ~~视图叠加功能~~ ✅
3. ~~模型信息显示~~ ✅
4. ~~坐标转换和复制~~ ✅
5. ~~MPCORB 文件加载~~ ✅
6. ~~批量处理对话框参数集成~~ ✅
7. ~~MPC 报告候选传入~~ ✅

### 低优先级 (锦上添花)
1. ~~最近文件菜单~~ ✅
2. 计划任务（功能设计待定）🔲
3. ~~在线文档 URL 更新~~ ✅

---

## 技术债务

1. **测试覆盖** ✅
   - 新增测试文件 `tests/test_main_window_features.py`，包含 35 个测试用例
   - 覆盖文件加载、拉伸显示、批量对齐/检测、模型加载、保存、右键菜单、WCS 坐标等
   - 全套 497 个测试通过，无回归

2. **文档完善**
   - 每个完成的功能需要更新 `docs/architecture.md` 和 `docs/ui_ux_design.md`
   - API 文档需要同步更新

3. **配置管理** ✅
   - `AppConfig` 已新增 `recent_folders: list` 字段
   - 配置文件格式保持向后兼容

4. **TODO 注释清理** ✅
   - `main_window.py` 历史 TODO 注释已完成清理
   - 当前 `src/` 中仅保留 2 处实质性 TODO

---

## 完成检查清单

- [x] 已完成功能的 TODO 注释已移除并替换为实现代码
- [x] 功能已在测试文件中验证 (48 个测试全部通过)
- [x] 相关文档已更新 (本文档)
- [x] 原有 511 个测试无回归
- [ ] 剩余实质 TODO (计划任务/轨道计算) 需继续实现
- [x] 清理遗留 TODO 注释（2处）

---

## 变更日志

| 日期 | 说明 |
|------|------|
| 2026-03-13 | 完成 ViT 全图检测文档收口：`SCANNDetector` 架构 TODO 标记为已完成；`src/` TODO 统计更新为 1 处（仅轨道计算） |
| 2026-02-21 | 清理 `main_window.py` 中 2 处历史 TODO 注释；`src/` TODO 统计更新为 2 处（均为实质未完成项） |
| 2026-02-21 | 执行实现进度核查：源码 TODO 4 处（实质 2 + 注释遗留 2），帮助菜单在线文档状态更新为已完成，补充当前测试函数统计（716） |
| 2026-02-09 | 初始版本，整理所有 TODO 项 |
| 2026-02-09 | 完成 22 个 TODO 方法实现，新增 35 个测试用例，标记已完成项目 |
| 2026-02-09 | 完成剩余集成功能: 训练对话框/批量处理/查询服务/MPC报告，新增 13 个测试，总计 48+463=511 测试全部通过 |
