# SCANN (Star/Source Classification and Analysis Neural Network)

这是一个用于天文图像（三联图）分类和数据管理的工具集。它包含了数据下载、模型训练、归一化参数计算以及一个基于 PyQt5 的图形用户界面。

## 项目结构

- `SCANN.py`: 主 GUI 应用程序，支持数据下载、联动查看和分类。
- `train_triplet_resnet_augmented.py`: 使用 ResNet-18 进行三联图分类的模型训练脚本，支持数据增强和加权采样。
- `calc_triplet_mean_std.py`: 计算数据集均值和标准差的工具脚本，用于训练时的归一化。
- `dataset/`: 存放训练数据的目录。
  - `positive/`: 正样本 PNG 图像。
  - `negative/`: 负样本 PNG 图像。
- `best_model.pth`: 训练好的模型权重文件。
- `requirements.txt`: 项目依赖包列表。

## 安装指南

首先，请确保已安装 Python 3.7+。然后，使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 脚本使用方法

### 1. 主 GUI 程序 (`SCANN.py`)

这是项目的主要交互界面，集成了数据下载和可视化功能。

- **运行方法**:
  ```bash
  python SCANN.py
  ```
- **功能**:
  - 自动从 NADC 下载 JPG 和 FITS 天文数据。
  - 管理本地数据库（SQLite）中的数据联动信息。
  - 提供图像查看器，支持缩放和分类操作。

### 2. 模型训练 (`train_triplet_resnet_augmented.py`)

用于训练 ResNet-18 分类器。它将 80x240 的三联图切分为三个 80x80 的通道（差异图、新图、参考图）。

- **运行方法**:
  ```bash
  python train_triplet_resnet_augmented.py --data dataset --epochs 30 --batch 32
  ```
- **主要参数**:
  - `--data`: 数据集根目录（默认为 `dataset`）。
  - `--epochs`: 训练轮数。
  - `--batch`: 批次大小。
  - `--lr`: 学习率。
  - `--target_recall`: 目标召回率（用于保存模型）。

### 3. 计算归一化参数 (`calc_triplet_mean_std.py`)

在训练新数据集之前，建议先计算数据集的均值和标准差。

- **运行方法**:
  ```bash
  python calc_triplet_mean_std.py --neg dataset/negative --pos dataset/positive
  ```
- **输出**: 脚本会输出 `mean` 和 `std`，你可以将其复制到训练脚本的 `Normalize` 转换中。

## 数据说明

项目处理的是 **三联图 (Triplet Images)**，尺寸通常为 80x240 像素。这些图被水平切分为三部分：
1. **左 (Diff)**: 差异图。
2. **中 (New)**: 新发现的图像。
3. **右 (Ref)**: 参考图像。

这些图像按指定的通道顺序（如 0, 1, 2）堆叠成 3 通道张量输入神经网络。

# SCANN v2 架构设计文档

## 1. 概述

SCANN v2 (Star/Source Classification and Analysis Neural Network) 是一个天文图像分析工具，
用于从新旧天文图像中检测移动天体（如小行星、彗星）和暂现源（如超新星）。

### 1.1 与 v1 的主要区别

| 特性 | v1 | v2 |
|------|----|----|
| 工作模式 | 三联图 JPG（差异图+新图+参考图） | 新旧 FITS 文件夹直接对比 |
| 文件格式 | JPG/PNG | FITS（含 Header） |
| 图像对齐 | 无 | 以新图为参考，仅移动旧图 |
| AI 模式 | 小裁剪图分类 | 全图检测 + 方框/十字标记 |
| 显存需求 | 无限制 | ≤ 8GB |
| 叠加功能 | 无 | MPCORB 已知小行星叠加 |
| 外部查询 | 无 | VSX/MPC/SIMBAD/TNS |
| 归算报告 | 无 | MPC 80列格式 |

### 1.2 设计原则

- **分层架构**: Core → Service → GUI，严格单向依赖
- **TDD 驱动**: 测试先行，Core 层 100% 可测试
- **FITS 优先**: 所有操作基于 FITS 数据和头信息
- **显存友好**: AI 推理控制在 8GB 以内
- **图像为王**: 所有 UI 控件为图像让路，侧边栏可折叠
- **渐进式披露**: 首屏仅核心功能，高级功能通过菜单按需展开
- **状态可见性**: 当前显示状态（新图/旧图/反色/闪烁）始终有明确视觉反馈

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         GUI Layer (PyQt5)                            │
│  MainWindow │ ImageViewer │ Widgets │ Dialogs                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Widgets:                                                        │ │
│  │  OverlayLabel │ SuspectTable │ HistogramPanel │ BlinkSpeedSlider│ │
│  │  CollapsibleSidebar │ MpcorbOverlay │ CoordinateLabel           │ │
│  │  NoScrollSpinBox                                                │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │ Dialogs:                                                        │ │
│  │  SettingsDialog │ TrainingDialog │ BatchProcessDialog           │ │
│  │  MpcReportDialog │ QueryResultPopup │ ShortcutHelpDialog        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                         Service Layer                                │
│  BlinkService │ DetectionService │ QueryService │ ExclusionService   │
│  SchedulerService                                                    │
├──────────────────────────────────────────────────────────────────────┤
│                          Core Layer                                  │
│  FitsIO │ ImageProcessor │ Aligner │ CandidateDetector │ Astrometry  │
│  Mpcorb │ ObservationReport │ Config                                 │
├──────────────────────────────────────────────────────────────────────┤
│                          AI Layer                                    │
│  SCANNDetector │ InferenceEngine │ Trainer │ FitsDataset             │
│  TargetMarker                                                        │
├──────────────────────────────────────────────────────────────────────┤
│                         Data Layer                                   │
│  Database │ FileManager │ Config                                     │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. 模块设计

### 3.1 Core Layer (`src/scann/core/`)

纯业务逻辑，零 GUI 依赖，100% 单元可测试。

#### fits_io.py - FITS 文件操作
- `read_fits(path) -> FitsImage`: 读取 FITS 文件，返回数据+头信息
- `write_fits(path, data, header, bit_depth)`: 保存为整数 FITS（16/32bit）
- `read_header(path) -> dict`: 仅读取文件头
- `get_observation_datetime(header) -> datetime`: 从头信息提取观测时间
- **约束**: 保存时绝不修改 FITS 文件头

#### image_processor.py - 图像处理
- `histogram_stretch(data, black_point, white_point) -> ndarray`: 直方图拉伸（仅显示）
- `invert(data) -> ndarray`: 反色（仅显示）
- `denoise(data, method) -> ndarray`: 去噪点
- `pseudo_flat_field(data, kernel_size) -> ndarray`: 伪平场

#### image_aligner.py - 图像对齐
- `align(new_image, old_image) -> AlignResult`: 以新图为参考，仅移动旧图
- `batch_align(new_images, old_images) -> list[AlignResult]`: 批量对齐
- **约束**: 绝不移动新图

#### candidate_detector.py - 候选体检测
- `detect_candidates(new_data, old_data, params) -> list[Candidate]`: 检测可疑目标
- `compute_features(candidate, new_data, old_data) -> CandidateFeatures`: 特征计算

#### mpcorb.py - MPCORB 处理
- `load_mpcorb(path) -> list[Asteroid]`: 加载 MPCORB 文件
- `compute_positions(asteroids, datetime, observatory) -> list[SkyPosition]`: 计算位置
- `filter_by_magnitude(asteroids, limit_mag) -> list[Asteroid]`: 按极限星等过滤

#### astrometry.py - 天文坐标
- `pixel_to_wcs(x, y, header) -> (ra, dec)`: 像素坐标→天球坐标
- `wcs_to_pixel(ra, dec, header) -> (x, y)`: 天球坐标→像素坐标
- `plate_solve(image, params) -> WCS`: 天文定位

#### observation_report.py - 观测报告
- `generate_mpc_report(observations, observatory_code) -> str`: 生成 MPC 80列报告
- `format_80col_line(obs) -> str`: 格式化单行

#### config.py - 配置管理
- `AppConfig`: 完整配置数据类
- `TelescopeConfig`: 望远镜/相机参数（像素大小、焦距、旋转角等）
- `load_config() / save_config()`: 持久化

### 3.2 AI Layer (`src/scann/ai/`)

#### model.py - 模型定义
- `SCANNDetector`: 全图目标检测模型（控制显存 ≤ 8GB）
- `SCANNClassifier`: 兼容 v1 的裁剪图分类器

#### inference.py - 推理引擎
- `InferenceEngine`: GPU 推理管理
  - `detect(image) -> list[Detection]`: 全图检测
  - `classify_patches(patches) -> list[float]`: 裁剪图分类
  - CUDA 多线程并行计算

#### trainer.py - 训练管线
- `Trainer`: 完整训练流程
  - `train(config) -> TrainResult`: 训练
  - `evaluate(model, dataset) -> Metrics`: 评估

#### dataset.py - 数据集
- `FitsDataset`: FITS 训练数据集
- `TargetAnnotation`: 目标标注（像素位置 + 类别）

#### target_marker.py - 目标标记
- `mark_target(image, position, marker_type) -> MarkedImage`: 在图上标记
- `save_marked_fits(image, targets, original_header, save_path)`: 保存标记图

### 3.3 Data Layer (`src/scann/data/`)

#### database.py - 数据库
- `CandidateDatabase`: SQLite 候选体数据库（异步写入）
- `LinkageDatabase`: FITS 联动数据库

#### file_manager.py - 文件管理
- `scan_fits_folder(path) -> list[FitsFileInfo]`: 扫描 FITS 文件夹
- `match_new_old_pairs(new_folder, old_folder) -> list[ImagePair]`: 新旧图配对
- `organize_training_data(...)`: 训练数据组织

#### downloader.py - 下载引擎（继承 v1）

### 3.4 Service Layer (`src/scann/services/`)

协调多个 Core 模块完成复杂业务流程。

#### blink_service.py - 闪烁服务
- `BlinkService`: 管理闪烁状态和速度
  - `start(speed_ms)` / `stop()` / `tick() -> which_image`

#### detection_service.py - 检测管线
- `DetectionPipeline`: 完整检测流程
  1. 对齐 → 2. 检测候选 → 3. AI 评分 → 4. 排除已知 → 5. 排序输出

#### query_service.py - 外部查询
- `query_vsx(ra, dec)` / `query_mpc(ra, dec)` / `query_simbad(ra, dec)` / `query_tns(ra, dec)`
- `check_artificial_satellite(ra, dec, datetime)`

#### exclusion_service.py - 已知排除
- `ExclusionService`: 综合 MPCORB + 外部查询排除已知天体

#### scheduler_service.py - 计划任务
- `SchedulerService`: 定时爬取 HMT 目录、自动下载、自动检测

### 3.5 GUI Layer (`src/scann/gui/`)

> 详见 [UI/UX 设计文档](ui_ux_design.md)

#### main_window.py - 主窗口
- **菜单栏**: 文件 | 处理 | AI | 查询 | 视图 | 设置 | 帮助
- **可折叠侧边栏** (240px, Ctrl+B 切换):
  - 文件夹按钮 (新图/旧图)
  - 批量对齐/检测按钮 + 进度条
  - 图像配对列表 (`QListWidget`)
  - 可疑目标表格 (`SuspectTableWidget`)
- **图像区域** (弹性填充，≥ 75% 窗口面积):
  - 浮层状态标签 (NEW/OLD/INV, `OverlayLabel`)
  - FITS 图像查看器 (`FitsImageViewer`)
- **控制栏** (40px):
  - 新图/旧图切换、闪烁 toggle、闪烁速度滑块、反色 toggle
  - 直方图拉伸按钮
  - 标记真/假/下一个
- **状态栏**: 当前图类型 | 像素坐标 | 天球坐标 | 缩放百分比

#### image_viewer.py - FITS 图像查看器
- `FitsImageViewer(QGraphicsView)`:
  - 中键拖拽（新旧图同步移动，共享 viewport 变换矩阵）
  - 滚轮缩放 (锚点在鼠标位置)
  - 方框/十字线候选体标记绘制
  - 左键选点 → `point_clicked` 信号
  - 右键 → `right_click` 信号 → 上下文查询菜单
  - MPCORB 已知小行星叠加层

#### widgets/ - 自定义组件
- `no_scroll_spinbox.py`: 禁用滚轮的 SpinBox / DoubleSpinBox
- `coordinate_label.py`: 可选择复制的坐标标签
- `overlay_label.py` (**新增**): 半透明浮层状态标签 (NEW/OLD/INV)
- `suspect_table.py` (**新增**): 带 AI 评分排序/筛选/右键菜单的可疑目标表格
- `histogram_panel.py` (**新增**): 实时直方图 + 黑白点滑块 (仅调显示)
- `blink_speed_slider.py` (**新增**): 带数值显示的闪烁速度控制 (50~2000ms)
- `collapsible_sidebar.py` (**新增**): 可折叠/展开的侧面板
- `mpcorb_overlay.py` (**新增**): 图像上的已知小行星叠加绘制层

#### dialogs/ - 对话框 (**新增目录**)
- `settings_dialog.py`: 分页设置 (望远镜/天文台/检测/AI/保存/高级)
- `training_dialog.py`: AI 训练配置 + 进度监控 + Loss 曲线
- `batch_process_dialog.py`: 批量降噪/伪平场，另存为 FITS
- `mpc_report_dialog.py`: MPC 80 列报告预览/复制/导出
- `query_result_popup.py`: 外部查询结果浮窗
- `shortcut_help_dialog.py`: 快捷键列表

## 4. 快捷键设计

### 4.1 核心快捷键 (单键，窗口焦点内)

| 快捷键 | 功能 | 作用域 | 条件 |
|--------|------|--------|------|
| R | 切换闪烁 | 窗口内 | 已加载图像 |
| I | 切换反色 | 窗口内 | 已加载图像 |
| Y | 标记为真 | 窗口内 | 有选中候选 |
| N | 标记为假 | 窗口内 | 有选中候选 |
| 1 | 显示新图 | 窗口内 | 已加载图像 |
| 2 | 显示旧图 | 窗口内 | 已加载图像 |
| F | 适配窗口 | 图像区域 | — |
| Space | 下一个候选 | 窗口内 | 有候选列表 |
| 滚轮 | 放大缩小 | 图像区域 | — |
| 中键拖拽 | 拖动图片 | 图像区域 | — |

### 4.2 扩展快捷键 (组合键)

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 打开新图文件夹 |
| Ctrl+Shift+O | 打开旧图文件夹 |
| Ctrl+B | 切换侧边栏 |
| Ctrl+, | 打开设置 |
| Ctrl+S | 保存当前图像 |
| Ctrl+E | 导出 MPC 报告 |
| ← / → | 上/下一组图像配对 |

### 4.3 快捷键约束
- 所有快捷键使用 `Qt.WindowShortcut`，非全局
- 单字母快捷键在文本输入框获得焦点时自动失效

## 5. 数据流

```
FITS文件夹(新) ─┐
                 ├──→ 配对 → 对齐 → 候选检测 → AI评分 → 已知排除 → 可疑列表
FITS文件夹(旧) ─┘                                        ↑
                                                    MPCORB + 外部查询

可疑列表 ──→ 人工复核 (Y/N) ──→ MPC 80列报告
```

## 6. GUI 信号-槽连接

```
文件夹按钮.clicked  → FileManager.scan → file_list 更新 → 加载配对 → ImageViewer
btn_blink.clicked   → BlinkService.toggle → QTimer start/stop → 切换 NEW/OLD
btn_detect.clicked  → DetectionPipeline (工作线程) → SuspectTableWidget 更新
suspect_list.clicked → ImageViewer.centerOn(candidate) + draw_markers
ImageViewer.right_click → QMenu → QueryService.query_xxx → QueryResultPopup
```

## 7. 约束

- **显存**: AI 推理 ≤ 8GB
- **FITS 头**: 任何保存操作不得修改原始 FITS 文件头
- **保存格式**: 整数（16/32bit 可选），不使用浮点
- **快捷键**: 非全局，仅窗口焦点内有效
- **滚轮**: 所有数字输入框禁用滚轮调整
- **图像优先**: 图像区域占窗口 ≥ 75%，侧边栏可折叠
- **暗色主题**: 背景 `#1E1E1E`，图像区域 `#141414`
- **最小窗口**: 1024×768，宽度 < 1200px 时侧边栏自动折叠

MIT