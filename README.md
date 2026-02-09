# SCANN (Star/Source Classification and Analysis Neural Network)

SCANN v2 是一个专业的天文图像分析工具，用于从新旧 FITS 图像中检测移动天体（如小行星、彗星）和暂现源（如超新星）。该项目采用分层架构设计，支持 AI 辅助检测、闪烁比对、MPC 报告生成等功能。

## 版本说明

本项目包含两个版本：
- **v1（legacy）**: 基于三联图 JPG 的经典版本
- **v2（推荐）**: 基于 FITS 文件的专业版本，功能更强大

本文档主要描述 v2 版本。

## v1 与 v2 对比

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
| 闪烁比对 | 无 | 支持速度调节的实时闪烁 |

## v2 项目结构

```
scann_v2/
├── src/scann/              # 主要源代码
│   ├── app.py             # 应用入口
│   ├── ai/                # AI 模块（检测、训练、推理）
│   ├── core/              # 核心业务逻辑（FITS IO、图像处理）
│   ├── services/          # 服务层（闪烁、检测、查询）
│   ├── gui/               # GUI 组件
│   └── data/              # 数据管理
├── tests/                 # 测试套件
├── docs/                  # 文档（架构、UI/UX 设计）
├── logs/                  # 日志文件
└── pyproject.toml         # 项目配置
```

## v1 项目结构（遗留）

- `SCANN.py`: 主 GUI 应用程序，支持数据下载、联动查看和分类。
- `train_triplet_resnet_augmented.py`: 使用 ResNet-18 进行三联图分类的模型训练脚本，支持数据增强和加权采样。
- `calc_triplet_mean_std.py`: 计算数据集均值和标准差的工具脚本，用于训练时的归一化。
- `dataset/`: 存放训练数据的目录（PNG 格式）。
- `best_model.pth`: 训练好的模型权重文件。
- `requirements.txt`: v1 版本依赖包列表。

## v2 安装指南

### 环境要求

- Python 3.9+
- CUDA-capable GPU（可选，用于 AI 加速）
- Windows / Linux / macOS

### 安装步骤

1. **克隆或下载项目**
   ```bash
   cd scann_v2
   ```

2. **创建虚拟环境（推荐）**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```

3. **安装项目**
   ```bash
   pip install -e .
   ```

   或安装开发版本（包含测试工具）：
   ```bash
   pip install -e ".[dev]"
   ```

### 可选：GPU 支持

如果您有 NVIDIA GPU 并希望加速 AI 推理，请安装 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## v2 使用方法

### 1. 启动应用程序

```bash
# 方式一：直接运行
python src/scann/app.py

# 方式二：使用命令行工具（安装后）
scann
```

### 2. 快速上手

#### 加载数据
1. 点击"打开"按钮，选择新图文件夹和旧图文件夹
2. 程序会自动按文件名配对图像

#### 闪烁比对
- 按 `R` 键或点击"闪烁"按钮切换新旧图
- 使用滑块调节闪烁速度（0.1s - 2.0s）

#### AI 检测
- 点击"AI 检测"按钮，系统自动扫描全图
- 检测结果显示在侧边栏的可疑目标列表
- 点击列表项可跳转到目标位置

#### 标记目标
- 在可疑目标上按 `Y` 标记为真目标
- 按 `N` 标记为假目标
- 真目标会显示方框标记，假目标显示十字标记

#### 生成报告
- 点击"生成报告"导出 MPC 80 列格式
- 支持复制到剪贴板或保存为文本文件

### 3. 快捷键

| 快捷键 | 功能 |
|--------|------|
| `R` | 切换新旧图 / 闪烁 |
| `N` | 标记为假目标 |
| `Y` | 标记为真目标 |
| `I` | 切换反色 |
| 鼠标中键 | 拖动图像 |
| 滚轮 | 缩放图像 |

### 4. 高级功能

#### MPCORB 叠加
1. 在设置中配置望远镜参数（焦距、像素大小、天文台代码）
2. 加载 MPCORB 文件
3. 系统会自动计算并叠加已知小行星位置
4. 按极限星等过滤暗目标

#### 外部数据库查询
- 右键点击可疑目标
- 选择查询服务（VSX、MPC、SIMBAD、TNS）
- 排除已知天体

#### 图像处理
- **直方图拉伸**: 调整显示范围，不改变原始数据
- **去噪点**: 去除热像素
- **伪平场**: 平滑背景
- 所有处理仅影响显示，可另存为新 FITS 文件

#### AI 训练
1. 标记足够的真/假样本
2. 点击"训练模型"打开训练对话框
3. 配置训练参数并开始训练
4. 训练完成后自动加载新模型

### 5. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_logger.py -v

# 查看覆盖率
pytest --cov=src/scann --cov-report=html
```

## v1 使用方法（遗留）

如果您需要使用 v1 版本：

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
- **输出**: 脚本会输出 `mean` 和 `std`，您可以将其复制到训练脚本的 `Normalize` 转换中。

## 数据格式说明

### v2: FITS 文件
- 标准天文 FITS 格式
- 包含完整的头信息（观测时间、坐标、曝光等）
- 支持 16/32 位整数
- 图像对齐以新图为参考，仅移动旧图

### v1: 三联图 (Triplet Images)（遗留）
项目处理的是 **三联图**，尺寸通常为 80x240 像素。这些图被水平切分为三部分：
1. **左 (Diff)**: 差异图。
2. **中 (New)**: 新发现的图像。
3. **右 (Ref)**: 参考图像。

这些图像按指定的通道顺序（如 0, 1, 2）堆叠成 3 通道张量输入神经网络。

## 架构设计

SCANN v2 采用分层架构设计：

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

### 核心模块说明

#### AI Layer
- **SCANNDetector**: 全图目标检测模型（控制显存 ≤ 8GB）
- **InferenceEngine**: GPU 推理管理，支持 CUDA 多线程并行
- **Trainer**: 完整训练流程
- **FitsDataset**: FITS 训练数据集管理

#### Core Layer
- **FitsIO**: FITS 文件读写操作
- **ImageProcessor**: 图像处理（直方图拉伸、反色、去噪、伪平场）
- **ImageAligner**: 图像对齐（以新图为参考，仅移动旧图）
- **CandidateDetector**: 候选体检测
- **Mpcorb**: MPCORB 文件处理和位置计算
- **Astrometry**: 天文坐标转换
- **ObservationReport**: MPC 80列报告生成

#### Service Layer
- **BlinkService**: 闪烁比对服务
- **DetectionService**: 完整检测管线（对齐→检测→评分→排除→排序）
- **QueryService**: 外部数据库查询（VSX/MPC/SIMBAD/TNS）
- **ExclusionService**: 已知天体排除
- **SchedulerService**: 计划任务（定时下载、自动检测）

#### GUI Layer
- **MainWindow**: 主窗口，可折叠侧边栏
- **ImageViewer**: FITS 图像查看器
- **Widgets**: 各种 UI 组件（表格、滑块、标签等）
- **Dialogs**: 对话框（设置、训练、报告等）

## 常见问题

### Q: v1 和 v2 哪个版本更推荐使用？
A: **v2** 是推荐版本。它支持 FITS 格式、全图 AI 检测、MPC 报告生成等更专业的功能。

### Q: GPU 是必须的吗？
A: 不是必须的。CPU 可以运行，但 AI 推理速度会较慢。如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch 可以显著提升速度。

### Q: 如何提高 AI 检测准确率？
A: 标记更多的真/假样本进行训练。建议每类至少 100 个样本，多样性越强效果越好。

### Q: 图像对齐会修改原始 FITS 文件吗？
A: 不会。所有图像处理操作仅影响显示，不会修改原始数据。如需保存处理后的图像，可以另存为新文件。

### Q: 如何配置望远镜参数？
A: 在设置对话框中输入：
- 像素大小（μm 或 arcsec/pix）
- 焦距（mm）
- 相机旋转角（°）
- 天文台代码

### Q: MPC 报告可以直接提交吗？
A: 可以。报告完全符合 MPC 80列格式标准。建议人工复核后再提交。

## 技术栈

- **GUI 框架**: PyQt5
- **AI 框架**: PyTorch, torchvision
- **天文数据处理**: astropy
- **图像处理**: opencv-python, scikit-image
- **数值计算**: numpy
- **测试框架**: pytest, pytest-qt

## 许可证

本项目遵循相关开源许可证。

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

---

**SCANN v2** - 让天文图像分析更高效！

## 附录

### 详细的架构设计

如需了解更详细的架构设计、模块说明和 UI/UX 设计，请参阅 `scann_v2/docs/` 目录下的文档：

- `architecture.md` - 完整的系统架构设计文档
- `ui_ux_design.md` - UI/UX 设计文档
- `new requirements.md` - 详细的功能需求
- `TODO.md` - 开发进度和待办事项

### v1 与 v2 的技术细节对比

| 方面 | v1 | v2 |
|------|----|----|
| **数据格式** | JPG/PNG 三联图 | FITS（完整头信息） |
| **图像处理** | 无对齐 | 以新图为参考自动对齐 |
| **AI 模型** | ResNet-18 分类器 | 全图检测器（显存≤8GB） |
| **输入方式** | 80x240 三联图 | 新旧 FITS 文件夹 |
| **标注方式** | 标注图像类别 | 方框/十字标记目标位置 |
| **检测方式** | 分类真/假 | 检测 + 评分 + 排序 |
| **坐标支持** | 无 | 完整 WCS 天球坐标 |
| **报告生成** | 无 | MPC 80 列标准格式 |
| **已知排除** | 无 | MPCORB + 外部查询 |
| **测试覆盖** | 无 | pytest 完整测试套件 |

### 日志系统

v2 版本内置完善的日志系统，所有操作都会记录到 `logs/scann.log` 文件中。日志包含：
- 应用启动和关闭事件
- 文件加载和保存操作
- AI 检测和训练过程
- 错误和异常信息

### 配置文件

v2 使用 `SCANN_config.json` 存储应用配置，包括：
- 望远镜参数（焦距、像素大小、旋转角）
- 天文台代码
- AI 模型路径
- 界面偏好设置

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