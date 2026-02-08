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

## 2. 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    GUI Layer (PyQt5)                   │
│  MainWindow │ ImageViewer │ Settings │ TrainingDialog  │
├──────────────────────────────────────────────────────┤
│                   Service Layer                        │
│  BlinkService │ DetectionService │ QueryService │ ...  │
├──────────────────────────────────────────────────────┤
│                    Core Layer                          │
│  FitsIO │ ImageProcessor │ Aligner │ Detector │ AI     │
├──────────────────────────────────────────────────────┤
│                    Data Layer                          │
│  Database │ FileManager │ Config                       │
└──────────────────────────────────────────────────────┘
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

#### main_window.py
- 紧凑布局，最大化图像显示区域
- 两个按钮：显示新图/旧图
- 可疑目标列表（AI评分、坐标可复制）

#### image_viewer.py
- 中键拖拽（新旧图同步移动）
- 滚轮缩放
- 支持方框/十字线标记显示

#### widgets/no_scroll_spinbox.py
- 所有数字输入禁用滚轮

## 4. 快捷键设计

| 快捷键 | 功能 | 作用域 |
|--------|------|--------|
| r | 切换闪烁 | 窗口内 |
| n | 标记为假 | 窗口内 |
| y | 标记为真 | 窗口内 |
| 滚轮 | 放大缩小 | 图像区域 |
| i | 切换反色 | 窗口内 |
| 中键拖拽 | 拖动图片 | 图像区域 |

## 5. 数据流

```
FITS文件夹(新) ─┐
                 ├──→ 配对 → 对齐 → 候选检测 → AI评分 → 已知排除 → 可疑列表
FITS文件夹(旧) ─┘                                        ↑
                                                    MPCORB + 外部查询
```

## 6. 约束

- **显存**: AI 推理 ≤ 8GB
- **FITS 头**: 任何保存操作不得修改原始 FITS 文件头
- **保存格式**: 整数（16/32bit 可选），不使用浮点
- **快捷键**: 非全局，仅窗口焦点内有效
- **滚轮**: 所有数字输入框禁用滚轮调整
