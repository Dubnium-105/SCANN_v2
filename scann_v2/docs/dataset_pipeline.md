# 数据集处理与数据库设计

本文档描述 SCANN v2 当前的数据集预处理流程、任务生成规则，以及统一数据集数据库 `scann_dataset.db` 的表结构和职责边界。

目标是解决以下问题：

- Windows 资源管理器不允许同目录重名，但未处理数据集中可能存在“同一天区的多次新图”和“同一个旧图被多个任务复用”
- 旧图不应为了配对而被复制多份
- 本地查看、本地标注、在线标注、训练读取需要共享同一套任务与路径真相
- 当数据规模达到 `10000` 张图片、数千个标注时，纯 JSON 清单会成为性能瓶颈

## 1. 总体原则

当前方案采用“文件系统保存二进制 FITS，SQLite 保存结构化元数据”的混合设计：

- 原始输入文件仍然保存在数据集目录中
- 预处理后的对齐裁剪产物仍然保存在 `new/`、`old/`、`new_marked/`
- 所有任务、原始文件、产物路径、当前标注、修订历史、锁状态统一写入 `scann_dataset.db`
- `preprocessed_tasks.json` 和 `annotations.json` 仅保留为兼容/导出视图，不再是主索引

这意味着：

- “文件名”不再是主键
- “任务”不再通过重新扫目录推断
- 下游链路优先按数据库中的精确路径读取文件

## 2. 目录约定

数据集根目录当前约定如下：

```text
dataset/
|-- dataset_raw/
|   |-- new/          # 用户放入原始新图
|   |-- old/          # 用户放入唯一旧图
|   `-- new_marked/   # 用户放入带标记的新图
|-- new/              # 预处理输出：对齐裁剪后的新图
|-- old/              # 预处理输出：对齐裁剪后的旧图
|-- new_marked/       # 预处理输出：对齐裁剪后的带标记新图
|-- scann_dataset.db  # 数据集主数据库
|-- preprocessed_tasks.json
`-- annotations.json
```

输入侧的关键约束是：

- `dataset_raw/old/` 中每个“原始文件名对应的天区”只放一张旧图
- `dataset_raw/new/` 中可以存在多个同一天区的新图，例如资源管理器自动生成的 `field_001 (2).fits`
- `dataset_raw/new_marked/` 与 `dataset_raw/new/` 对应，允许同样的重命名模式

## 3. 预处理流程

入口：

- `scann.services.dataset_preprocess_service.DatasetPreprocessService.prepare_dataset`

当前执行顺序如下：

1. 确保目录结构存在
2. 兼容旧输入目录
3. 扫描 `dataset_raw/*` 并登记原始文件
4. 规划任务
5. 对每个任务生成或复用对齐裁剪产物
6. 汇总为可消费任务列表
7. 将兼容清单写回 `preprocessed_tasks.json`

### 3.1 旧目录兼容

如果用户仍把文件直接放在：

- `new/`
- `old/`
- `new_marked/`

预处理服务会把这些文件迁移到 `dataset_raw/*`，并保留旧的标准化命名兼容行为。这个分支仅用于过渡，主流程应当以 `dataset_raw/*` 为准。

### 3.2 扫描原始文件

扫描目标：

- `dataset_raw/new`
- `dataset_raw/old`
- `dataset_raw/new_marked`

每个 FITS 文件都会抽取并写入 `raw_assets`：

- `asset_role`：`new` / `old` / `new_marked`
- `field_name` / `field_key`
- `capture_key`
- `relpath`
- `file_name` / `file_stem`
- `date_obs`
- 文件大小、修改时间

其中：

- `field_key` 用于表达“同一天区”
- `capture_key` 用于表达“同一次采集”

归一化规则由 `DatasetStorage` 提供，核心处理包括：

- 去掉时间前缀 `YYYYMMDDThhmmss__`
- 去掉 `FW_` / `fw_` / `Fw_`
- 去掉 `__aligned_crop`
- 去掉资源管理器复制后缀 ` (2)`、` (3)` 等

因此：

- `field_001.fits`
- `field_001 (2).fits`
- `FW_field_001.fits`

都可以归一到同一个 `field_key`

而 `capture_key` 会保留复制后缀以区分不同新图任务。

### 3.3 任务规划

任务规划以 `new` 为驱动，即：

- 每个 `new` 原始文件生成一个任务
- 每个任务最多关联一张 `old`
- 每个任务最多关联一张 `new_marked`

配对规则如下。

#### 旧图选择

旧图通过 `field_key` 选择：

- `dataset_raw/old/field_001.fits`
- `dataset_raw/new/field_001.fits`
- `dataset_raw/new/field_001 (2).fits`

这两个新图任务都会复用同一个 `old_asset_id`

这正是数据库化后的核心收益之一：旧图只保留一份文件和一条原始资产记录。

#### 带标记新图选择

带标记新图优先按 `capture_key` 精确配对：

- `field_001.fits` 优先配 `field_001.fits`
- `field_001 (2).fits` 优先配 `field_001 (2).fits`

若找不到完全相同的 `capture_key`，则退回到 `field_key` 级别顺序匹配。

### 3.4 任务 ID 规则

任务 ID 由以下信息组成：

- 优先使用 `DATE-OBS` 提取出的 `date_token`
- 再拼接 `field_name`
- 如果仍冲突，则追加 `__01`、`__02`

示例：

- `20260115T203000__field_001`
- `20260115T204500__field_001`
- `20260115T204500__field_001__01`

注意：

- “同一天区的不同新图”通常会因为 `DATE-OBS` 不同而自然得到不同任务 ID
- 只有当 `date_token + field_name` 仍重复时才会追加序号

### 3.5 对齐裁剪产物

每个任务会生成或复用以下产物：

- `new/{task_id}__aligned_crop.fts`
- `old/{task_id}__aligned_crop.fts`
- `new/{task_id}__aligned.marker`
- `old/{task_id}__aligned.marker`
- 可选：`new_marked/{task_id}__aligned_crop.fts`

对齐流程为：

1. 读取任务对应的原始 `new` 和 `old`
2. 以 `new` 为参考，对 `old` 做亮度匹配
3. 进行图像对齐
4. 计算有效重叠区域
5. 将 `new` 与 `old` 裁成同一视场
6. 如果存在 `new_marked`，按同一裁剪框裁出带标记图

状态会写回 `tasks.preprocess_status`：

- `pending`
- `ready`
- `missing_old`
- `align_failed`
- 以及后续标注/占用态

## 4. 数据库主表设计

主数据库文件：

- `scann_dataset.db`

实现位置：

- `src/scann/core/dataset_storage.py`

### 4.1 `raw_assets`

原始文件资产表。

一条记录对应 `dataset_raw/*` 中的一个文件。

关键字段：

- `asset_id`：主键
- `asset_role`：`new` / `old` / `new_marked`
- `field_key`
- `field_name`
- `capture_key`
- `relpath`
- `file_name`
- `file_stem`
- `date_obs`
- `status`：`active` / `missing`

职责：

- 记录原始输入文件
- 支持重新扫描后的增量同步
- 为任务表提供可复用的 `old_asset_id`

### 4.2 `tasks`

任务主表。

一条记录对应一个可查看、可标注、可训练消费的任务。

关键字段：

- `task_id`：主键
- `field_key`
- `field_name`
- `capture_key`
- `new_asset_id`：唯一
- `old_asset_id`：可为空，可被多个任务共享
- `new_marked_asset_id`：可为空
- `preprocess_status`
- `current_source_view`
- `current_label`
- `current_detail_type`
- `current_annotation_count`
- `current_ai_suggestion`
- `current_ai_confidence`
- `annotation_updated_at`
- `local_viewed_at`
- `local_annotation_status`
- `online_annotation_status`
- `claim_client_id`
- `claim_locked_at`
- `claim_expires_at`
- `crop_x0/x1/y0/y1`
- `align_dx/dy`

这里有两个很重要的约束：

- `new_asset_id` 唯一，保证“每张新图只生成一个任务”
- `old_asset_id` 不唯一，允许“多个任务共享同一张旧图”

### 4.3 `task_artifacts`

任务产物表。

一条记录对应一个任务的一个处理结果文件。

`artifact_role` 当前包括：

- `aligned_new`
- `aligned_old`
- `aligned_new_marked`
- `new_marker`
- `old_marker`

职责：

- 将任务 ID 映射到实际产物路径
- 避免下游通过文件名反推路径

### 4.4 `task_annotation_boxes_current`

当前标注框表。

一条记录对应任务当前版本中的一个标注框。

关键字段：

- `task_id`
- `box_index`
- `x/y/width/height`
- `label`
- `detail_type`
- `confidence`

职责：

- 保存“当前生效”的标注结果
- 供桌面端、在线端、训练直接读取

### 4.5 `annotation_revisions`

标注修订头表。

一条记录对应一次提交或一次回滚。

关键字段：

- `revision_id`
- `task_id`
- `source_view`
- `parent_revision_id`
- `rollback_of_revision_id`
- `submitted_by`
- `origin`：`local` / `online` / 其他来源
- `saved_at`
- `metadata_json`

### 4.6 `annotation_revision_boxes`

标注修订明细表。

一条记录对应某个 revision 下的一个标注框。

职责：

- 存储历史版本的完整框集合
- 支持历史查询、差异计算、回滚

## 5. 状态字段设计

### 5.1 预处理状态

`tasks.preprocess_status` 统一表达任务当前所处阶段。

当前常见值：

- `pending`
- `ready`
- `viewed`
- `annotated`
- `claimed`
- `missing`
- `missing_old`
- `align_failed`

说明：

- `ready` 表示已生成可消费产物
- `claimed` 表示在线端已被某客户端领取
- `align_failed` 虽然没有对齐产物，但任务记录仍然存在，可用于排障

### 5.2 本地/在线状态分离

当前设计中，本地与在线状态单独落字段：

- `local_annotation_status`
- `online_annotation_status`

这样可以分别回答：

- 桌面端是否已看过该任务
- 桌面端是否已标过
- 在线端是否已标过

避免多个入口互相覆盖同一个“是否已标注”的结论。

### 5.3 锁状态

任务锁直接落在 `tasks` 表中：

- `claim_client_id`
- `claim_locked_at`
- `claim_expires_at`

在线标注的领取、续租、释放不再依赖纯内存结构，数据库成为单机内的锁真相来源。

## 6. 本地查看 / 本地标注 / 在线标注

### 6.1 本地查看

桌面端通过 `collect_preprocessed_tasks()` 或数据库中的 `task_artifacts` 获取任务列表。

加载任务后会更新：

- `tasks.local_viewed_at`
- 必要时将 `preprocess_status` 从 `ready` 推进到 `viewed`

### 6.2 本地标注

桌面端标注通过 `FitsAnnotationStorage` 写入：

- `tasks.current_*`
- `task_annotation_boxes_current`
- `local_annotation_status`

导出 `annotations.json` 时，内容来自数据库当前视图，而不是反过来驱动数据库。

### 6.3 在线标注

在线标注通过 `AnnotationService` 写入：

- 当前标注表
- 修订头表
- 修订框表
- `online_annotation_status`

在线回滚并不会修改原始资产或任务主键，只会：

- 追加新的 rollback revision
- 将当前标注重置为目标 revision 的内容

## 7. 下游消费方式

### 7.1 训练

训练集读取当前优先级如下：

1. 使用数据库导出的 `paths`
2. 使用任务产物表定位 `aligned_new` / `aligned_old`
3. 仅在兼容旧数据时，回退到基于名字的配对

因此：

- 训练不再要求通过 `new/old` 目录重新扫出样本主键
- 同名任务不会再因为文件名归一化而串样本

### 7.2 兼容清单

以下文件仍可能存在：

- `preprocessed_tasks.json`
- `annotations.json`

但它们现在只承担：

- 兼容旧接口
- 对外导出
- 调试查看

它们不再是系统内部的主事实来源。

## 8. 典型案例

### 8.1 单旧图复用

输入：

- `dataset_raw/old/field_001.fits`
- `dataset_raw/new/field_001.fits`
- `dataset_raw/new/field_001 (2).fits`
- `dataset_raw/new_marked/field_001.fits`
- `dataset_raw/new_marked/field_001 (2).fits`

结果：

- 生成两个任务
- 两个任务的 `new_asset_id` 不同
- 两个任务的 `new_marked_asset_id` 不同
- 两个任务的 `old_asset_id` 相同

### 8.2 同一天区多次采集

输入：

- `field_001.fits`
- `field_001 (2).fits`

如果两张新图的 `DATE-OBS` 不同，则通常会得到：

- `20260115T203000__field_001`
- `20260115T204500__field_001`

无需额外序号。

### 8.3 完全同名冲突

如果 `date_token + field_name` 仍重复，则生成：

- `20260115T203000__field_001`
- `20260115T203000__field_001__01`

数据库和文件系统都会使用最终 `task_id` 区分这两个任务。

## 9. 与旧方案的区别

旧方案的主要问题是：

- 依赖文件名和目录扫描推断任务
- 同名样本容易被覆盖
- 旧图需要通过复制才能“伪装成一对一”
- 当前标注和历史版本分散在多套 JSON/SQLite 中

当前方案的主要变化是：

- 任务以数据库为中心，而不是以文件名为中心
- 旧图在表中复用，不在文件系统里复制
- 预处理、查看、标注、锁、训练使用同一份任务主表
- JSON 降级为兼容输出，而不是主存储

## 10. 当前实现入口

与本文档直接相关的实现文件：

- `src/scann/core/dataset_storage.py`
- `src/scann/services/dataset_preprocess_service.py`
- `src/scann/core/fits_annotation_storage.py`
- `src/scann/native_annotation/annotation_service.py`
- `src/scann/native_annotation/task_lock_service.py`
- `src/scann/native_annotation/dataset_service.py`
- `src/scann/gui/controllers/pair_controller.py`
- `src/scann/ai/dataset.py`
- `src/scann/ai/training_worker.py`

如果这套流程或表结构发生变化，应优先更新本文档，再更新依赖本文档的概要说明。
