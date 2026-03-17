# v2 标注存储升级方案（SQLite）

## 背景

当数据规模达到：
- 数千张 FITS 图
- 单图数万框标注

使用单个 `annotations.json` 会出现：
1. 每次保存都需要全量重写，I/O 和序列化开销高
2. 文件体积持续膨胀，解析和加载时间上升
3. 崩溃恢复粒度粗，容易出现半写入风险

## 目标

- 将标注持久化从「单文件全量写」切换为「数据库增量写」
- 保持与历史 JSON 数据集兼容
- 保持训练流程可读取统一标注文档结构

## 存储结构

默认文件：
- `annotations.db`：主存储（SQLite）
- `annotations.json`：轻量 manifest（声明存储后端与 db 文件名）

### 表设计

`images`
- `id TEXT PRIMARY KEY`
- `file_name TEXT NOT NULL`
- `label TEXT`
- `detail_type TEXT`
- `ai_suggestion TEXT`
- `ai_confidence REAL`
- `updated_at TEXT NOT NULL`

`bboxes`
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `image_id TEXT NOT NULL`（外键 -> `images.id`）
- `box_index INTEGER NOT NULL`
- `x INTEGER NOT NULL`
- `y INTEGER NOT NULL`
- `width INTEGER NOT NULL`
- `height INTEGER NOT NULL`
- `label TEXT`
- `detail_type TEXT`
- `confidence REAL`

索引：
- `idx_bboxes_image_id` on `bboxes(image_id)`

SQLite pragma：
- `journal_mode=WAL`
- `synchronous=NORMAL`
- `foreign_keys=ON`

## 读写策略

1. **加载优先级**
   - manifest 指向 SQLite -> 读 `annotations.db`
   - 若存在 `annotations.db` -> 直接读取
   - 否则回退读取 legacy `annotations.json(images)`

2. **写入策略**
   - 按样本增量写（`upsert_sample`）
   - 仅替换该样本的 bbox，避免全量重写

3. **迁移策略**
   - 首次加载 legacy JSON 时自动迁移到 SQLite（一次性）

## 兼容性

- 训练路径已支持通过统一读取层自动兼容：
  - SQLite 主存储
  - legacy JSON
- 导出功能仍可输出标准 `annotations.json`（用于离线交换）

## 性能收益（预期）

相对全量 JSON 重写：
- 保存复杂度从约 $O(N)$（全量样本）降低为 $O(k)$（当前样本框数）
- 大数据集下交互保存延迟显著降低
- I/O 峰值和文件锁冲突概率下降
