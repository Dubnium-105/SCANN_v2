# 数据完整性、迁移与固定分区运行手册

本手册对应实施计划的 `WP-01`～`WP-03`。所有命令默认先在数据库副本执行；没有完成异盘在线备份、完整性检查和计数核对前，不得对生产数据库执行迁移。

## 1. 数据库迁移

```bash
scann-dataset-migrate /path/to/dataset \
  --db-path /path/to/copied-scann_dataset.db
```

迁移命令会：

- 创建基础 schema 中缺失的表/列；
- 按编号执行 `schema_migrations` 中尚未应用的迁移；
- 校验已应用迁移的名称和 checksum；
- 输出已应用迁移及 `PRAGMA integrity_check`。

迁移具有以下约束：

- 同一迁移只能使用一个固定 ID；
- 已发布迁移不得修改名称或 checksum source；
- 单个迁移在 SQLite savepoint 内执行；
- 失败迁移会回滚，且不会写入 `schema_migrations`；
- 数据库包含当前代码不认识的更高版本迁移时拒绝启动。

## 2. 只读数据审计

```bash
scann-dataset-audit /path/to/dataset \
  --db-path /path/to/copied-scann_dataset.db \
  --output /safe/report/audit.json
```

默认检查：

- 必需表和各表计数；
- SQLite `integrity_check`、`foreign_key_check`；
- 原始资产、任务产物、训练快照和分区 manifest；
- 当前框和历史 revision 的标签合法性；
- `tasks.current_annotation_count` 与实际框数；
- AI prelabel 缓存框数；
- 模型产物存在性及已记录的 SHA256。

退出码：

- `0`：没有 error；普通 warning 不阻塞；
- `1`：使用 `--strict` 且存在 warning；
- `2`：存在 error。

审计不会调用 `DatasetStorage.ensure_schema()`，数据库连接使用 SQLite `mode=ro` 和 `query_only`。

## 3. 生成固定分区草案

```bash
scann-dataset-partition /path/to/dataset \
  --db-path /path/to/copied-scann_dataset.db \
  --partition-name gold-v1 \
  --seed 42 \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --output /safe/report/gold-v1.json
```

分组优先级：

1. `night_key + field_key`
2. `capture_key`
3. `field_key`
4. `task_id`

同一 group 不会跨 train、validation 和 test。只有一个独立 group 的类别会优先留在 train，并在 manifest 的 `limited_group_classes` 中明确报告。

分区 manifest 包含：

- taxonomy 版本；
- 算法版本和 seed；
- 三个 split 的任务、group 和细类支持；
- group overlap 审计；
- canonical SHA256。

相同输入、参数和名称必须生成相同 `partition_id` 与 checksum。

## 4. API 注册与激活

管理员可通过：

```text
POST /api/dataset-partitions
GET  /api/dataset-partitions
```

创建并查看不可变分区。`activate=false` 时只登记草案；`activate=true` 时将其设为唯一活动分区。

活动分区生效后：

- 默认训练快照只包含 train + validation；
- 显式选择 gold-test task 创建训练快照会被拒绝；
- 快照升级为 `version=3.0`；
- 快照保存 taxonomy、partition ID、manifest checksum 和 split 信息；
- 原始标注框不会被回填或修改。

## 5. 生产执行清单

1. 记录当前 commit、容器 image ID、服务健康状态和数据库路径。
2. 使用 SQLite online backup 备份到 `/mnt/disk4`。
3. 对备份执行 integrity、SHA256 和核心表计数。
4. 再复制一个验证副本。
5. 对验证副本执行 migrate、第二次 migrate、audit 和 partition planning。
6. 对比迁移前后业务表计数；只允许新增迁移/分区控制表。
7. 审核 gold test 类别与独立 group 覆盖。
8. 备份通过且回滚路径验证后，才允许生产迁移。
9. 分区先注册为 inactive；确认后再 activate。
10. 激活后创建首个 snapshot 3.0，并再次确认 gold-test task 没有进入快照。
