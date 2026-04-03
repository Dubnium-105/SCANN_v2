# 全尺寸图像输入 + ViT-Huge 低比特 KV 压缩逐提交实现清单

## 1. 文档目标

本文用于把“全尺寸图像输入 + ViT-Huge + 低比特 KV 压缩”方案细化为可直接执行的逐提交实现清单。

约束与原则如下：

- 保留当前实验框架中“配置驱动 + 可插拔模型/量化模块 + 统一 benchmark”的组织方式。
- 默认以推理显存优化为第一目标，训练期显存优化作为后续扩展。
- 第一阶段先做 PyTorch 参考实现，不要求一开始就写自定义 CUDA kernel。
- 必须保留“全模块消融”能力，不能只做局部 patch 后失去整模型对照。
- 必须支持后续比较 `baseline / 局部模块压缩 / 全模块压缩 / 残差增强`。

---

## 2. 当前代码现状

当前仓库相关现状：

- `ViT` 模型入口目前仅显式支持 `vit_b_16`。
- 数据集预处理接口目前默认使用单个 `image_size: int`，更偏向正方形输入。
- 当前 `TurboQuant-style` 实现仅覆盖 `Swin` 的 `ShiftedWindowAttention`。
- 当前 `TurboQuant-style` 逻辑是“运行时压缩 + 立即重建”，不是“低比特驻留 + 分块解码”。

因此，本次改造不能只在现有 `Swin` 路径上做小修，而是需要新增一条面向 `ViT` 的 attention compression 路线。

---

## 3. 总体拆分策略

推荐按 10 个提交推进，每个提交都应保持可运行。

提交顺序建议：

1. 扩展配置与输入尺寸表达
2. 扩展模型工厂，加入 `ViT-Huge`
3. 引入 ViT attention patch 骨架
4. 实现 `PackedKV` 数据结构与 pack/unpack
5. 实现分块解码版 attention 参考实现
6. 将压缩 attention 接入 ViT encoder block
7. 接入实验 runner、指标与配置
8. 补齐“全模块消融”开关与实验矩阵
9. 增加 `QJL residual` 可选分支
10. 文档、测试、结果模板与回归校验

---

## 4. 逐提交实现清单

### Commit 1: 扩展全尺寸输入配置表达

建议提交信息：

```text
feat(experiments): support full-resolution image size config for vit experiments
```

目标：

- 让实验配置可以表达全尺寸输入或非方形输入。
- 不在第一步引入压缩逻辑，只先把输入接口改正确。

修改范围：

- `scann_v2/src/scann/experiments/legacy_dataset.py`
- `scann_v2/src/scann/experiments/legacy_runner.py`
- 相关配置读取与校验逻辑

具体改动：

- 将 `image_size` 从单个 `int` 扩展为：
  - `int`
  - `[height, width]`
  - 特殊模式 `keep`
- 将 resize 逻辑改为支持：
  - `resize -> [H, W]`
  - `pad_resize -> [H, W]`
  - `keep -> 保持原始尺寸`
- 在实验配置 dataclass / schema 中新增统一解析函数，例如：
  - `resolve_image_size_spec(...)`
- 保留旧配置兼容，不破坏当前 `224` 单值写法。

验收标准：

- 旧实验配置不受影响。
- 新配置可以传 `[1024, 1024]` 或 `[H, W]`。
- 数据集加载在 `keep` / 非方形 resize 下不报错。

测试：

- 新增 `tests/test_legacy_dataset_image_size_spec.py`

---

### Commit 2: 扩展模型工厂，加入 ViT-Huge 与全尺寸位置编码插值

建议提交信息：

```text
feat(models): add vit_huge experiment entry with positional embedding resize
```

目标：

- 在实验工厂内支持 `vit_h_14` 或团队选定的 `ViT-Huge` 入口。
- 支持全尺寸输入下的位置编码插值。

修改范围：

- `scann_v2/src/scann/experiments/legacy_runner.py`
- 如有必要，新增 `scann_v2/src/scann/experiments/vit_factory.py`

具体改动：

- 在 `create_experiment_model(...)` 中加入 `vit_h_14` 分支。
- 把 `ViT` 模型创建逻辑从当前大函数中抽离为独立 helper。
- 实现位置编码插值的统一入口：
  - 正方形输入优先使用现有插值思路
  - 非方形输入新增二维插值逻辑
- 保留 `vit_b_16` 旧入口不变。

验收标准：

- `vit_b_16` 与 `vit_h_14` 均可创建。
- 输入分辨率改变后位置编码能正常加载。
- `pretrained=True/False` 都可跑通 smoke forward。

测试：

- 新增 `tests/test_vit_factory.py`

---

### Commit 3: 引入 ViT attention patch 骨架

建议提交信息：

```text
feat(attention): add vit attention patch scaffold for compression experiments
```

目标：

- 新增专门面向 `ViT` 的 attention patch 入口。
- 先完成模块定位与替换，不急着接入低比特压缩。

修改范围：

- 新增 `scann_v2/src/scann/experiments/vit_attention_compression.py`
- `scann_v2/src/scann/experiments/legacy_runner.py`

具体改动：

- 新增：
  - `create_vit_attention_compression_model(...)`
  - `iter_vit_attention_modules(...)`
  - `patch_vit_attention_modules(...)`
- 约定压缩 patch 的输入输出接口。
- 记录每个 attention block 的：
  - block index
  - hidden dim
  - num heads
  - head dim
- 增加 `layer_selector` 机制，支持后续消融：
  - `all`
  - `first_n`
  - `last_n`
  - `middle`
  - `explicit_indices`

验收标准：

- 可以枚举 ViT 所有 self-attention 模块。
- patch 后模型前向输出 shape 不变。
- 关闭压缩时数值与原模型一致或接近完全一致。

测试：

- 新增 `tests/test_vit_attention_patch.py`

---

### Commit 4: 实现 PackedKV 数据结构与 4-bit pack/unpack

建议提交信息：

```text
feat(quant): add packed 4-bit kv representation for vit attention
```

目标：

- 先做真正会省显存的低比特存储层。
- 第一版只做 `4-bit`，暂不做 `3-bit`。

修改范围：

- 新增 `scann_v2/src/scann/experiments/packed_kv.py`
- `scann_v2/src/scann/experiments/vit_attention_compression.py`

具体改动：

- 定义 `PackedKV4Bit` dataclass，字段建议包括：
  - `codes`
  - `scales`
  - `zero_points` 或 `centroids`
  - `original_shape`
  - `group_size`
  - `token_range`
  - `head_dim`
- 实现：
  - `pack_tensor_4bit(...)`
  - `unpack_tensor_4bit_block(...)`
  - `pack_kv_per_head(...)`
- 默认粒度先做：
  - per-head
  - per-token-group 或 per-channel-group
- `codes` 使用 `uint8` 打包，2 个 4-bit code 共用 1 byte。

验收标准：

- pack/unpack 后相对误差可控。
- `codes` 的真实 dtype 为 `uint8`，不是 `int64` bucket index。
- 与当前“立即重建”的旧路径相比，显存占用路径发生实质变化。

测试：

- 新增 `tests/test_packed_kv.py`

---

### Commit 5: 实现分块解码版 attention 参考实现

建议提交信息：

```text
feat(attention): add streaming attention with blockwise kv dequantization
```

目标：

- 不再整块恢复 `K/V`。
- 不再强制落地完整 attention matrix。

修改范围：

- `scann_v2/src/scann/experiments/vit_attention_compression.py`

具体改动：

- 实现：
  - `streaming_packed_attention(...)`
  - `online_softmax_update(...)`
  - `decode_kv_block(...)`
- 计算流程改为：
  - `q` 保持 `bf16/fp16`
  - `packed_k / packed_v` 常驻低比特
  - 每次解码一个 token block
  - 逐 block 累积输出
- 默认参数：
  - `token_block_size=64`
  - 支持 `32 / 64 / 128`

验收标准：

- 与 dense attention baseline 相比，输出误差处于可接受范围。
- attention 子模块峰值显存明显下降。
- 可以在小模型上先跑通数值对照。

测试：

- 新增 `tests/test_streaming_attention.py`

---

### Commit 6: 将压缩 attention 接入 ViT encoder block

建议提交信息：

```text
feat(vit): integrate packed kv streaming attention into vit encoder blocks
```

目标：

- 把 pack + streaming attention 真的接进 ViT block。
- 这一提交完成后，应能跑端到端推理 benchmark。

修改范围：

- `scann_v2/src/scann/experiments/vit_attention_compression.py`
- `scann_v2/src/scann/experiments/legacy_runner.py`

具体改动：

- 完成 patched forward：
  - 单独计算 `q`
  - 单独计算 `k`
  - 立刻 pack `k`
  - 单独计算 `v`
  - 立刻 pack `v`
  - 进入 streaming attention
- 默认保留以下开关：
  - `quantize_k=True/False`
  - `quantize_v=True/False`
  - `preserve_cls_token=True/False`
  - `enabled_layer_indices`
- 先做 reference path，不强求极致速度。

验收标准：

- `ViT-B` 上可跑通。
- `ViT-H` 上在 batch=1 下可跑通。
- 压缩模式与 baseline 模式都可被统一 benchmark。

测试：

- 新增 `tests/test_vit_compressed_forward.py`

---

### Commit 7: 接入实验 runner、配置与指标

建议提交信息：

```text
feat(experiments): add vit kv compression configs benchmarks and metrics
```

目标：

- 让新路线进入统一实验框架。
- 不再依赖手工 patch 测试。

修改范围：

- `scann_v2/src/scann/experiments/legacy_runner.py`
- `scann_v2/scripts/experiments/`
- `scann_v2/experiments/configs/`

具体改动：

- 新增实验配置字段：
  - `attention_compression_mode`
  - `kv_bits`
  - `group_size`
  - `token_block_size`
  - `preserve_cls_token`
  - `enabled_layer_indices`
  - `quantize_k`
  - `quantize_v`
- 新增 benchmark 指标：
  - `peak_gpu_memory_attention_only_mb`
  - `packed_kv_size_mb`
  - `token_count`
  - `num_patched_layers`
- 新增配置文件：
  - `legacy_vit_b16_fullres_baseline.json`
  - `legacy_vit_b16_fullres_k4_stream.json`
  - `legacy_vit_h14_fullres_baseline.json`
  - `legacy_vit_h14_fullres_k4_stream.json`

验收标准：

- 一条命令可运行 baseline 与压缩版对比。
- 输出 csv / summary 中能看到新增指标。

测试：

- 新增 `tests/test_vit_compression_config.py`

---

### Commit 8: 补齐全模块消融矩阵

建议提交信息：

```text
feat(ablation): add full-module ablation matrix for vit kv compression
```

目标：

- 明确保留“全模块消融”。
- 避免后续只能比较“某个局部 patch”而无法比较整模型压缩收益。

修改范围：

- `scann_v2/src/scann/experiments/legacy_runner.py`
- `scann_v2/experiments/configs/`
- 新增消融配置模板

必须保留的消融维度：

- 层范围：
  - `all_layers`
  - `first_25pct`
  - `middle_50pct`
  - `last_25pct`
  - `custom_indices`
- 压缩对象：
  - `K only`
  - `V only`
  - `K/V both`
- token 范围：
  - `patch_only`
  - `patch_plus_cls`
  - `cls_preserved`
- 精度设置：
  - `bf16 baseline`
  - `K 4-bit`
  - `KV 4-bit`
- 结构开关：
  - `streaming on/off`
  - `full attention matrix on/off`

必须有的“全模块”配置：

- `all_layers + K only`
- `all_layers + K/V both`
- `all_layers + K/V both + cls preserved`

输出要求：

- 所有消融结果必须进入统一汇总表。
- 汇总表里必须显式包含：
  - `layer_scope`
  - `kv_target`
  - `cls_policy`
  - `streaming_enabled`

验收标准：

- 可以完整跑“局部模块”和“全模块”两类对照。
- 全模块配置不会被代码路径特殊排除。

测试：

- 新增 `tests/test_vit_ablation_matrix.py`

---

### Commit 9: 增加可选的 QJL residual 分支

建议提交信息：

```text
feat(residual): add optional qjl residual correction for packed vit kv
```

目标：

- 在不破坏第一版省显存路径的前提下，加入精度恢复选项。
- 注意这里必须做成可选，不得覆盖基础 4-bit 路线。

修改范围：

- `scann_v2/src/scann/experiments/packed_kv.py`
- `scann_v2/src/scann/experiments/vit_attention_compression.py`
- 配置与 runner

具体改动：

- 增加：
  - `residual_mode = none | qjl_sign_norm`
  - `qjl_dim`
- 残差表示建议先做：
  - `sign bits`
  - `norm scalar`
- 仍然坚持 blockwise decode，不恢复整块 residual tensor。

验收标准：

- `residual_mode=none` 与之前结果一致。
- `residual_mode=qjl_sign_norm` 可单独 benchmark。
- 显存上升可控，精度有恢复空间。

测试：

- 新增 `tests/test_qjl_residual_mode.py`

---

### Commit 10: 文档、结果模板与回归校验

建议提交信息：

```text
docs(experiments): document vit huge fullres kv compression workflow and result schema
```

目标：

- 补齐文档、结果模板和回归检查脚本。
- 让后续正式实验可重复执行。

修改范围：

- `scann_v2/docs/`
- `scann_v2/experiments/results/` 模板
- `scann_v2/scripts/experiments/`

具体改动：

- 新增结果模板字段：
  - `token_count`
  - `packed_kv_size_mb`
  - `layer_scope`
  - `kv_target`
  - `cls_policy`
  - `streaming_enabled`
  - `residual_mode`
- 新增文档：
  - 运行步骤
  - 配置解释
  - 消融矩阵说明
  - 推荐先跑顺序
- 加一个轻量回归脚本，至少能检查：
  - baseline 是否可跑
  - 全模块路径是否可跑
  - csv 字段是否齐全

验收标准：

- 新人能按文档复现实验。
- 配置、脚本、结果表头三者一致。

---

## 5. 全模块消融保留要求

这一部分是硬性约束，不应在实现过程中被弱化。

必须满足：

- 任意压缩策略都能切换为 `all_layers`。
- `all_layers` 必须经过真实 benchmark，而不是只在配置层面保留。
- 不允许只保留“中间层压缩”或“单层 demo”而删除全模块模式。
- 结果表中必须能单独筛出“全模块压缩”全部实验。
- 回归测试必须覆盖至少 1 个 `all_layers` 配置。

推荐默认消融矩阵：

- `baseline_dense`
- `all_layers_k_only_4bit`
- `all_layers_kv_4bit`
- `all_layers_kv_4bit_cls_preserved`
- `middle_50pct_kv_4bit`
- `last_25pct_kv_4bit`
- `all_layers_kv_4bit_qjl`

---

## 6. 推荐先跑顺序

为了降低实现风险，建议按下面顺序实际验证：

1. `ViT-B + 224` 上先跑通 patch scaffold
2. `ViT-B + full-resolution` 上先跑通 `K only + 4-bit + streaming`
3. `ViT-B + full-resolution` 上补齐 `KV both`
4. `ViT-B + full-resolution` 上补齐 `all_layers` 消融
5. `ViT-H + full-resolution` 上先跑 baseline
6. `ViT-H + full-resolution` 上跑 `all_layers_k_only_4bit`
7. `ViT-H + full-resolution` 上跑 `all_layers_kv_4bit`
8. 最后再接入 `QJL residual`

---

## 7. 第一阶段不做的内容

以下内容建议明确延后，避免第一轮实现过重：

- 自定义 CUDA kernel
- 3-bit 真正 bit-pack
- 训练态反向传播压缩
- 和 FSDP / tensor parallel 同轮集成
- 权重量化与 KV 压缩同时深度耦合

---

## 8. 完成标准

当满足以下条件时，可认为第一阶段改造完成：

- `ViT-B` 与 `ViT-H` 都能跑 baseline 与压缩版
- 支持全尺寸输入
- 支持 `all_layers` 全模块压缩
- `PackedKV4Bit` 真实使用 `uint8` 打包
- attention 采用分块解码，而非整块重建
- benchmark 表中能比较 baseline / 局部压缩 / 全模块压缩 / QJL 可选增强

---

## 9. 建议文档后续联动

本文件完成后，建议同步更新以下文档：

- `scann_v2/docs/experiment_plan.md`
- `scann_v2/docs/旧数据集验证计划文档_Transformer训练验证与环境搭建.md`

更新内容：

- 增加 `ViT-Huge + full-resolution + packed KV` 路线说明
- 增加“全模块消融保留”说明
- 增加结果表字段说明
