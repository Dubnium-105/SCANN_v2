# Label Studio Phase B 配置模板（原生 FITS 标注）

> 版本：v1.0  
> 日期：2026-03-19

## 1. 目标

为 Phase B 提供可直接落地的 Label Studio 标注模板，满足：

1. 使用 `js9_iframe` 进行 FITS 主判读；
2. 使用 `js9_regions_json` 作为可提交结果字段；
3. 保留 `RectangleLabels` 作为降级方案。

---

## 2. 推荐 Label Config（XML）

请在项目 Label Config 中使用 `bridge/app.py` 的函数：`get_label_studio_phaseb_label_config()`。

该模板已包含：

- `<HyperText name="js9_iframe" value="$js9_iframe"/>`
- `<Image name="preview_png" value="$preview_png"/>`
- `<RectangleLabels ...>`（降级画板）
- `<TextArea name="js9_regions_json" ... value="$js9_regions_json"/>`

---

## 3. 提交链路约定

1. 宿主页面在提交前调用 viewer 的 `collectRegions()`；
2. 将返回值 `JSON.stringify(regions)` 写入 `js9_regions_json` 对应输入控件；
3. 提交任务；
4. bridge webhook 优先解析 annotation result 中的 `js9_regions_json`；
5. 若不存在该字段，回退到 task data，再回退到 `rectanglelabels`。

---

## 4. 兼容策略

- 当 `js9_regions_json` 字段不存在：启用旧 `rectanglelabels` 解析。
- 当 `js9_regions_json` 明确为 `[]`：按“用户清空 region”处理，不再回退旧框结果。
