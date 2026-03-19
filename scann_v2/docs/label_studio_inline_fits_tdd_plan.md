# Label Studio 任务页内嵌 FITS 对比标注（TDD 实施计划）

> 版本：v0.3（执行中）
> 日期：2026-03-19
> 状态：进行中

---

## 1. 背景与问题

当前目标：在 Label Studio 任务页内完成 FITS 判读闭环，减少 PNG 依赖与外链跳转割裂。

---

## 2. 本次目标范围

1. 在同一任务页面内嵌 FITS Viewer。
2. 支持 new/old/new_marked 切换、闪烁、反色、拉伸控制。
3. `/tasks/pull` 同时输出兼容字段与新字段（`js9_embed_url`）。
4. 保持 webhook 回写链路不破坏。

---

## 3. TDD 迭代与当前进度

### Iteration 1：`js9_embed_url`（已完成）

- [x] 新增 URL 组装函数：`_make_js9_embed_url()`
- [x] 增加参数编码与注入防护（`urlencode` + HTML/JS 安全处理）
- [x] 新增单元测试：
  - `tests/bridge/test_js9_embed_url.py`

### Iteration 2：`js9_iframe` 升级为内嵌 viewer（已完成）

- [x] `js9_iframe` 改为内嵌 iframe + 回退链接
- [x] 支持 `new_marked` 缺失分支
- [x] 新增单元测试：
  - `tests/bridge/test_js9_iframe_html.py`

### Iteration 3：新增 `/viewer/js9` 路由（已完成）

- [x] 新增路由：`GET /viewer/js9`
- [x] 页面内提供最小控件集：
  - 新图/旧图/新图(标注)
  - 闪烁开关 + 速度
  - 反色
  - 拉伸重置/自动拉伸（postMessage 触发，便于后续对接 JS9 API）
- [x] 新增 API 测试：
  - `tests/bridge/test_bridge_viewer_route.py`

### Iteration 4：`/tasks/pull` 负载升级（已完成）

- [x] 新增字段：`js9_embed_url`
- [x] 保留兼容字段：`new_url/old_url/new_marked_url/preview_png/js9_iframe`
- [x] 新增测试：
  - `tests/bridge/test_pull_tasks_embed_payload.py`

### Iteration 5：回归与文档（进行中）

- [x] 本文档已创建并回写执行进度
- [x] 自动化测试通过（`pytest tests/bridge -q`：6 passed）
- [ ] 手工验收（浏览器）
- [ ] 真实样本试跑（>=20）

---

## 4. 本次代码落地清单

- `bridge/app.py`（新增）
  - `TaskRecord` 增加 `js9_embed_url`
  - 新增 `_make_js9_embed_url()`
  - 升级 `_make_js9_iframe()`
  - 新增 `/viewer/js9`
  - 升级 `/tasks/pull` 负载
  - 保留 `/webhook/labelstudio`

- `tests/bridge/conftest.py`（新增）
- `tests/bridge/test_js9_embed_url.py`（新增）
- `tests/bridge/test_js9_iframe_html.py`（新增）
- `tests/bridge/test_bridge_viewer_route.py`（新增）
- `tests/bridge/test_pull_tasks_embed_payload.py`（新增）

---

## 5. 风险与后续

1. 当前 viewer 页面已预留 `postMessage` 控制接口；后续可直接接入真实 JS9 API 完成拉伸/反色的原生调用。
2. 若部署存在 CSP/跨域限制，需统一域名与协议。
3. Phase B（JS9 region 主标注）仍需独立专题实现。

---

## 6. 下一步建议

1. 执行 `tests/bridge` 自动化测试并固化到 CI。
2. 在真实数据集上进行 MANUAL-01~MANUAL-05 验收。
3. 评估 Phase B（region 同步到 LS 结果结构）技术方案。
