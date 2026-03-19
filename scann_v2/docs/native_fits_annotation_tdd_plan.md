# SCANN 原生FITS在线标注协作平台 - TDD 逐提交实施计划

本文档基于 `native_fits_annotation_platform_design.md` 设计方案，采用**测试驱动开发 (TDD)** 方法，将整个系统的开发拆解为从零到一的逐提交（Commit-by-Commit）开发指南。

每次提交都应遵循 **红 (Red) -> 绿 (Green) -> 重构 (Refactor)** 的 TDD 节奏先编写测试用例（此时测试失败），再实现能让测试通过的最小代码，最后进行代码重构优化。

---

## 阶段一：后端 MVP 基础架构与 API (FastAPI)

### Commit 1: 初始化后端与测试环境
* **状态:** `done`（2026-03-19）
* **Test (Red):** 
  * 编写 `tests/test_health.py`，断言 `GET /api/health` 返回 `{"status": "ok"}`。
* **Impl (Green):** 
  * 初始化 FastAPI 应用。
  * 编写 `/api/health` 路由。
* **Refactor:** 
  * 提取路由配置，配置 CORS 中间件供前端后续使用。
* **验证结果:**
  * `pytest tests/test_health.py -q` ✅ 1 passed

### Commit 2: 数据集目录遍历 API
* **状态:** `done`（2026-03-19）
* **Test (Red):** 
  * 编写 `tests/test_dataset.py`。
  * Mock 测试目录（包含 `old/`, `new/`, `new_marked/` 以及符合规则的 `.fts` 文件）。
  * 断言 `GET /api/tasks` 能够返回任务列表聚合结果，每个任务包含同一目标在三个目录下的相对路径。
* **Impl (Green):** 
  * 引入 `pathlib` 扫描本地文件。
  * 根据同名文件聚合生成 Task Session 数据结构。
* **Refactor:** 
  * 将目录扫描逻辑提取为独立的 `DatasetService`，配置化存储基准路径。
* **验证结果:**
  * `pytest tests/test_dataset.py tests/test_health.py -q` ✅ 2 passed

### Commit 3: 临时渲染方案 - FITS 转 PNG (MVP用)
* **状态:** `done`（2026-03-19）
* **Test (Red):** 
  * 提供一个最小的 FITS 测试文件。
  * 编写 `tests/test_fits_render.py`，断言 `GET /api/render/{filepath}` 成功返回 `image/png` 格式数据。
* **Impl (Green):** 
  * 引入 `astropy` 读取 FITS 文件数据流。
  * 使用基本默认算法（如 ZScale）拉伸数据并截断至 8-bit。
  * 利用 `PIL` 或 `matplotlib` 生成基础 PNG 并在内存中返回。
* **Refactor:** 
  * 将 FITS 提取及转换引擎抽取为独立的 `FITSEngine` 工具类，缓存结果。
* **验证结果:**
  * `pytest tests/test_fits_render.py tests/test_dataset.py tests/test_health.py -q` ✅ 3 passed

### Commit 4: 标注数据保存 API
* **状态:** `done`（2026-03-19）
* **Test (Red):** 
  * 编写 `tests/test_annotation.py`。
  * 发送一串模拟的标注 JSON 数据到 `POST /api/annotations/{task_id}`。
  * 断言接口返回成功，且磁盘上对应的 `positive/` 或 `negative/` 文件夹内生成了相应的 JSON 或 SQLite 记录。
* **Impl (Green):** 
  * 实现保存路由。
  * 将传入数据写入到对应的归档文件夹中。
* **Refactor:** 
  * 合并数据校验逻辑，使用 Pydantic Model 校验前台传送坐标及属性格式的合法性。
* **验证结果:**
  * `pytest tests/test_annotation.py tests/test_fits_render.py tests/test_dataset.py tests/test_health.py -q` ✅ 4 passed

---

## 阶段二：前端 MVP 构建与核心交互 (Vue 3 + Vite)

### Commit 5: 初始化前端项目与 Vitest
* **Test (Red):** 
  * 编写 `src/components/__tests__/App.spec.js`，断言页面能正确渲染标题（例如：包含 "SCANN Native Annotation"）。
* **Impl (Green):** 
  * 使用 Vite 初始化 Vue 3 模板，集成 Vitest。
  * 引入 TailwindCSS，构建基础暗黑 (Dark) 框架布局（Header, Canvas, Sidebar）。
* **Refactor:** 
  * 组件化拆分（拆出 Header、Inspector、Canvas 预留位）。

### Commit 6: Konva 画布与多图加载机制
* **Test (Red):** 
  * 编写画布测试：模拟 API 返回的由 `old`, `new`, `new_marked` 组成的任务对象。
  * 断言组件 Store 内部成功创建 3 个图像状态节点，并且画布上默认呈现 `new` 图层。
* **Impl (Green):** 
  * 引入 `Konva.js` 或 `vue-konva`。
  * 请求后端 `GET /api/tasks` 后并列请求 PNG 渲染接口以预加载图片进入内存。
  * 将图像置入 Konva 的底层。
* **Refactor:** 
  * 将图像加载逻辑封装为 `useImageLoader` Composables，管理加载进度和状态。

### Commit 7: 毫秒级闪视 (Blinking) 与画布控制
* **Test (Red):** 
  * 触发键盘 `Space` 或 `Tab` 按键事件。
  * 断言当前活跃视图状态 (Current View) 按序从 `new` -> `new_marked` -> `old` 轮转。
* **Impl (Green):** 
  * 绑定全局 `keydown` 监听器。
  * 根据当前 View 状态，控制 Konva 中特定目标图像图层的 `visible` 属性。
* **Refactor:** 
  * 封装 `useBlinkControl`。添加基础平移 (Pan) 和鼠标滚轮缩放 (Zoom) 支持，确保三图在缩放平移时位置绝对绑定。

### Commit 8: 画布标注工具 (BBox) 与提交流程
* **Test (Red):** 
  * 模拟鼠标在画布按下(mousedown)、移动(mousemove)、抬起(mouseup)。
  * 断言生成了一个 BBox 标注对象（包含相对图片的正确 x, y, width, height）。
* **Impl (Green):** 
  * 在 Konva 中添加独占的独立 Annotation Layer。
  * 监听鼠标事件绘制矩形框（Rect）。
  * 实现『提交』按钮，将当前 Layer 的标注信息 POST 到后端。
* **Refactor:** 
  * 解耦绘制工具状态，支持快速在“移动模式”和“框选标注模式”间切换。

---

## 阶段三：前端原生 FITS 高阶功能 (阶段二的进化)

### Commit 9: 原生 FITS 前端解析与数据池
* **Test (Red):** 
  * 前端单元测试模拟加载原生 `.fts` 二进制流（不再加载 PNG）。
  * 断言能正确解析出 Headers 字典以及 Float32 像素数据数组。
* **Impl (Green):** 
  * 引入 `fits.js`。
  * 请求 FITS 文件并解析其 Data Unit 回传给组件。
* **Refactor:** 
  * 使用 Web Worker 移出 FITS 文件解压或解析逻辑，防止解析期间导致主线程卡顿（UI假死）。

### Commit 10: 实时直方图拉伸与反色
* **Test (Red):** 
  * 触发右侧拉伸滑块改变 `min`、`max` 状态。
  * 触发反色 Toggle 开关。
  * 断言最终注入 Canvas 的 ImageData 数组被实时正确计算（边界截断、255映射及 255-x 反色）。
* **Impl (Green):** 
  * 在视图层添加直方图及双滑块 UI 控制组件。
  * 实现基于 JS 循环的 Float32 到 Uint8ClampedArray 的渲染转换器，重新绘制至画布。
* **Refactor:** 
  * 若 JS 性能出现瓶颈（如 2048x2048 超大图），切换至 WebGL Fragment Shader 进行显卡渲染优化。

---

## 阶段四：进阶工具与基础防冲突

### Commit 11: 协作防冲突与锁定机制
* **Test (Red):** 
  * 后端：模拟用户 A 请求获取下一组图像。断言返回的图像被加上 'Lock'（占用锁定标记）。
  * 后端：模拟用户 B 获取图像队伍，断言其无法拿到用户 A 已锁定的图像。
* **Impl (Green):** 
  * 引入基础内存哈希表或 Redis 的 Session 锁定机制，根据 Client ID 占位。
  * 标注完成后释放该任务的 Lock。
* **Refactor:** 
  * 加入超时释放机制（Client 掉线长时间未提交，自动释放图像让别人标注）。

### Commit 12: 进阶标注工具 (点与多边形)与属性切换
* **Test (Red):** 
  * 前端模拟切换工具到 Point/Polygon，断言在画布渲染相应的图形。
  * 点击列表中的标注，赋予 "True Positive" / "Artifact" 的标签修改。
* **Impl (Green):** 
  * Konva 数据层扩充，支持多边形连线逻辑和单个居中点映射逻辑。
  * 右侧 Inspector 提供对选中 Annotation 的下拉选单更改。
* **Refactor:** 
  * 将通用标注事件进行基类化，减少画框、画点与画多边形之间的冗余代码。

---

## 阶段五：用户管理与标注版本控制

### Commit 13: 基础用户与鉴权系统 (JWT)
* **Test (Red):** 
  * 后端: 编写 `tests/test_auth.py`，未带 Token 访问受保护接口应返回 `401 Unauthorized`。登录测试需返回有效 JWT。
  * 前端: 模拟登录表单，断言获取 Token 并无缝存入 Vuex/Pinia Store，应用头栏能正确显示解析到的用户名。
* **Impl (Green):** 
  * 后端集成 `PyJWT` 与身份加密库（如 `passlib`），实现 `/api/login` 获取 Token 的鉴权路由。
  * 使用 FastAPI 的 `Depends` 依赖注入将所有读写接口用 OAuth2 密码流保护起来。
  * 前端新增 Login 视图，通过 Router 路由守卫自动拦截未登录访问跳回登录页。
* **Refactor:** 
  * 提取并模块化用户的权限校验逻辑（如判断角色是 `admin` 还是 `annotator`），简化后端鉴权样板代码。

### Commit 14: 标注版本控制与历史回溯
* **Test (Red):** 
  * 编写 `tests/test_versions.py`。针对同一 `task_id`，模拟不同用户 (User A 和 User B) 或同用户的多次调用标注保存 API。
  * 断言：`GET /api/annotations/{task_id}/history` 能返回包含正确时间线、UserID 的版本列表，且请求特定历史版本 `Revision ID` 能精确调出保存时的标注坐标。
* **Impl (Green):** 
  * 扩充后端数据库模型：将原生单一的 `Annotation` 拆分为携带历史快照架构的 `AnnotationRevision` 表 (包含自增/UUID版本号、提交人 ID、时间戳、序列化后的 JSON 结构内容)。
  * 前端右侧控制台拉取对应图像的合并历史记录，并在 UI 上渲染带有时间戳的层级版本列表（Timeline）。
* **Refactor:** 
  * 在数据库写入后挂载一个异步触发器 (Background Task)，将最后一次写入的（Top Revision）标注结果自动清洗并静默同步至物理系统 `positive/` 或 `negative/` JSON，剥离算法运行和持久化查询。