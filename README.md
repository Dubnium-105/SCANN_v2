# SCANN

SCANN 是一个面向天文图像分析的仓库，目前同时包含两条代码线：

- `SCANN.py`：历史遗留的单文件桌面程序
- `scann_v2/`：当前维护的主工程，包含桌面版、原生 FITS 标注平台、测试、脚本和文档

如果你要继续开发，请默认进入 `scann_v2/`。

## 当前维护范围

`scann_v2/` 目前包含两套主要运行面：

- PyQt5 桌面应用：用于 FITS 图像浏览、配对、对齐、候选体检测、查询和人工复核
- 原生 FITS 标注平台：FastAPI 后端 + Vue 前端，用于在线标注、任务分配、历史回滚和数据集预处理

## 快速开始

### 1. 桌面应用

```powershell
cd scann_v2
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m scann.app
```

也可以在安装后直接运行：

```powershell
scann
```

### 2. 运行测试

```powershell
cd scann_v2
pytest
```

### 3. 启动原生 FITS 标注后端

后端依赖单独放在 `docker/backend/requirements.txt` 中。下面示例从仓库根目录启动：

```powershell
pip install -r docker\backend\requirements.txt
$env:PYTHONPATH = (Resolve-Path .\scann_v2\src)
$env:SCANN_NATIVE_DATASET_ROOT = (Resolve-Path .\dataset)
python -m uvicorn scann.native_annotation.app:app --reload
```

### 4. 启动原生 FITS 标注前端

```powershell
cd scann_v2\frontend
npm ci
npm run dev
```

## 仓库结构

```text
.
|-- SCANN.py                # 遗留 v1 程序
|-- dataset/                # 示例或运行时数据目录
|-- docker/                 # Linux 部署、容器和发布脚本
`-- scann_v2/
    |-- src/scann/          # 当前主代码
    |-- frontend/           # 原生 FITS 标注前端
    |-- tests/              # pytest 自动化测试
    |-- scripts/            # 打包、诊断和 legacy 脚本
    `-- docs/               # 当前维护中的项目文档
```

## 文档入口

- `scann_v2/docs/README.md`：项目文档索引
- `scann_v2/docs/architecture.md`：当前代码结构与模块边界
- `scann_v2/docs/native_annotation.md`：原生 FITS 标注平台说明
- `docker/README.md`：Docker 部署与 CI/CD 说明
- `scann_v2/tests/README.md`：测试目录约定
- `scann_v2/scripts/README.md`：脚本目录约定

## 遗留内容说明

- `SCANN.py` 和 `scann_v2/scripts/legacy/` 仍保留，主要用于兼容旧流程或参考
- 历史性的重构计划、提交清单、阶段性 backlog 和设计草稿已从仓库文档中移除，避免继续误导维护者
