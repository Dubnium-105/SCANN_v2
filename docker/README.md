# Docker 部署与 CI/CD

`docker/` 目录现在只负责部署、打包和容器构建，不再复制一份应用源码。

镜像直接从仓库中的当前代码构建：

- 后端源码：`scann_v2/src/scann/native_annotation`
- 前端源码：`scann_v2/frontend`

## 目录内容

- `docker-compose.yml`：生产部署编排
- `deploy.sh`：在目标 Linux 机器上部署
- `deploy_from_runner.sh`：供 GitHub Actions 自托管 runner 调用
- `package_bundle.ps1` / `package_bundle.sh`：打包部署包
- `backend/Dockerfile`：后端镜像
- `frontend/Dockerfile`：前端镜像
- `.env.example`：环境变量模板

## 推荐部署结构

```text
/srv/scann/
|-- app/        # 仓库检出目录或部署包解压目录
`-- dataset/    # FITS 数据与运行时标注数据
```

## 关键环境变量

至少需要配置：

```dotenv
FRONTEND_PORT=8080
BACKEND_BIND_ADDRESS=127.0.0.1
BACKEND_PORT=8000
SCANN_DATASET_DIR=/srv/scann/dataset
SCANN_NATIVE_JWT_SECRET=replace-this-with-a-long-random-secret
SCANN_NATIVE_JWT_EXPIRE_MINUTES=120
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
```

其中数据目录通常至少包含：

- `${SCANN_DATASET_DIR}/new`
- `${SCANN_DATASET_DIR}/old`
- `${SCANN_DATASET_DIR}/new_marked`

## 首次部署

```bash
cd /srv/scann/app
git clone <your-repo-url> .
cd docker
cp .env.example .env
vim .env
chmod +x deploy.sh
./deploy.sh
```

## 日常更新

```bash
cd /srv/scann/app
git pull
cd docker
./deploy.sh
```

## 健康检查

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8080/health
```

## CI/CD 流程

仓库当前的流水线定义在 `.github/workflows/pipeline.yml`。

在 PR 和推送到 `main` 时，GitHub Actions 会执行：

1. 原生标注后端测试
2. 前端测试和打包
3. Docker Compose 校验
4. 前后端镜像构建检查

当变更推送到 `main` 且前面全部通过后，还会：

5. 在带有 `self-hosted`、`linux`、`scann-prod` 标签的 runner 上执行部署
6. 调用 `docker/deploy_from_runner.sh`
7. 用健康检查判定部署是否成功

## 生产 runner 要求

目标 Linux 机器至少需要：

- Docker Engine
- Docker Compose plugin
- `rsync`
- `curl`

建议把 GitHub 仓库变量或环境变量 `DEPLOY_PATH` 设为部署根目录，例如：

```text
DEPLOY_PATH=/srv/scann/app
```

## 打包部署包

如果不通过 `git pull` 更新，也可以先在本地打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\package_bundle.ps1
```

或：

```bash
bash ./docker/package_bundle.sh
```

## 维护说明

原先分散在 `README.md`、`DEPLOYMENT.md` 和 `CI_CD.md` 的内容已经合并到当前文档，
避免三份文件重复描述同一套部署流程。
