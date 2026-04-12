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

如果需要启用 PostgreSQL 标注同步，建议同时配置：

```dotenv
SCANN_ANNOTATION_SYNC_ENABLED=true
SCANN_ANNOTATION_SYNC_DATABASE_URL=postgresql://user:password@host:5432/scann_annotation?sslmode=require
SCANN_ANNOTATION_SYNC_DATASET_ID=observatory-2026-04
SCANN_ANNOTATION_SYNC_SCHEMA=scann_backup
SCANN_ANNOTATION_SYNC_INTERVAL_SECONDS=300
SCANN_ANNOTATION_SYNC_CONNECT_TIMEOUT_SECONDS=30
```

注意：
- `docker/.env` 必须保持 `KEY=value` 格式，不能写成 `KEY = value` 或 `KEY: value`
- `SCANN_ANNOTATION_SYNC_DATABASE_URL` 使用 PostgreSQL/libpq DSN，不是 JDBC URL
- 如果云 PostgreSQL 没有启用 SSL，请把 `sslmode=require` 改成 `sslmode=disable`

## 部署拓扑

当前 Docker 部署包含 3 个服务：

- `backend`
  - 继续处理常规标注 API、任务锁、FITS 读取和历史查询
  - 保持原有 bridge 网络和端口暴露方式不变
- `frontend`
  - 继续对外提供 Web 界面
  - 默认把 `/api/*` 代理到 `backend`
- `sync-backend`
  - 专门处理 `/api/annotation-sync/*`
  - 使用 `host` 网络访问云 PostgreSQL
  - 通过 Unix socket 暴露给 `frontend`，不单独开放宿主机 HTTP 端口

这样做的目的是只让“同步到云 PG”这一条链路绕过某些路由器设备上的 Docker bridge 出网问题，其它功能和原有部署保持一致。

## PostgreSQL 同步注意事项

- 远端表不会预先创建；第一次成功执行同步时会自动创建 `scann_backup.annotation_*` 表
- 首次部署后如果管理员前端菜单中的“手动同步”还没执行成功，直接查询 `annotation_sync_state` 会提示表不存在，这是预期行为
- 远端数据库用户至少需要这些权限：

```sql
GRANT CONNECT ON DATABASE scann_annotation TO scann_sync;
GRANT USAGE, CREATE ON SCHEMA scann_backup TO scann_sync;
ALTER SCHEMA scann_backup OWNER TO scann_sync;
```

- 如果云 PostgreSQL 支持 SSL，建议继续使用 `sslmode=require`
- 如果使用 DBeaver，请分开填写主机、端口、数据库、用户名和密码；不要把 `postgresql://user:password@host:port/db` 直接当作 JDBC URL

其中数据目录通常至少包含：

- `${SCANN_DATASET_DIR}/new`
- `${SCANN_DATASET_DIR}/old`
- `${SCANN_DATASET_DIR}/new_marked`

## 首次部署

```bash
cd /srv/scann/app
git clone <your-repo-url>
cd SCANN_v2/docker
cp .env.example .env
vim .env
chmod +x deploy.sh
./deploy.sh
```

## 日常更新

```bash
cd /srv/scann/app/SCANN_v2
git pull
cd docker
chmod +x deploy.sh
./deploy.sh
```

## 健康检查

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8080/health
```

同步链路部署后，还建议检查：

```bash
docker compose ps
docker inspect scann-native-sync-backend --format '{{.HostConfig.NetworkMode}}'
```

预期 `scann-native-sync-backend` 的网络模式为 `host`。管理员登录前端后，可通过顶部“标注同步”菜单触发增量或全量同步。

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
