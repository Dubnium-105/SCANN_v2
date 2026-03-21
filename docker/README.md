# SCANN Native Annotation - Docker Deployment

此目录包含用于部署 SCANN Native Annotation 应用的所有必要文件。

## 目录结构

```
docker/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/scann/native_annotation/
│       ├── __init__.py
│       ├── app.py
│       ├── routes.py
│       ├── auth_service.py
│       ├── dataset_service.py
│       ├── fits_engine.py
│       ├── annotation_service.py
│       └── task_lock_service.py
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.cjs
│   ├── postcss.config.cjs
│   └── src/
├── docker-compose.yml
├── .env.example
└── DEPLOYMENT.md
```

## 快速开始

### 1. 准备环境

确保本地服务器已安装：
- Docker
- Docker Compose
- FRP 客户端（OpenWRT 可通过 Luci 配置）

### 2. 配置环境变量

```bash
cp .env.example .env
nano .env
```

### 3. 准备数据集

```bash
mkdir -p dataset/new
mkdir -p dataset/old
mkdir -p dataset/new_marked
mkdir -p dataset/annotations

# 将 FITS 文件复制到相应目录
# cp /path/to/your/fits/*.fts dataset/new/
```

### 4. 构建并启动

```bash
docker-compose build
docker-compose up -d
```

### 5. 查看日志

```bash
docker-compose logs -f
```

### 6. 访问服务

- 前端: http://localhost
- 后端 API: http://localhost:8000/api/health

默认账号:
- 用户名: annotator / 密码: scann123
- 管理员: admin / 密码: admin123

## 云服务器 + FRP 配置

详细配置步骤请参考 [DEPLOYMENT.md](./DEPLOYMENT.md)。

### 云服务器 FRP 配置 (frps.toml)

```toml
bindPort = 7000
vhostHTTPPort = 80
vhostHTTPSPort = 443
auth.token = "your-strong-token-here"
webServer.addr = "0.0.0.0"
webServer.port = 7500
webServer.user = "admin"
webServer.password = "admin-password"
```

### OpenWRT FRP 客户端配置

通过 Luci 界面配置或创建 `/etc/frp/frpc.toml`:

```toml
serverAddr = "your-cloud-server-ip"
serverPort = 7000
auth.token = "your-strong-token-here"

[[proxies]]
name = "scann-backend"
type = "http"
localIP = "127.0.0.1"
localPort = 8000
customDomains = ["scann-backend.yourdomain.com"]

[[proxies]]
name = "scann-frontend"
type = "http"
localIP = "127.0.0.1"
localPort = 80
customDomains = ["scann.yourdomain.com"]
```

## 常用命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f backend
docker-compose logs -f frontend

# 进入容器
docker exec -it scann-backend sh
docker exec -it scann-frontend sh

# 查看状态
docker-compose ps

# 备份数据
tar -czf scann-backup-$(date +%Y%m%d).tar.gz dataset/ logs/
```

## 故障排查

### 端口冲突
修改 `.env` 文件中的端口设置：
```bash
BACKEND_PORT=8001
FRONTEND_PORT=8080
```

### 容器无法启动
```bash
docker-compose logs backend
docker-compose logs frontend
```

### FRP 连接失败
1. 检查云服务器安全组是否开放 7000 端口
2. 检查 FRP 认证令牌是否一致
3. 查看云服务器 FRP 日志

## 注意事项

1. **OpenWRT 资源限制**: 如果 OpenWRT 服务器资源有限，建议在性能更强的机器上构建镜像后传输
2. **数据备份**: 定期备份 `dataset/` 目录
3. **安全**: 生产环境请修改默认密码，使用 HTTPS
4. **日志**: 日志文件存储在 `logs/` 目录

## 支持

如有问题，请参考 [DEPLOYMENT.md](./DEPLOYMENT.md) 或联系技术支持。
