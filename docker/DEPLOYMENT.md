# SCANN Native Annotation 容器化部署方案

## 方案概述

本方案用于在本地 Linux OpenWRT 服务器上容器化部署 SCANN Native Annotation 服务，并通过云服务器 FRP 中转实现外网访问。

## 架构说明

```
[用户浏览器] → [云服务器 FRP Server] → [本地 OpenWRT FRP Client] → [Docker 容器]
                                                         ↓
                                                    [Backend: 8000]
                                                    [Frontend: 80]
```

## 部署步骤

### 1. 准备服务器环境

在云服务器上操作（假设为 Ubuntu/Debian）：

```bash
# 安装 Docker 和 Docker Compose
sudo apt update
sudo apt install -y docker.io docker-compose-plugin

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 下载 FRP Server（选择合适版本）
wget https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_amd64.tar.gz
tar -zxvf frp_0.52.3_linux_amd64.tar.gz
cd frp_0.52.3_linux_amd64
```

### 2. 配置云服务器 FRP

创建 `frps.toml`：

```bash
sudo nano frps.toml
```

内容：

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

启动 FRP Server：

```bash
# 使用 systemd 管理 FRP
sudo nano /etc/systemd/system/frps.service
```

内容：

```ini
[Unit]
Description=FRP Server Service
After=network.target

[Service]
Type=simple
User=root
Restart=on-failure
RestartSec=5s
ExecStart=/root/frp_0.52.3_linux_amd64/frps -c /root/frp_0.52.3_linux_amd64/frps.toml
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start frps
sudo systemctl enable frps
sudo systemctl status frps
```

确保云服务器安全组开放端口：7000（FRP）、80（HTTP）、443（HTTPS）、7500（管理面板）

### 3. 准备本地 OpenWRT 服务器

在 OpenWRT 上安装 Docker（如果未安装）：

```bash
opkg update
opkg install docker docker-compose dockerd dockerd-compose
/etc/init.d/dockerd start
/etc/init.d/dockerd enable
```

### 4. 配置 OpenWRT FRP Client

通过 OpenWRT Luci 界面配置 FRPC：

1. 登录 OpenWRT Luci
2. 导航到：Services → FRPC（需要先安装 luci-app-frpc）
3. 添加 FRPC 配置：

**General Settings:**
- Enabled: 勾选
- Server Address: 云服务器公网IP
- Server Port: 7000
- Auth Token: your-strong-token-here

**Proxy Settings - Backend (SCANN Backend):**
- Name: scann-backend
- Type: HTTP
- Local IP: localhost（或 Docker 网关 IP，如 172.17.0.1）
- Local Port: 8000
- Custom Domains: scann-backend.yourdomain.com

**Proxy Settings - Frontend (SCANN Frontend):**
- Name: scann-frontend
- Type: HTTP
- Local IP: localhost（或 Docker 网关 IP）
- Local Port: 80
- Custom Domains: scann.yourdomain.com

如果使用命令行配置（不通过 Luci）：

在 OpenWRT 上创建配置：

```bash
mkdir -p /etc/frp
nano /etc/frp/frpc.toml
```

内容：

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

启动 FRPC：

```bash
# 下载 FRPC
wget -O /usr/bin/frpc https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_mipsle.tar.gz
tar -zxvf /usr/bin/frpc
chmod +x /usr/bin/frpc

# 启动
/usr/bin/frpc -c /etc/frp/frpc.toml &
```

### 5. 传输部署文件到 OpenWRT

在本地 Windows 机器上：

```powershell
# 进入 docker 目录
cd G:\wksp\SCANN_v2\docker

# 复制所有文件到 OpenWRT 服务器（需要安装 WinSCP 或使用 scp）
# 假设 OpenWRT IP 为 192.168.1.1
scp -r . root@192.168.1.1:/root/scann-deploy/
```

或者手动传输：
1. 将 `docker` 文件夹中的所有文件通过 SCP/SFTP 传输到 OpenWRT 的 `/root/scann-deploy/` 目录
2. 包括：
   - `backend/` 目录
   - `frontend/` 目录
   - `docker-compose.yml`
   - `.env.example`

### 6. 在 OpenWRT 上部署应用

SSH 连接到 OpenWRT：

```bash
ssh root@192.168.1.1
cd /root/scann-deploy
```

准备数据集目录：

```bash
# 创建数据集目录
mkdir -p dataset/annotations
mkdir -p dataset/positive
mkdir -p dataset/negative
mkdir -p logs

# 如果有现有数据，复制到 dataset 目录
# scp local_dataset/* root@192.168.1.1:/root/scann-deploy/dataset/
```

配置环境变量：

```bash
# 复制并编辑 .env 文件
cp .env.example .env
nano .env
```

修改 `.env` 文件内容：

```bash
# SCANN Native Annotation - Environment Configuration

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=80

# Task Lock Timeout (in seconds)
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
```

构建并启动容器：

```bash
# 构建镜像（首次部署需要，OpenWRT 可能资源有限，建议在性能更强的机器上构建后传输）
# 如果在 OpenWRT 上构建资源不足，可以在其他 Linux 机器上构建后 save/load

docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 查看服务状态
docker-compose ps
```

**替代方案：在其他机器上构建镜像**

由于 OpenWRT 资源有限，建议在 Ubuntu/Debian 机器上构建镜像：

```bash
# 在 Ubuntu 机器上
cd scann-deploy

# 修改 docker-compose.yml 中的架构为 linux/amd64
docker buildx build --platform linux/amd64 -t scann-backend:latest ./backend
docker buildx build --platform linux/amd64 -t scann-frontend:latest ./frontend

# 保存镜像为 tar 文件
docker save scann-backend:latest -o scann-backend.tar
docker save scann-frontend:latest -o scann-frontend.tar

# 传输到 OpenWRT
scp scann-*.tar root@192.168.1.1:/root/scann-deploy/
```

在 OpenWRT 上加载镜像：

```bash
cd /root/scann-deploy
docker load -i scann-backend.tar
docker load -i scann-frontend.tar
rm scann-*.tar
```

然后启动服务：

```bash
docker-compose up -d
```

### 7. 验证部署

检查容器状态：

```bash
# 查看运行的容器
docker ps

# 查看日志
docker logs scann-backend
docker logs scann-frontend

# 进入容器检查
docker exec -it scann-backend sh
docker exec -it scann-frontend sh
```

检查服务健康：

```bash
# Backend health
curl http://localhost:8000/api/health

# Frontend health
wget -O- http://localhost/health
```

检查 FRP 状态：

访问云服务器 FRP 管理面板：`http://your-cloud-server-ip:7500`，确认代理已连接。

### 8. 外网访问测试

通过域名访问：
- 前端：http://scann.yourdomain.com
- 后端 API：http://scann-backend.yourdomain.com/api/health

### 9. 日常维护命令

查看日志：

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

重启服务：

```bash
docker-compose restart
```

停止服务：

```bash
docker-compose down
```

更新部署：

```bash
# 停止并删除旧容器
docker-compose down

# 拉取新代码/传输新文件
# scp 新文件到服务器

# 重新构建和启动
docker-compose build
docker-compose up -d
```

备份数据：

```bash
# 备份数据集
tar -czf scann-dataset-backup-$(date +%Y%m%d).tar.gz dataset/

# 备份日志
tar -czf scann-logs-backup-$(date +%Y%m%d).tar.gz logs/
```

## 故障排查

### 容器无法启动

```bash
# 查看详细错误信息
docker-compose logs backend
docker-compose logs frontend

# 检查端口占用
netstat -tulnp | grep -E '80|8000'
```

### FRP 连接失败

1. 检查云服务器安全组是否开放 7000 端口
2. 检查 OpenWRT 防火墙设置
3. 查看云服务器 FRP 日志：
```bash
journalctl -u frps -f
```

### 性能优化

如果 OpenWRT 资源紧张：

1. 减少容器资源限制（修改 docker-compose.yml）：
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
```

2. 使用外部 PostgreSQL 替代 SQLite（如果需要高并发）
3. 启用 Nginx 缓存

## 安全建议

1. 定期更新镜像和依赖
2. 使用强密码和令牌
3. 启用 HTTPS（在云服务器配置 SSL 证书，使用 Caddy/Nginx 反向代理）
4. 限制 API 访问频率
5. 定期备份数据
