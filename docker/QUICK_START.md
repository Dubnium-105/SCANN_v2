# SCANN Native Annotation 快速部署指南

本指南提供在本地 Linux OpenWRT 服务器上容器化部署并通过云服务器 FRP 中转的完整命令。

---

## 📋 前提条件

- 云服务器（Ubuntu/Debian）一台
- 本地 OpenWRT 服务器一台（已安装 Docker）
- 域名（可选，用于 HTTPS）

---

## ☁️ 步骤 1：云服务器配置

### 1.1 安装 Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
```

### 1.2 下载并安装 FRP Server

```bash
cd ~
wget https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_amd64.tar.gz
tar -zxvf frp_0.52.3_linux_amd64.tar.gz
cd frp_0.52.3_linux_amd64
```

### 1.3 创建 FRP 配置文件

```bash
cat > frps.toml << 'EOF'
bindPort = 7000
vhostHTTPPort = 80
vhostHTTPSPort = 443
auth.token = "your-very-strong-random-token-change-this"
webServer.addr = "0.0.0.0"
webServer.port = 7500
webServer.user = "admin"
webServer.password = "another-strong-password"
EOF
```

### 1.4 创建 systemd 服务

```bash
cat > /etc/systemd/system/frps.service << 'EOF'
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
EOF
```

### 1.5 启动 FRP Server

```bash
sudo systemctl daemon-reload
sudo systemctl start frps
sudo systemctl enable frps
sudo systemctl status frps
```

### 1.6 开放防火墙端口

```bash
# UFW 防火墙
sudo ufw allow 7000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 7500/tcp
sudo ufw reload

# 或使用 iptables（如果使用云服务商安全组，请在控制台配置）
```

---

## 🏠 步骤 2：本地 OpenWRT 配置

### 2.1 安装 Docker（如果未安装）

```bash
opkg update
opkg install docker docker-compose dockerd dockerd-compose
/etc/init.d/dockerd start
/etc/init.d/dockerd enable
```

### 2.2 配置 FRP Client

#### 方法 A：通过 Luci Web 界面（推荐）

1. 访问 OpenWRT Web 界面：http://192.168.1.1
2. 导航到：Network → FRPC（需先安装 luci-app-frpc）
3. 添加配置：

**General Settings:**
- Enabled: ✓
- Server Address: `<云服务器公网IP>`
- Server Port: `7000`
- Auth Token: `your-very-strong-random-token-change-this`

**Proxy - Backend:**
- Name: `scann-backend`
- Type: `HTTP`
- Local IP: `127.0.0.1`
- Local Port: `8000`
- Custom Domains: `scann-backend.yourdomain.com`

**Proxy - Frontend:**
- Name: `scann-frontend`
- Type: `HTTP`
- Local IP: `127.0.0.1`
- Local Port: `80`
- Custom Domains: `scann.yourdomain.com`

#### 方法 B：命令行配置

```bash
# 创建配置目录
mkdir -p /etc/frp

# 创建配置文件
cat > /etc/frp/frpc.toml << 'EOF'
serverAddr = "<云服务器公网IP>"
serverPort = 7000
auth.token = "your-very-strong-random-token-change-this"

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
EOF

# 下载 FRPC（根据架构选择）
# OpenWRT 常见架构：mipsle, mips, aarch64, x86_64
wget -O /tmp/frpc.tar.gz https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_mipsle.tar.gz
tar -zxOf /tmp/frpc.tar.gz frpc > /usr/bin/frpc
chmod +x /usr/bin/frpc
rm /tmp/frpc.tar.gz

# 测试运行
/usr/bin/frpc -c /etc/frp/frpc.toml &
```

### 2.3 启动 FRP Client

```bash
# 添加开机自启动
cat > /etc/init.d/frpc << 'EOF'
#!/bin/sh /etc/rc.common
START=99
STOP=10

start() {
    /usr/bin/frpc -c /etc/frp/frpc.toml &
}

stop() {
    killall frpc
}

restart() {
    stop
    sleep 1
    start
}
EOF

chmod +x /etc/init.d/frpc
/etc/init.d/frpc enable
/etc/init.d/frpc start
```

---

## 📦 步骤 3：准备并传输部署文件

### 3.1 在本地 Windows 机器上准备文件

```powershell
# 进入 docker 目录
cd G:\wksp\SCANN_v2\docker

# 运行准备脚本
powershell -ExecutionPolicy Bypass -File prepare_deployment.ps1

# 打包文件
bash pack_deployment.sh

# 或者使用 PowerShell（需要 Git Bash）
# 会在当前目录生成 scann-deployment-YYYYMMDD-HHMMSS.tar.gz
```

### 3.2 传输到 OpenWRT

**使用 SCP（需要 OpenSSH 客户端）：**

```powershell
# 传输文件（替换 <openwrt-ip> 为实际 IP）
scp G:\wksp\SCANN_v2\docker\scann-deployment-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/
```

**使用 WinSCP：**
1. 下载 WinSCP：https://winscp.net/
2. 连接到 OpenWRT
3. 上传 `scann-deployment-*.tar.gz` 到 `/root/scann-deploy/`

**使用 FTP/SFTP：**
如果服务器启用了 FTP/SFTP，使用 FileZilla 或类似工具。

---

## 🚀 步骤 4：在 OpenWRT 上部署

### 4.1 SSH 连接到 OpenWRT

```bash
ssh root@<openwrt-ip>
```

### 4.2 解压部署文件

```bash
# 创建部署目录
mkdir -p /root/scann-deploy
cd /root/scann-deploy

# 解压文件
tar -xzf scann-deployment-*.tar.gz

# 清理压缩包
rm scann-deployment-*.tar.gz

# 查看文件
ls -la
```

### 4.3 配置环境变量

```bash
# 复制并编辑配置文件
cp .env.example .env
vi .env
```

编辑内容（按 i 进入编辑模式，Esc 退出编辑，:wq 保存退出）：

```bash
# SCANN Native Annotation - Environment Configuration

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=80

# Task Lock Timeout (in seconds)
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
```

### 4.4 创建数据集目录

```bash
# 创建必要目录
mkdir -p dataset/new
mkdir -p dataset/old
mkdir -p dataset/new_marked
mkdir -p dataset/annotations
mkdir -p logs

# 上传 FITS 文件到 dataset/new/ 目录
# 使用 scp 或其他工具上传
```

**上传 FITS 文件示例：**

```powershell
# 在本地 Windows 上
scp "G:\wksp\SCANN_v2\dataset\new\*.fts" root@<openwrt-ip>:/root/scann-deploy/dataset/new/
scp "G:\wksp\SCANN_v2\dataset\old\*.fts" root@<openwrt-ip>:/root/scann-deploy/dataset/old/
```

### 4.5 构建并启动 Docker 容器

```bash
# 由于 OpenWRT 资源有限，建议在性能更好的机器上构建镜像后传输
# 如果 OpenWRT 资源充足，可以直接构建：

docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 按 Ctrl+C 退出日志查看
```

**替代方案：在其他机器上构建镜像**

如果 OpenWRT 资源不足，在 Ubuntu/Debian 机器上构建：

```bash
# 在 Ubuntu 机器上
cd /path/to/scann-deploy

# 修改 docker-compose.yml 或直接构建
docker buildx build --platform linux/amd64 -t scann-backend:latest ./backend
docker buildx build --platform linux/amd64 -t scann-frontend:latest ./frontend

# 保存镜像
docker save scann-backend:latest | gzip > scann-backend.tar.gz
docker save scann-frontend:latest | gzip > scann-frontend.tar.gz

# 传输到 OpenWRT
scp scann-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/
```

在 OpenWRT 上加载镜像：

```bash
# SSH 到 OpenWRT
cd /root/scann-deploy

# 加载镜像
gunzip < scann-backend.tar.gz | docker load
gunzip < scann-frontend.tar.gz | docker load

# 清理
rm scann-*.tar.gz

# 启动服务
docker-compose up -d
```

### 4.6 检查服务状态

```bash
# 查看运行的容器
docker ps

# 查看服务日志
docker-compose logs backend
docker-compose logs frontend

# 检查健康状态
curl http://localhost:8000/api/health
curl http://localhost/health

# 进入容器调试
docker exec -it scann-backend sh
docker exec -it scann-frontend sh
```

---

## ✅ 步骤 5：验证部署

### 5.1 检查 FRP 连接

访问云服务器 FRP 管理面板：
- URL: `http://<云服务器公网IP>:7500`
- 用户名: `admin`
- 密码: `another-strong-password`

确认两个代理都已连接：
- scann-backend
- scann-frontend

### 5.2 通过域名访问（需要 DNS 配置）

将以下域名解析到云服务器公网 IP：
- `scann.yourdomain.com` → `<云服务器公网IP>`
- `scann-backend.yourdomain.com` → `<云服务器公网IP>`

访问：
- 前端：http://scann.yourdomain.com
- 后端 API：http://scann-backend.yourdomain.com/api/health

### 5.3 通过云服务器 IP 访问

如果未配置域名，可以使用云服务器公网 IP：
- 前端：http://<云服务器公网IP>（需修改 FRP 配置去掉 customDomains）
- 后端：http://<云服务器公网IP>:<映射端口>

---

## 🔧 日常维护命令

```bash
# 查看所有容器状态
docker ps

# 查看服务日志
docker-compose logs -f
docker-compose logs -f backend
docker-compose logs -f frontend

# 重启服务
docker-compose restart

# 停止服务
docker-compose stop

# 启动服务
docker-compose start

# 完全停止并删除容器
docker-compose down

# 重新构建和启动
docker-compose down
docker-compose build
docker-compose up -d

# 进入容器
docker exec -it scann-backend sh
docker exec -it scann-frontend sh

# 查看资源使用
docker stats

# 备份数据
cd /root/scann-deploy
tar -czf scann-backup-$(date +%Y%m%d).tar.gz dataset/ logs/

# 恢复数据
tar -xzf scann-backup-YYYYMMDD.tar.gz
```

---

## 🔒 安全建议

1. **修改默认密码**
   ```bash
   # 编辑 .env 或通过 API 修改
   # 默认账号：
   # - annotator / scann123
   # - admin / admin123
   ```

2. **配置 HTTPS**
   - 在云服务器安装 Nginx 或 Caddy
   - 配置 SSL 证书（Let's Encrypt 免费证书）
   - 反向代理到 FRP 端口

3. **限制访问**
   - 配置防火墙规则
   - 使用 IP 白名单
   - 启用 Rate Limiting

4. **定期更新**
   ```bash
   # 在 OpenWRT 上
   docker-compose pull
   docker-compose up -d
   ```

5. **定期备份**
   ```bash
   # 设置定时备份（cron）
   crontab -e
   # 添加：0 3 * * * cd /root/scann-deploy && tar -czf ../backups/scann-$(date +\%Y\%m\%d).tar.gz dataset/ logs/
   ```

---

## 🐛 故障排查

### FRP 连接失败

```bash
# 检查云服务器 FRP 日志
ssh root@<云服务器-ip>
journalctl -u frps -f

# 检查 OpenWRT FRPC 日志
logread | grep frpc

# 检查网络连通性
ping <云服务器-ip>
telnet <云服务器-ip> 7000
```

### 容器无法启动

```bash
# 查看详细错误
docker-compose logs backend
docker-compose logs frontend

# 检查端口占用
netstat -tulnp | grep -E '80|8000'

# 检查磁盘空间
df -h

# 检查内存使用
free -m
```

### 前端无法访问后端

```bash
# 检查后端健康
curl http://localhost:8000/api/health

# 检查网络
docker network inspect scann-network
docker exec scann-frontend ping scann-backend
```

---

## 📚 参考资料

- [完整部署文档](./DEPLOYMENT.md)
- [文件复制指南](./COPY_FILES.md)
- [FRP 官方文档](https://github.com/fatedier/frp)
- [Docker Compose 文档](https://docs.docker.com/compose/)

---

## 🆘 获取帮助

如果遇到问题：

1. 查看相关日志文件
2. 参考故障排查章节
3. 查阅项目 GitHub Issues
4. 联系技术支持

---

**部署愉快！** 🎉
