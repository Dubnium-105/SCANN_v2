# SCANN Native Annotation 完整部署命令清单

本文档提供所有部署所需的具体命令，按步骤分类。

---

## 📋 目录

- [云服务器配置](#云服务器配置)
- [本地 OpenWRT 配置](#本地-openwrt-配置)
- [部署文件准备](#部署文件准备)
- [传输到服务器](#传输到服务器)
- [在 OpenWRT 上部署](#在-openwrt-上部署)
- [验证和测试](#验证和测试)
- [维护命令](#维护命令)
- [故障排查](#故障排查)

---

## ☁️ 云服务器配置

### 安装 Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
docker --version
docker compose version
```

### 下载 FRP Server

```bash
cd ~
wget https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_amd64.tar.gz
tar -zxvf frp_0.52.3_linux_amd64.tar.gz
cd frp_0.52.3_linux_amd64
ls -la
```

### 创建 FRP 配置

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

cat frps.toml
```

### 创建 systemd 服务

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

cat /etc/systemd/system/frps.service
```

### 启动 FRP Server

```bash
sudo systemctl daemon-reload
sudo systemctl start frps
sudo systemctl enable frps
sudo systemctl status frps

# 查看日志
sudo journalctl -u frps -f
# 按 Ctrl+C 退出
```

### 配置防火墙

```bash
# 使用 UFW
sudo ufw allow 7000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 7500/tcp
sudo ufw status

# 或使用 iptables
sudo iptables -A INPUT -p tcp --dport 7000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7500 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

---

## 🏠 本地 OpenWRT 配置

### 安装 Docker

```bash
opkg update
opkg install docker docker-compose dockerd dockerd-compose

# 启动 Docker
/etc/init.d/dockerd start
/etc/init.d/dockerd enable

# 验证安装
docker --version
docker ps
```

### 配置 FRP Client - 命令行方式

```bash
# 创建配置目录
mkdir -p /etc/frp

# 创建配置文件（替换 <云服务器IP> 和 <your-token>）
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

cat /etc/frp/frpc.toml
```

### 下载 FRPC

```bash
# 根据你的 OpenWRT 架构选择
# 常见架构：mipsle, mips, aarch64, x86_64

# 检查架构
uname -m

# 下载（以 mipsle 为例）
wget -O /tmp/frpc.tar.gz https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_mipsle.tar.gz

# 解压并安装
tar -zxOf /tmp/frpc.tar.gz frpc > /usr/bin/frpc
chmod +x /usr/bin/frpc
rm /tmp/frpc.tar.gz

# 验证
frpc --version
```

### 创建 FRP 服务

```bash
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

# 检查状态
logread | grep frpc
```

### 测试 FRP 连接

```bash
# 测试连接云服务器
ping <云服务器公网IP>
telnet <云服务器公网IP> 7000

# 手动启动 FRPC 测试
/usr/bin/frpc -c /etc/frp/frpc.toml
```

---

## 📦 部署文件准备

### Windows 上准备文件

```powershell
# 进入项目目录
cd G:\wksp\SCANN_v2\docker

# 运行准备脚本
powershell -ExecutionPolicy Bypass -File prepare_deployment.ps1

# 查看生成的文件
Get-ChildItem -Recurse
```

### 验证文件完整性

```powershell
# 检查后端文件
Get-ChildItem "G:\wksp\SCANN_v2\docker\backend\src\scann\native_annotation\*.py"

# 检查前端文件
Get-ChildItem "G:\wksp\SCANN_v2\docker\frontend\"

# 检查配置文件
Get-ChildItem "G:\wksp\SCANN_v2\docker\" | Where-Object { $_.Name -like "*.yml" -or $_.Name -like ".env*" }
```

### 打包部署文件

```powershell
cd G:\wksp\SCANN_v2\docker

# 使用 bash（Git Bash 或 WSL）
bash pack_deployment.sh

# 或者使用 PowerShell（需要 tar 命令）
# Windows 10/11 自带 tar
tar -czf scann-deployment.tar.gz backend/ frontend/ docker-compose.yml .env.example README.md DEPLOYMENT.md

# 查看打包结果
Get-ChildItem scann-deployment*.tar.gz
```

---

## 📤 传输到服务器

### 使用 SCP（推荐）

#### 在 Windows PowerShell 中

```powershell
# 首先检查 OpenSSH 客户端是否安装
Get-WindowsCapability -Online -Name OpenSSH.Client*

# 如果未安装，安装 OpenSSH
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# 传输文件
scp G:\wksp\SCANN_v2\docker\scann-deployment-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/

# 如果需要传输数据集
scp -r G:\wksp\SCANN_v2\dataset\* root@<openwrt-ip>:/root/scann-deploy/dataset/
```

#### 在 Git Bash / WSL 中

```bash
# 传输文件
scp /g/wksp/SCANN_v2/docker/scann-deployment-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/

# 传输数据集
scp -r /g/wksp/SCANN_v2/dataset/* root@<openwrt-ip>:/root/scann-deploy/dataset/
```

### 使用 WinSCP

1. 下载 WinSCP：https://winscp.net/
2. 安装并打开 WinSCP
3. 新建站点：
   - 协议：SFTP
   - 主机名：`<openwrt-ip>`
   - 用户名：`root`
   - 密码：你的 OpenWRT 密码
4. 连接后：
   - 导航到 `/root/scann-deploy/`
   - 上传 `scann-deployment-*.tar.gz`
5. 上传完成后，使用 SSH 解压

### 使用 FTP/SFTP

```bash
# 安装 vsftpd（可选）
opkg update
opkg install vsftpd
/etc/init.d/vsftpd start
/etc/init.d/vsftpd enable

# 然后使用 FileZilla 连接并上传文件
```

---

## 🚀 在 OpenWRT 上部署

### SSH 连接到 OpenWRT

```bash
ssh root@<openwrt-ip>

# 如果是首次连接，会有安全提示，输入 yes
```

### 创建部署目录

```bash
mkdir -p /root/scann-deploy
cd /root/scann-deploy
```

### 解压部署文件

```bash
# 解压
tar -xzf scann-deployment-*.tar.gz

# 查看解压后的文件
ls -la

# 清理压缩包
rm scann-deployment-*.tar.gz
```

### 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置（按 i 进入编辑，Esc 退出，:wq 保存）
vi .env
```

编辑后的内容：

```bash
# SCANN Native Annotation - Environment Configuration

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=80

# Task Lock Timeout (in seconds)
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
```

或者直接创建配置：

```bash
cat > .env << 'EOF'
BACKEND_PORT=8000
FRONTEND_PORT=80
SCANN_NATIVE_TASK_LOCK_TIMEOUT_SECONDS=1200
EOF

cat .env
```

### 创建数据集目录

```bash
# 创建目录
mkdir -p dataset/new
mkdir -p dataset/old
mkdir -p dataset/new_marked
mkdir -p dataset/annotations
mkdir -p logs

# 验证
ls -la dataset/
```

### 上传 FITS 文件

```powershell
# 在 Windows PowerShell 中上传
scp "G:\wksp\SCANN_v2\dataset\new\*.fts" root@<openwrt-ip>:/root/scann-deploy/dataset/new/
scp "G:\wksp\SCANN_v2\dataset\old\*.fts" root@<openwrt-ip>:/root/scann-deploy/dataset/old/
scp "G:\wksp\SCANN_v2\dataset\new_marked\*.fts" root@<openwrt-ip>:/root/scann-deploy/dataset/new_marked/
```

```bash
# 在 OpenWRT 上验证
ls -la dataset/new/
ls -la dataset/old/
ls -la dataset/new_marked/
```

### 构建和启动 Docker 容器

#### 方法一：在 OpenWRT 上构建（如果资源充足）

```bash
cd /root/scann-deploy

# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 按 Ctrl+C 退出日志
```

#### 方法二：在其他机器上构建后加载（推荐）

在 Ubuntu/Debian 机器上：

```bash
cd /path/to/scann-deploy

# 构建镜像
docker buildx build --platform linux/amd64 -t scann-backend:latest ./backend
docker buildx build --platform linux/amd64 -t scann-frontend:latest ./frontend

# 保存镜像
docker save scann-backend:latest | gzip > scann-backend.tar.gz
docker save scann-frontend:latest | gzip > scann-frontend.tar.gz

# 传输到 OpenWRT
scp scann-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/
```

在 OpenWRT 上：

```bash
cd /root/scann-deploy

# 加载镜像
gunzip < scann-backend.tar.gz | docker load
gunzip < scann-frontend.tar.gz | docker load

# 清理压缩包
rm scann-*.tar.gz

# 验证镜像
docker images

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 检查服务状态

```bash
# 查看运行的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 查看服务日志
docker-compose logs backend
docker-compose logs frontend

# 实时查看日志
docker-compose logs -f backend
# 按 Ctrl+C 退出

# 检查容器资源使用
docker stats
```

### 测试服务

```bash
# 测试后端健康检查
curl http://localhost:8000/api/health

# 测试前端健康检查
wget -O- http://localhost/health

# 测试后端 API
curl http://localhost:8000/api/tasks
```

---

## ✅ 验证和测试

### 检查 FRP 连接

```bash
# 在云服务器上
ssh root@<云服务器-ip>

# 查看 FRP 日志
journalctl -u frps -f

# 或者访问管理面板
# http://<云服务器IP>:7500
```

### 访问管理面板

打开浏览器：
- URL: `http://<云服务器公网IP>:7500`
- 用户名: `admin`
- 密码: `another-strong-password`

检查是否显示两个代理：
- scann-backend (status: online)
- scann-frontend (status: online)

### 配置 DNS（可选）

将以下域名解析到云服务器公网 IP：
- `scann.yourdomain.com` → `<云服务器公网IP>`
- `scann-backend.yourdomain.com` → `<云服务器公网IP>`

### 测试外网访问

```bash
# 测试前端
curl http://scann.yourdomain.com

# 测试后端 API
curl http://scann-backend.yourdomain.com/api/health
```

### 登录测试

打开浏览器访问：http://scann.yourdomain.com

使用默认账号登录：
- 用户名: `annotator`
- 密码: `scann123`

或管理员账号：
- 用户名: `admin`
- 密码: `admin123`

---

## 🔧 维护命令

### 服务管理

```bash
cd /root/scann-deploy

# 重启所有服务
docker-compose restart

# 重启单个服务
docker-compose restart backend
docker-compose restart frontend

# 停止所有服务
docker-compose stop

# 启动所有服务
docker-compose start

# 完全停止并删除容器
docker-compose down

# 重新构建和启动
docker-compose down
docker-compose build
docker-compose up -d
```

### 日志管理

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs frontend

# 实时查看日志
docker-compose logs -f
docker-compose logs -f backend

# 查看最后 100 行日志
docker-compose logs --tail=100
```

### 容器管理

```bash
# 列出所有容器
docker ps -a

# 进入容器
docker exec -it scann-backend sh
docker exec -it scann-frontend sh

# 在容器内执行命令
docker exec scann-backend ls -la
docker exec scann-backend cat /app/logs/*.log

# 退出容器
exit
```

### 数据备份

```bash
cd /root/scann-deploy

# 备份数据集和日志
tar -czf scann-backup-$(date +%Y%m%d-%H%M%S).tar.gz dataset/ logs/

# 仅备份数据集
tar -czf dataset-backup-$(date +%Y%m%d).tar.gz dataset/

# 仅备份日志
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/

# 列出备份文件
ls -lh scann-backup-*.tar.gz
```

### 数据恢复

```bash
cd /root/scann-deploy

# 停止服务
docker-compose down

# 恢复备份
tar -xzf scann-backup-YYYYMMDD-HHMMSS.tar.gz

# 启动服务
docker-compose up -d
```

### 定时备份

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 3 点备份）
0 3 * * * cd /root/scann-deploy && tar -czf ../backups/scann-$(date +\%Y\%m\%d).tar.gz dataset/ logs/

# 创建备份目录
mkdir -p /root/backups

# 查看定时任务
crontab -l
```

---

## 🐛 故障排查

### FRP 连接问题

```bash
# 检查网络连通性
ping <云服务器公网IP>

# 检查端口是否开放
telnet <云服务器公网IP> 7000

# 检查云服务器 FRP 日志
ssh root@<云服务器-ip>
journalctl -u frps -n 100

# 检查 OpenWRT FRPC 日志
logread | grep frpc

# 重启 FRPC
/etc/init.d/frpc restart
```

### Docker 容器问题

```bash
# 查看容器状态
docker ps -a

# 查看容器详细信息
docker inspect scann-backend
docker inspect scann-frontend

# 查看容器日志
docker logs scann-backend
docker logs scann-frontend

# 重启容器
docker restart scann-backend
docker restart scann-frontend

# 删除并重新创建容器
docker-compose down
docker-compose up -d
```

### 端口占用问题

```bash
# 检查端口占用
netstat -tulnp | grep -E '80|8000'

# 查看进程
ps aux | grep -E 'docker|frp'

# 杀死占用端口的进程
kill -9 <PID>
```

### 磁盘空间不足

```bash
# 检查磁盘使用
df -h

# 检查目录大小
du -sh /root/scann-deploy/*
du -sh /root/scann-deploy/dataset/*

# 清理 Docker 未使用的镜像
docker system prune -a

# 清理日志
rm /root/scann-deploy/logs/*.log
```

### 内存不足

```bash
# 检查内存使用
free -m

# 查看进程内存使用
top

# 重启服务
docker-compose restart
```

### API 错误

```bash
# 测试 API 端点
curl -v http://localhost:8000/api/health
curl -v http://localhost:8000/api/tasks

# 查看 Docker 网络
docker network ls
docker network inspect scann-network

# 测试容器间连接
docker exec scann-frontend ping scann-backend
```

### 前端无法访问后端

```bash
# 检查后端是否运行
curl http://localhost:8000/api/health

# 检查前端配置
docker exec scann-frontend cat /usr/share/nginx/html/index.html

# 进入前端容器调试
docker exec -it scann-frontend sh
# 在容器内
ls -la /usr/share/nginx/html/
cat /usr/share/nginx/html/index.html
exit
```

---

## 📞 获取帮助

### 查看文档

```bash
# 在 OpenWRT 上
cd /root/scann-deploy
cat README.md
cat DEPLOYMENT.md
```

### 查看相关日志

```bash
# Docker 日志
docker-compose logs

# FRPC 日志
logread | grep frpc

# 系统日志
logread
```

### 重置部署

```bash
cd /root/scann-deploy

# 停止并删除所有容器
docker-compose down -v

# 删除镜像
docker rmi scann-backend:latest
docker rmi scann-frontend:latest

# 重新部署
docker-compose build
docker-compose up -d
```

---

## 🎉 完成

恭喜！你已成功部署 SCANN Native Annotation。

**下次登录时：**

```bash
# SSH 到 OpenWRT
ssh root@<openwrt-ip>

# 查看服务状态
cd /root/scann-deploy
docker-compose ps

# 查看日志
docker-compose logs -f
```

**访问应用：**
- 前端：http://scann.yourdomain.com
- 后端 API：http://scann-backend.yourdomain.com
- FRP 管理面板：http://<云服务器IP>:7500

---

**祝你使用愉快！** 🚀
