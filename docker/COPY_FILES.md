# 部署文件复制指南

如果自动复制脚本没有成功复制所有文件，请按以下步骤手动复制。

## 方法一：使用 PowerShell 脚本

在本地 Windows 机器上运行：

```powershell
cd G:\wksp\SCANN_v2\docker
powershell -ExecutionPolicy Bypass -File prepare_deployment.ps1
```

## 方法二：手动复制命令

如果自动脚本失败，请在 PowerShell 中手动运行以下命令：

### 1. 创建后端目录结构

```powershell
New-Item -ItemType Directory -Path "G:\wksp\SCANN_v2\docker\backend\src\scann\native_annotation" -Force
```

### 2. 复制后端 Python 文件

```powershell
Copy-Item "G:\wksp\SCANN_v2\scann_v2\src\scann\native_annotation\*.py" "G:\wksp\SCANN_v2\docker\backend\src\scann\native_annotation\" -Force
```

### 3. 创建前端目录结构

```powershell
New-Item -ItemType Directory -Path "G:\wksp\SCANN_v2\docker\frontend\src" -Force
```

### 4. 复制前端源码

```powershell
# 复制所有前端文件
Copy-Item "G:\wksp\SCANN_v2\scann_v2\frontend\*" "G:\wksp\SCANN_v2\docker\frontend\" -Recurse -Force -Exclude @("node_modules", "dist")
```

### 5. 验证文件

```powershell
# 检查后端文件
Get-ChildItem "G:\wksp\SCANN_v2\docker\backend\src\scann\native_annotation\*.py"

# 检查前端文件
Get-ChildItem "G:\wksp\SCANN_v2\docker\frontend\"
```

## 方法三：使用 Git（推荐）

如果你使用 Git 管理项目，可以创建一个分支用于部署：

```bash
cd G:\wksp\SCANN_v2

# 创建并切换到部署分支
git checkout -b deployment

# 仅提交 docker 目录需要的文件
git add docker/backend/ docker/frontend/ docker/docker-compose.yml docker/Dockerfile* docker/.env.example

# 提交
git commit -m "Prepare deployment files"

# 推送到远程仓库（可选）
git push origin deployment

# 然后在服务器上克隆或拉取
```

## 验证清单

在打包之前，请确保以下文件存在：

### 后端必需文件

```
docker/backend/
├── Dockerfile
├── requirements.txt
└── src/scann/native_annotation/
    ├── __init__.py
    ├── app.py
    ├── routes.py
    ├── auth_service.py
    ├── dataset_service.py
    ├── fits_engine.py
    ├── annotation_service.py
    └── task_lock_service.py
```

### 前端必需文件

```
docker/frontend/
├── Dockerfile
├── nginx.conf
├── package.json
├── vite.config.js
├── tailwind.config.cjs
├── postcss.config.cjs
├── index.html
└── src/
    ├── main.js
    ├── App.vue
    ├── style.css
    ├── components/
    ├── composables/
    ├── fits/
    ├── router/
    ├── services/
    └── views/
```

### 根目录必需文件

```
docker/
├── docker-compose.yml
├── .env.example
├── .env
├── README.md
├── DEPLOYMENT.md
└── prepare_deployment.ps1
```

## 打包部署

文件准备好后，运行打包脚本：

```powershell
cd G:\wksp\SCANN_v2\docker

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File pack_deployment.sh

# 或者使用 Git Bash
bash pack_deployment.sh
```

这将创建一个 `.tar.gz` 文件，可以安全地传输到 OpenWRT 服务器。

## 传输到服务器

### 使用 SCP（推荐）

```powershell
# 在 PowerShell 中安装 OpenSSH 客户端（如果尚未安装）
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# 传输文件
scp G:\wksp\SCANN_v2\docker\scann-deployment-*.tar.gz root@<openwrt-ip>:/root/scann-deploy/
```

### 使用 WinSCP

1. 下载并安装 WinSCP
2. 连接到 OpenWRT 服务器
3. 将 `scann-deployment-*.tar.gz` 上传到 `/root/scann-deploy/` 目录

### 使用 FTP/SFTP

如果服务器启用了 FTP 或 SFTP 服务，可以使用 FileZilla 或其他客户端工具上传文件。

## 在服务器上解压

SSH 连接到 OpenWRT 服务器后：

```bash
# 创建部署目录
mkdir -p /root/scann-deploy
cd /root/scann-deploy

# 解压文件
tar -xzf scann-deployment-*.tar.gz

# 删除压缩包（可选）
rm scann-deployment-*.tar.gz

# 配置环境变量
cp .env.example .env
nano .env

# 启动服务
docker-compose up -d
```

## 故障排查

### 文件复制失败

如果 PowerShell 脚本报错，请检查：
1. 路径是否正确
2. 文件是否存在
3. 是否有足够的权限

### 打包失败

如果打包脚本失败，可以手动使用 tar（需要 Git Bash 或 WSL）：

```bash
# 在 Git Bash 中
cd G:/wksp/SCANN_v2/docker
tar -czf scann-deployment.tar.gz backend/ frontend/ docker-compose.yml .env.example README.md DEPLOYMENT.md
```

或者在 Linux/WSL 环境中打包：
```bash
cd /mnt/g/wksp/SCANN_v2/docker
tar -czf scann-deployment.tar.gz backend/ frontend/ docker-compose.yml .env.example README.md DEPLOYMENT.md
```

### 前端文件不完整

如果前端文件不完整，可能需要手动复制 `node_modules` 外的所有文件：

```powershell
$source = "G:\wksp\SCANN_v2\scann_v2\frontend"
$dest = "G:\wksp\SCANN_v2\docker\frontend"

# 排除 node_modules 和 dist
Get-ChildItem -Path $source -Recurse | Where-Object {
    $_.FullName -notlike "*node_modules*" -and $_.FullName -notlike "*dist*" -and $_.PSIsContainer
} | ForEach-Object {
    $relativePath = $_.FullName.Substring($source.Length + 1)
    $destPath = Join-Path $dest $relativePath
    New-Item -ItemType Directory -Path $destPath -Force | Out-Null
}

# 复制文件
Get-ChildItem -Path $source -Recurse -File | Where-Object {
    $_.FullName -notlike "*node_modules*" -and $_.FullName -notlike "*dist*"
} | ForEach-Object {
    $relativePath = $_.FullName.Substring($source.Length + 1)
    $destPath = Join-Path $dest $relativePath
    Copy-Item $_.FullName $destPath -Force
}
```
