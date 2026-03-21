# SCANN Native Annotation Deployment Preparation Script
# 用于将所有必要的文件复制到 docker 目录以便部署

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SCANN Native Annotation 部署准备工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 定义路径
$workspaceRoot = "G:\wksp\SCANN_v2"
$dockerDir = Join-Path $workspaceRoot "docker"
$backendSrc = Join-Path $workspaceRoot "scann_v2\src\scann\native_annotation"
$frontendSrc = Join-Path $workspaceRoot "scann_v2\frontend"
$backendDest = Join-Path $dockerDir "backend\src\scann\native_annotation"
$frontendDest = Join-Path $dockerDir "frontend"

# 1. 清理并创建目录
Write-Host "[1/6] 清理并创建目录结构..." -ForegroundColor Yellow
Remove-Item -Path "$backendDest\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$frontendDest\src\*" -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $backendDest -Force | Out-Null
New-Item -ItemType Directory -Path "$frontendDest\src" -Force | Out-Null
Write-Host "   ✓ 目录结构创建完成" -ForegroundColor Green
Write-Host ""

# 2. 复制后端 Python 文件
Write-Host "[2/6] 复制后端 Python 文件..." -ForegroundColor Yellow
$backendFiles = @(
    "__init__.py",
    "app.py",
    "routes.py",
    "auth_service.py",
    "dataset_service.py",
    "fits_engine.py",
    "annotation_service.py",
    "task_lock_service.py"
)

foreach ($file in $backendFiles) {
    $src = Join-Path $backendSrc $file
    if (Test-Path $src) {
        Copy-Item $src $backendDest -Force
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (不存在，已跳过)" -ForegroundColor Red
    }
}
Write-Host ""

# 3. 复制前端源码
Write-Host "[3/6] 复制前端源码..." -ForegroundColor Yellow
Copy-Item -Path "$frontendSrc\*" -Destination "$frontendDest" -Recurse -Force -Exclude @("node_modules", "dist", ".DS_Store")
Write-Host "   ✓ 前端文件复制完成" -ForegroundColor Green
Write-Host ""

# 4. 复制前端配置文件（如果需要）
Write-Host "[4/6] 验证配置文件..." -ForegroundColor Yellow
$configFiles = @(
    "package.json",
    "vite.config.js",
    "tailwind.config.cjs",
    "postcss.config.cjs",
    "index.html"
)

foreach ($file in $configFiles) {
    $dest = Join-Path $frontendDest $file
    if (Test-Path $dest) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file (缺失)" -ForegroundColor Red
    }
}
Write-Host ""

# 5. 创建 .env 文件
Write-Host "[5/6] 创建 .env 文件..." -ForegroundColor Yellow
$envExample = Join-Path $dockerDir ".env.example"
$envFile = Join-Path $dockerDir ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "   ✓ .env 文件已创建（请根据需要修改）" -ForegroundColor Green
} else {
    Write-Host "   ℹ .env 文件已存在，保持不变" -ForegroundColor Cyan
}
Write-Host ""

# 6. 创建打包脚本
Write-Host "[6/6] 创建部署打包脚本..." -ForegroundColor Yellow
$packScript = @"
# 打包部署文件
# 在此目录运行此脚本以创建用于传输的压缩包

`$archiveName = "scann-deployment-`$(Get-Date -Format 'yyyyMMdd-HHmmss').tar.gz"

Write-Host "正在打包部署文件..." -ForegroundColor Cyan
tar -czf "`$archiveName" `
    backend/ `
    frontend/ `
    docker-compose.yml `
    .env `
    README.md `
    DEPLOYMENT.md

Write-Host ""
Write-Host "✓ 打包完成: `$archiveName" -ForegroundColor Green
Write-Host ""
Write-Host "请将此文件传输到 OpenWRT 服务器：" -ForegroundColor Yellow
Write-Host "  scp `$archiveName root@<openwrt-ip>:/root/scann-deploy/" -ForegroundColor Cyan
Write-Host ""
Write-Host "然后在 OpenWRT 上解压：" -ForegroundColor Yellow
Write-Host "  ssh root@<openwrt-ip>" -ForegroundColor Cyan
Write-Host "  cd /root/scann-deploy" -ForegroundColor Cyan
Write-Host "  tar -xzf `$archiveName" -ForegroundColor Cyan
Write-Host "  rm `$archiveName" -ForegroundColor Cyan
Write-Host "  cp .env.example .env" -ForegroundColor Cyan
Write-Host "  nano .env  # 编辑配置" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor Cyan
"@

$packScriptPath = Join-Path $dockerDir "pack_deployment.sh"
$packScriptPath | Out-File -FilePath $packScriptPath -Encoding UTF8
Write-Host "   ✓ 打包脚本已创建: pack_deployment.sh" -ForegroundColor Green
Write-Host ""

# 完成
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "准备完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "接下来：" -ForegroundColor Yellow
Write-Host "1. 编辑 docker/.env 文件配置环境变量" -ForegroundColor Cyan
Write-Host "2. 将 docker 文件夹传输到 OpenWRT 服务器" -ForegroundColor Cyan
Write-Host "3. 参考 DEPLOYMENT.md 完成部署" -ForegroundColor Cyan
Write-Host ""
Write-Host "快速打包命令：" -ForegroundColor Yellow
Write-Host "  cd G:\wksp\SCANN_v2\docker" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File pack_deployment.sh" -ForegroundColor Cyan
Write-Host ""
