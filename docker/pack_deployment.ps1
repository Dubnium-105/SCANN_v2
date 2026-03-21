# SCANN Native Annotation Deployment Packaging Script
# PowerShell 版本的打包脚本

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SCANN Native Annotation 打包工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 定义路径
$dockerDir = "G:\wksp\SCANN_v2\docker"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archiveName = "scann-deployment-$timestamp.tar.gz"

Write-Host "正在打包部署文件..." -ForegroundColor Yellow
Write-Host ""

# 检查 tar 是否可用
try {
    $tarVersion = & tar --version 2>&1 | Select-Object -First 1
    Write-Host "✓ 找到 tar: $tarVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 未找到 tar 命令" -ForegroundColor Red
    Write-Host "Windows 10/11 应该自带 tar 命令。" -ForegroundColor Yellow
    Write-Host "如果未安装，请安装 Git Bash 或 WSL。" -ForegroundColor Yellow
    exit 1
}

# 切换到 docker 目录
Push-Location $dockerDir

try {
    # 使用 Windows 风格的路径
    Write-Host "正在创建压缩包: $archiveName" -ForegroundColor Cyan

    # 使用 tar 命令打包（Windows tar 语法）
    & tar -czf $archiveName backend frontend docker-compose.yml .env.example README.md DEPLOYMENT.md QUICK_START.md COMMANDS.md COPY_FILES.md 2>&1

    if ($LASTEXITCODE -eq 0) {
        $fileSize = (Get-Item $archiveName).Length / 1MB
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "✓ 打包完成！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "文件: $archiveName" -ForegroundColor Cyan
        Write-Host "大小: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
        Write-Host "路径: $dockerDir\$archiveName" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "传输到 OpenWRT 服务器：" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "使用 SCP 传输：" -ForegroundColor Yellow
        Write-Host "  scp $archiveName root@<openwrt-ip>:/root/scann-deploy/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "使用 WinSCP 传输：" -ForegroundColor Yellow
        Write-Host "  1. 打开 WinSCP" -ForegroundColor Cyan
        Write-Host "  2. 连接到 root@<openwrt-ip>" -ForegroundColor Cyan
        Write-Host "  3. 上传 $archiveName 到 /root/scann-deploy/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "在 OpenWRT 上解压：" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  ssh root@<openwrt-ip>" -ForegroundColor Cyan
        Write-Host "  cd /root/scann-deploy" -ForegroundColor Cyan
        Write-Host "  mkdir -p /root/scann-deploy" -ForegroundColor Cyan
        Write-Host "  tar -xzf $archiveName" -ForegroundColor Cyan
        Write-Host "  rm $archiveName" -ForegroundColor Cyan
        Write-Host "  cp .env.example .env" -ForegroundColor Cyan
        Write-Host "  vi .env  # 编辑配置" -ForegroundColor Cyan
        Write-Host "  mkdir -p dataset/new dataset/old dataset/new_marked dataset/annotations logs" -ForegroundColor Cyan
        Write-Host "  docker-compose up -d" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "详细部署步骤请参考：" -ForegroundColor Yellow
        Write-Host "  QUICK_START.md - 快速开始指南" -ForegroundColor Cyan
        Write-Host "  COMMANDS.md  - 完整命令清单" -ForegroundColor Cyan
        Write-Host "  DEPLOYMENT.md - 详细部署文档" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "✗ 打包失败！" -ForegroundColor Red
        Write-Host "错误代码: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}
