$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$distDir = Join-Path $scriptDir "dist"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archiveName = "scann-linux-deploy-$timestamp.tar.gz"
$archivePath = Join-Path $distDir $archiveName

New-Item -ItemType Directory -Force -Path $distDir | Out-Null

Push-Location $repoRoot
try {
    tar `
        --exclude='scann_v2/frontend/node_modules' `
        --exclude='scann_v2/frontend/dist' `
        --exclude='docker/runtime' `
        --exclude='docker/dist' `
        --exclude='scann_v2/src/scann.egg-info' `
        --exclude='*/__pycache__' `
        -czf $archivePath `
        docker/.env.example `
        docker/backend `
        docker/DEPLOYMENT.md `
        docker/deploy.sh `
        docker/docker-compose.yml `
        docker/frontend `
        docker/README.md `
        scann_v2/frontend `
        scann_v2/pyproject.toml `
        scann_v2/src
}
finally {
    Pop-Location
}

Write-Host "Created deployment bundle: $archivePath"
