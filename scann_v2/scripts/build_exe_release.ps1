param(
    [string]$OutputDir = "release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$pyprojectPath = Join-Path $projectRoot "pyproject.toml"
if (-not (Test-Path $pyprojectPath)) {
    throw "pyproject.toml not found. Run this script inside scann_v2."
}

$version = ""
Get-Content -Path $pyprojectPath | ForEach-Object {
    if ($_ -match '^version\s*=\s*"([^"]+)"') {
        $script:version = $Matches[1]
    }
}
if (-not $version) {
    throw "Cannot parse version from pyproject.toml"
}

$dateTag = Get-Date -Format "yyyyMMdd"
$releaseName = "scann_v2-$version-exe-$dateTag"
$outputRoot = Join-Path $projectRoot $OutputDir
$releaseDir = Join-Path $outputRoot $releaseName
$zipPath = Join-Path $outputRoot "$releaseName.zip"

if ($Clean) {
    if (Test-Path (Join-Path $projectRoot "build")) {
        Remove-Item -Path (Join-Path $projectRoot "build") -Recurse -Force
    }
    if (Test-Path (Join-Path $projectRoot "dist")) {
        Remove-Item -Path (Join-Path $projectRoot "dist") -Recurse -Force
    }
    if (Test-Path $releaseDir) {
        Remove-Item -Path $releaseDir -Recurse -Force
    }
}

if (Test-Path $releaseDir) {
    throw "Release directory already exists: $releaseDir"
}

$workspaceRoot = Split-Path -Parent $projectRoot
$venvPython = Join-Path $workspaceRoot "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pythonCmd = "python"
}

$pyInstallerCheck = & $pythonCmd -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in build environment: $pythonCmd"
}

Write-Host "Building executable with PyInstaller (onedir)..."
& $pythonCmd -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name SCANN_v2 `
    --paths src `
    --add-data "scann_v2_config.json;." `
    --collect-all torch `
    --collect-all torchvision `
    --collect-all astropy `
    --collect-all astroquery `
    --collect-all skimage `
    --collect-all sklearn `
    src/scann/app.py | Out-Host

$distDir = Join-Path $projectRoot "dist\SCANN_v2"
if (-not (Test-Path $distDir)) {
    throw "PyInstaller output not found: $distDir"
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
Copy-Item -Path $distDir -Destination (Join-Path $releaseDir "SCANN_v2") -Recurse -Force

$runScript = @"
@echo off
setlocal
cd /d %~dp0\SCANN_v2
start "" SCANN_v2.exe
"@
Set-Content -Path (Join-Path $releaseDir "run_scann_v2.bat") -Value $runScript -Encoding ASCII

$note = @"
# SCANN v2 Executable Release

Usage (Windows):
1. Enter SCANN_v2 folder and run SCANN_v2.exe, or
2. Double-click run_scann_v2.bat

This distribution is fully bundled and does not require Python or pip on user machines.
"@
Set-Content -Path (Join-Path $releaseDir "EXE_RELEASE_README.md") -Value $note -Encoding UTF8

if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}

$zipSucceeded = $true
try {
    Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath -Force
} catch {
    $zipSucceeded = $false
    Write-Warning "Zip packaging skipped (package too large for Compress-Archive). Release folder is ready: $releaseDir"
}

Write-Host "Executable release generated: $releaseDir"
if ($zipSucceeded) {
    Write-Host "Zip package: $zipPath"
}
