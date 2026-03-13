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
$releaseName = "scann_v2-$version-minimal-$dateTag"
$outputRoot = Join-Path $projectRoot $OutputDir
$releaseDir = Join-Path $outputRoot $releaseName
$zipPath = Join-Path $outputRoot "$releaseName.zip"

if ($Clean -and (Test-Path $releaseDir)) {
    Remove-Item -Path $releaseDir -Recurse -Force
}

if (Test-Path $releaseDir) {
    throw "Release directory already exists: $releaseDir"
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

$includeItems = @("src", "pyproject.toml", "scann_v2_config.json")
foreach ($item in $includeItems) {
    $sourcePath = Join-Path $projectRoot $item
    if (-not (Test-Path $sourcePath)) {
        throw "Missing required item: $item"
    }
    Copy-Item -Path $sourcePath -Destination $releaseDir -Recurse -Force
}

$rootReadmePath = Join-Path (Split-Path -Parent $projectRoot) "README.md"
if (Test-Path $rootReadmePath) {
    Copy-Item -Path $rootReadmePath -Destination (Join-Path $releaseDir "README.md") -Force
}

$runScript = @"
@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install .
python src\scann\app.py
"@

Set-Content -Path (Join-Path $releaseDir "run_scann_v2.bat") -Value $runScript -Encoding ASCII

$note = @"
# SCANN v2 Minimal Release

Included runtime files:
- src/
- pyproject.toml
- scann_v2_config.json

If root README.md exists, it is copied as well.

Windows first run:
1. Double-click run_scann_v2.bat
2. Script creates .venv, installs dependencies, and launches SCANN v2
"@

Set-Content -Path (Join-Path $releaseDir "MINIMAL_RELEASE_README.md") -Value $note -Encoding UTF8

if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}

Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zipPath -Force

Write-Host "Minimal release generated: $releaseDir"
Write-Host "Zip package: $zipPath"
