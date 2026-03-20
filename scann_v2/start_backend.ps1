# SCANN Native Annotation Backend 启动脚本
$env:SCANN_NATIVE_DATASET_ROOT="G:\wksp\SCANN_v2\dataset"
Write-Host "SCANN_NATIVE_DATASET_ROOT set to: $env:SCANN_NATIVE_DATASET_ROOT"
G:\wksp\SCANN_v2\.venv\Scripts\python.exe -m uvicorn scann.native_annotation.app:app --reload --port 8000
