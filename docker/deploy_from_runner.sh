#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_PATH="${DEPLOY_PATH:-}"

if [[ -z "$DEPLOY_PATH" ]]; then
  echo "DEPLOY_PATH is not set. Configure it as a GitHub Actions repository/environment variable."
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required on the self-hosted runner."
  exit 1
fi

mkdir -p "$DEPLOY_PATH"

rsync -a --delete \
  --exclude='.git/' \
  --exclude='.github/' \
  --exclude='.venv/' \
  --exclude='.pytest_cache/' \
  --exclude='dataset/' \
  --exclude='docker/.env' \
  --exclude='docker/dist/' \
  --exclude='docker/runtime/' \
  --exclude='scann_v2/frontend/dist/' \
  --exclude='scann_v2/frontend/node_modules/' \
  --exclude='scann_v2/logs/' \
  --exclude='**/__pycache__/' \
  "$REPO_ROOT/" "$DEPLOY_PATH/"

if [[ ! -f "$DEPLOY_PATH/docker/.env" ]]; then
  cp "$DEPLOY_PATH/docker/.env.example" "$DEPLOY_PATH/docker/.env"
  echo "Created $DEPLOY_PATH/docker/.env from template. Fill in production values and rerun the workflow."
  exit 1
fi

chmod +x "$DEPLOY_PATH/docker/deploy.sh"

pushd "$DEPLOY_PATH/docker" >/dev/null
set -a
source ./.env
set +a

./deploy.sh

wait_for_health() {
  local name="$1"
  local url="$2"
  local attempt

  for attempt in {1..30}; do
    if curl -fsS "$url" >/dev/null; then
      echo "$name is healthy: $url"
      return 0
    fi
    sleep 2
  done

  echo "$name health check failed: $url"
  return 1
}

wait_for_health "backend" "http://127.0.0.1:${BACKEND_PORT:-8000}/api/health"
wait_for_health "frontend" "http://127.0.0.1:${FRONTEND_PORT:-8080}/health"
popd >/dev/null

echo "Deployment finished at $DEPLOY_PATH"
