#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_NAME="scann-linux-deploy-${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="$DIST_DIR/$ARCHIVE_NAME"

mkdir -p "$DIST_DIR"

cd "$REPO_ROOT"

tar \
  --exclude='scann_v2/frontend/node_modules' \
  --exclude='scann_v2/frontend/dist' \
  --exclude='docker/runtime' \
  --exclude='docker/dist' \
  --exclude='scann_v2/src/scann.egg-info' \
  --exclude='*/__pycache__' \
  -czf "$ARCHIVE_PATH" \
  docker/.env.example \
  docker/backend \
  docker/DEPLOYMENT.md \
  docker/deploy.sh \
  docker/docker-compose.yml \
  docker/frontend \
  docker/README.md \
  scann_v2/frontend \
  scann_v2/pyproject.toml \
  scann_v2/src

echo "Created deployment bundle: $ARCHIVE_PATH"
