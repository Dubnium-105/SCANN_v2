#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created docker/.env from .env.example. Edit it and rerun this script."
  exit 1
fi

invalid_env_lines="$(
  grep -nE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]+[:=]|^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:' .env || true
)"
if [[ -n "$invalid_env_lines" ]]; then
  echo "Invalid docker/.env syntax. Use KEY=value with no spaces around '=' and no ':' after the key."
  echo "$invalid_env_lines"
  exit 1
fi

DATASET_DIR="$(
  grep -E '^[[:space:]]*SCANN_DATASET_DIR=' .env | tail -n 1 | cut -d= -f2- || true
)"
DATASET_DIR="${DATASET_DIR%\"}"
DATASET_DIR="${DATASET_DIR#\"}"
DATASET_DIR="${DATASET_DIR%\'}"
DATASET_DIR="${DATASET_DIR#\'}"
DATASET_DIR="${DATASET_DIR:-./runtime/dataset}"
mkdir -p "$DATASET_DIR/new" "$DATASET_DIR/old" "$DATASET_DIR/new_marked"

export COMPOSE_BAKE="${COMPOSE_BAKE:-false}"

docker compose --env-file .env up -d --build
docker compose --env-file .env ps
