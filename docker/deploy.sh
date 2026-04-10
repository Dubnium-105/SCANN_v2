#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created docker/.env from .env.example. Edit it and rerun this script."
  exit 1
fi

set -a
source ./.env
set +a

DATASET_DIR="${SCANN_DATASET_DIR:-./runtime/dataset}"
mkdir -p "$DATASET_DIR/new" "$DATASET_DIR/old" "$DATASET_DIR/new_marked"

export COMPOSE_BAKE="${COMPOSE_BAKE:-false}"

docker compose --env-file .env up -d --build
docker compose --env-file .env ps
