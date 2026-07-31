#!/bin/sh
set -eu

seed_dir="${SCANN_TORCH_CACHE_SEED_DIR:-/opt/scann-cache-seed}"
cache_dir="${TORCH_HOME:-/opt/torch-cache}"

if [ -d "${seed_dir}" ]; then
    mkdir -p "${cache_dir}"
    cp -a -n "${seed_dir}/." "${cache_dir}/"
fi

exec "$@"
