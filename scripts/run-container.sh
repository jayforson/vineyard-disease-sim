#!/usr/bin/env bash
# Isolated container, no host networking/devices or privileged access.
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker run --rm --init --cpus=4 --memory=6g --pids-limit=512 \
  --shm-size=512m --user "$(id -u):$(id -g)" --entrypoint bash \
  -e HOME=/workspace/.cache/home -e ROS_HOME=/workspace/.cache/ros -e HF_HOME=/workspace/data/cache/hf_home \
  -e ROS_DOMAIN_ID=73 -e GZ_PARTITION=vineyard_disease_sim -e GZ_IP=127.0.0.1 -v "$PWD:/workspace" \
  vineyard-disease-sim:dev scripts/container-entrypoint.sh "$@"
