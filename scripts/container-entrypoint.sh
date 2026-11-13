#!/usr/bin/env bash
set -euo pipefail
# ROS setup scripts read optional variables that may be unset.
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
