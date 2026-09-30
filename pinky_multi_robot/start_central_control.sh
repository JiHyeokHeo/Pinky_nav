#!/usr/bin/env bash
set -eo pipefail
control_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$control_dir")"
if [[ ! -r "$workspace_dir/install/setup.bash" ]]; then
    echo "Missing workspace setup; build pinky_multi_robot first." >&2
    exit 1
fi
source /opt/ros/jazzy/setup.bash
source "$workspace_dir/install/setup.bash"
export ROS_DOMAIN_ID=52
exec ros2 launch "$control_dir/launch/central_control.launch.py" "$@"
