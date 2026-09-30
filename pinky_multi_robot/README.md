# Pinky two-robot coordinator

## Cross-domain Nav2 actions and route reservation

`domain_bridge` carries AMCL pose topics from robot Domains 20/22 to control
Domain 52. The guarded action proxy carries both `NavigateToPose` and
`ComputePathToPose` actions between those domains. Start the production stack
on the control PC with:

```bash
cd ~/ws/pinky_pro
colcon build --packages-select pinky_multi_robot --symlink-install
source install/setup.bash
ros2 launch pinky_multi_robot/launch/central_control.launch.py
```

For a two-robot PARK/PATROL mission, the mission server first asks each robot's
Nav2 planner for every route leg. The two route polylines are compared only
when their map frame names match. This check does not prove physical map
alignment; verify both robots' AMCL poses against the same map before driving.
Routes at least 0.35 m apart start together. If they come
closer, pinky2 waits for pinky1 to finish; if a waiting/parking position itself
blocks the other route, the mission is rejected before either robot moves.
Missing/stale AMCL pose, unavailable planners, and mismatched map frames also
abort the preflight. A 0.55 m current-position check remains as a reactive
fallback after motion begins; if fresh AMCL poses disappear while both robots
are moving, both Nav2 goals are cancelled. Robot-local Nav2 collision avoidance and emergency
stop are still required; this reservation is not an emergency-stop mechanism.

## PyQt map controller

After starting `domain_bridge` and `nav2_action_proxy`, start the Domain 52
desktop controller. Select a robot and click a free point in the map.

```bash
ros2 run pinky_multi_robot nav2_map_ui
```

It defaults to `pinky_navigation/map/my_pinky_map10.yaml` (1 cm per pixel).
Use a different map with:

```bash
ros2 run pinky_multi_robot nav2_map_ui -- --map-yaml /path/to/map.yaml
```

Choose **Both (simultaneous)** to send the clicked goal to both robots. To
configure parking, choose a robot, press **Set parking: next click**, and click
its parking point; **Go to saved parking** reuses that point. Parking locations
are stored locally in `~/.config/pinky_map_ui/parking_goals.yaml`.

For simultaneous but different tasks, choose each robot's Combined mission
task in the UI and use **Send combined mission**. Set goal (stage) selects the
corresponding staged-goal task automatically. A mission goal is sent to
`/execute_multi_robot_mission` on Domain 52, where the preflight above runs.

This node sends one RViz **Publish Point** click to two namespaced Nav2 action
servers. It never publishes `cmd_vel`; every robot keeps its own Nav2 safety
layers. When the robots approach within `conflict_distance`, the non-priority
robot's Nav2 goal is cancelled. It is sent again after the priority robot has
finished or the robots have separated.

Both robot maps must use the same physical coordinate convention: an `(x, y)`
point must refer to the same place for both robots. Configure a distinct TF
tree per robot (`pinky1/map`, `pinky2/map`, and so on) before use.

Build and run on the control PC:

```bash
cd ~/ws/pinky_pro
colcon build --packages-select pinky_multi_robot
source install/setup.bash
export ROS_DOMAIN_ID=22
ros2 run pinky_multi_robot two_pinky_coordinator
```

In RViz select **Publish Point** and click the desired location. The default
input is `/clicked_point`. Do not use Nav2 Goal directly at the same time.

The defaults expect `/pinky1` and `/pinky2`. Example for different namespaces:

```bash
ros2 run pinky_multi_robot two_pinky_coordinator --ros-args \
  -p robot_1_namespace:=pinky_a -p robot_2_namespace:=pinky_b \
  -p robot_1_goal_frame:=pinky_a/map -p robot_2_goal_frame:=pinky_b/map
```

Tune `conflict_distance` and `resume_distance` only after a slow supervised
test. A stopped robot is not always sufficient in a one-robot-wide passage;
for that layout, add a known wait waypoint before running autonomous sharing.
