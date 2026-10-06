# Pinky two-robot coordinator

## Cross-domain Nav2 action proxy

`domain_bridge` cannot bridge ROS actions. Run this process on the control PC
to expose the two Nav2 action servers on **Domain 52**:

```bash
cd ~/ws/pinky_pro
colcon build --packages-select pinky_multi_robot
source install/setup.bash
ros2 run pinky_multi_robot nav2_action_proxy
```

It relays `/pinky1/navigate_to_pose` (Domain 52 -> Domain 20) and
`/pinky2/navigate_to_pose` (Domain 52 -> Domain 22), including goal,
cancellation, feedback, and result. Keep `domain52_central_bridge.yaml`
running for AMCL pose topics only.

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

When both robots receive one shared UI goal, the controller watches their
AMCL poses. A close head-on encounter makes `pinky2` yield: it cancels its
goal, selects the first collision-free static-map point 0.20–0.50 m behind it
with a 0.30 m map-clearance check, then asks Nav2 to navigate there. Once the
robots are at least 0.85 m apart, it resumes pinky2's original goal. If no
safe retreat exists, pinky2 is held stopped for operator intervention. The
static map is only a first filter; Nav2's local costmap remains responsible
for live-obstacle safety.

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
