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

### 동시 임무 도착·경로 재검사 수정 (2026-10-06)

관제 UI의 **Nav2 계획 경로 · 경합 / 양보** 패널에서 경로와 상태를
확인할 수 있습니다. 빨강은 pinky1, 파랑은 pinky2의 관제 검사 경로,
노란 로봇 테두리는 양보 대기입니다. 종료한 임무는 점선으로 남습니다.
경로는 Nav2가 preflight/재검사에 반환한 계획이며 로봇의 모든 내부
재계획을 즉시 스트리밍하는 토픽은 아닙니다. 현재 UI와 지도 frame이
다르면 경로를 그리지 않고 불일치를 표시합니다.

진단 토픽 `/central_control/mission_view`는 Domain 52에서 2Hz로
전달하며, UI를 늦게 열어도 마지막 상태를 받도록 transient-local QoS를
사용합니다. 3초 이상 수신이 끊기면 현재 상태가 아니라 마지막 기록임을
표시합니다. 이 토픽은 주행 명령을 발행하지 않습니다. 출발 거부 시
대기점/주차점과 상대 경로의 거리 및 요구 여유거리도 표시합니다.

- 한 로봇의 도착 성공은 다른 로봇의 임무를 종료하지 않습니다.
- 완료한 로봇은 과거 경로의 이탈 검사에서 제외하지만, 현재 정차
  위치는 경합 계산에 남습니다. 정차한 로봇이 상대 경로를 막으면
  이동 중인 로봇이 양보합니다. 완료했다고 충돌 검사를 끄지 않습니다.
- 초기 경로에서 `route_deviation_limit` (기본 0.25m) 이상 벗어나도
  즉시 전체 임무를 취소하지 않습니다. `route_deviation_confirm_seconds`
  (기본 1.0초) 동안 계속 이탈하면 진행 중인 goal을 잠시 취소하고,
  최신 tracked pose에서 남은 목표까지 Nav2 경로를 다시 요청합니다.
  두 경로를 모두 받은 뒤 정차 위치·경합을 검사하고 같은 목표로 재개합니다.
- 위치가 오래됐거나 경로 계산/취소가 실패하면 정지 상태로 임무를
  실패 처리합니다. 검사 없이 새 경로로 주행하지 않습니다.
- 도착 성공과 양보/재계산 취소가 동시에 발생해도 성공한 goal은
  재발행하지 않습니다. 실제 Nav2 abort를 양보 취소로 오인하지 않습니다.
- 수정은 관제 PC의 `mission_action_server`에만 적용됩니다. 로봇
  Nav2나 YOLO 코드를 바꾸지 않았습니다. 관제 재시작으로 반영됩니다.

검증: 무동력 단위 테스트와 `test/mission_sim_smoke.py`의 loopback 격리
액션 테스트. 실제 두 로봇의 주행 결과를 보장하는 검증은 아닙니다.

### 등록 대피점으로 교착 해소 (2026-10-06)

관제 UI에서 **대피점 추가: 지도 클릭**을 누르고 통로 밖의 빈 공간을
클릭하세요. 보라색 사각형과 대피점 목록으로 표시됩니다. 임무 실행 중에는
추가·삭제할 수 없습니다. 등록은 이동 명령이 아니며, 정적 지도에서 벽으로부터
20cm 여유가 있는 점만 받습니다. 실제 도달 가능성은 임무 때 Nav2로 확인합니다.
두 로봇의 지도 좌표계는 동일한 물리 위치를 가리켜야 합니다.

처리 순서:

1. 대기·주차 위치가 상대 경로를 막지 않는 통과 순서가 있으면 순서를 변경합니다.
2. 양쪽 순서 모두 막히면 등록된 대피점 중 가까운 후보 최대 8개를 검사합니다.
   상대 로봇의 현재 위치를 피하는 이동 경로와 원래 목표로 돌아가는 경로를
   `ComputePathToPose`로 계산합니다. 대피점은 상대 예정 경로에서 기본 55cm
   이상 떨어져야 하고, 경로와 정지한 상대 로봇 사이 기본 35cm 여유를 확인합니다.
3. 상대 로봇을 대기시킨 채 대피 로봇에 `NavigateToPose`를 보냅니다.
   액션 성공 및 추적 위치의 도착 확인 후 원래 목표 경로를 재계산하여 재개합니다.
4. 주행 중 대기 위치가 통로를 막는 경우에도 같은 검사를 적용합니다.
   이미 완료한 로봇은 임의로 이동시키지 않습니다.

강제 후진이나 `/cmd_vel` 직접 발행은 하지 않습니다. Nav2가 방향을 바꿔
이동할 수 있으므로 좁은 공간에 대피점을 등록하지 마세요. 대피 실패·위치 유실·
취소 미확인 시 다른 임무를 시작하지 않습니다. 대피는 로봇당 임무에서 한 번만
시도하며, 등록점이 없거나 양쪽 최종 주차 위치가 서로 막으면 기존처럼 거부합니다.

설정: `yield_detour_timeout` 기본 60초, `yield_arrival_tolerance` 기본 0.15m.
대피점은 `~/.config/pinky_map_ui/yield_points.yaml`에 지도 경로와 함께 저장합니다.
시뮬레이션은 `yield_points_sim.yaml`로 분리합니다. 다른 지도 경로를 열면 이전
대피점을 자동으로 적용하지 않습니다. 서버 입력은 Domain 52의
`/central_control/yield_points` (`geometry_msgs/PoseArray`, transient-local)입니다.
UI 진단에는 `대피점 이동`과 `상대 대피 중 정지` 상태가 표시됩니다.

관제 PC에서 빌드하고 기존 관제를 종료한 뒤 재시작하면 적용됩니다.
로봇 Nav2 패키지는 변경하지 않았습니다.

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_multi_robot --symlink-install
source install/setup.bash
bash ~/ws/pinky_pro/pinky_multi_robot/start_central_control.sh
```

검증: 관련 단위 테스트 81개 통과. 격리된 loopback DDS 액션 테스트에서
대피 도착 → 원래 경로 재계산 → 두 원래 목표 완료를 확인했습니다.
실제 로봇 주행이나 물리 시뮬레이션의 성공을 보장하는 결과는 아닙니다.

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
