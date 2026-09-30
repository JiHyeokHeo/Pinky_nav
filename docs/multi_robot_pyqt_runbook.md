# Pinky 멀티로봇 PyQt 실행 순서

현재 제어 경로는 아래와 같다.

```text
PyQt UI (Domain 52)
  → /execute_multi_robot_mission
  → mission_action_server (Domain 52)
  → nav2_action_proxy
  → Pinky1 Nav2 (Domain 20), Pinky2 Nav2 (Domain 22)
```

`table_patrol_robot.launch.py`는 이 절차에서 실행하지 않는다.

## 사전 조건

- 각 실제 로봇에 `my_pinky_map10.yaml`과 `my_pinky_map10.pgm`이 있어야 한다.
- Pinky1은 Domain 20, Pinky2는 Domain 22를 사용한다.
- 중앙 PC에도 동일한 map YAML/PGM이 `pinky_navigation/map/`에 있어야 한다.
- 모든 PC가 같은 ROS 2 DDS 네트워크에서 서로 발견 가능해야 한다.

## Pinky1 PC — Domain 20

### 터미널 1: 하드웨어 bringup

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=20

ros2 launch pinky_bringup bringup_robot.launch.xml
```

기존에 라이다·카메라·robot_state_publisher를 함께 실행하는 실제 하드웨어 launch를 사용 중이면, 이 터미널에서는 그 기존 launch를 사용한다.

### 터미널 2: Nav2

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=20

ros2 launch pinky_navigation bringup_launch.xml \
  map:=$HOME/ws/pinky_pro/pinky_navigation/map/my_pinky_map10.yaml \
  params_file:=$HOME/ws/pinky_pro/pinky_navigation/params/nav2_params.yaml
```

확인:

```bash
ROS_DOMAIN_ID=20 ros2 action info /navigate_to_pose
```

`Action servers: 1`이 나와야 한다.

## Pinky2 PC — Domain 22

### 터미널 1: 하드웨어 bringup

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=22

ros2 launch pinky_bringup bringup_robot.launch.xml
```

### 터미널 2: Nav2

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=22

ros2 launch pinky_navigation bringup_launch.xml map:=$HOME/ws/pinky_pro/pinky_navigation/map/my_pinky_map10.yaml params_file:=$HOME/ws/pinky_pro/pinky_navigation/params/nav2_params.yaml
```

확인:

```bash
ROS_DOMAIN_ID=22 ros2 action info /navigate_to_pose
```

`Action servers: 1`이 나와야 한다.

## 중앙 PC — Domain 52

### 최초 한 번: 빌드

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_multi_robot --symlink-install
source install/setup.bash
```

### 터미널 1: AMCL pose bridge

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash

ros2 run domain_bridge domain_bridge \
  ~/ws/pinky_pro/pinky_multi_robot/domain52_central_bridge.yaml
```

### 터미널 2: Nav2 action proxy

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=52

ros2 run pinky_multi_robot nav2_action_proxy
```

정상 로그:

```text
Relaying /pinky1/navigate_to_pose, /pinky2/navigate_to_pose
```

### 터미널 3: 멀티 미션 서버

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=52

ros2 run pinky_multi_robot mission_action_server
```

### 터미널 4: PyQt UI

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=52

ros2 run pinky_multi_robot nav2_map_ui
```

## UI 사용

- 일반 이동: 로봇 선택 후 맵의 빈 곳 클릭. 이 동작은 `PARK` 미션 task를 보낸다.
- 동시 이동: `Both (simultaneous)` 선택 후 맵 클릭.
- 주차 저장: 로봇 선택 → **Set parking: next click** → 맵 클릭.
- 주차 실행: 로봇 선택 → **Go to saved parking**.
- 순찰점 추가: 로봇 선택 → **Add patrol point: next click** → 맵 클릭을 반복.
- 순찰 실행: lap 수 지정 → **Start patrol**.
- 미션 취소: **Cancel selected robot goal**.

저장 위치:

```text
~/.config/pinky_map_ui/parking_goals.yaml
~/.config/pinky_map_ui/patrol_points.yaml
```

## 종료 순서

PyQt UI → mission server → action proxy → pose bridge → 각 로봇 Nav2 → 하드웨어 bringup 순으로 각 터미널에서 `Ctrl+C`를 누른다.
