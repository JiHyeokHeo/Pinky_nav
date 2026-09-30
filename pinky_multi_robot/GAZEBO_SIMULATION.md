# 두 핑키 Gazebo 관제 시뮬레이션

실기 Domain 20·22·52와 분리된 **로컬 전용 Domain 152**에서 Gazebo 월드,
두 로봇 Nav2, 위치 확인 노드, 미션 서버, 관제 PyQt UI를 한 번에 실행한다.
시뮬레이션에는 도메인 액션 프록시가 필요 없다. 두 Nav2 서버가 같은 도메인에서
`/pinky1/navigate_to_pose`, `/pinky2/navigate_to_pose`와
`/pinky1/compute_path_to_pose`, `/pinky2/compute_path_to_pose`를 직접 제공한다.
관제 미션 서버가 출발 전 두 Nav2 경로를 비교한다. 경로가 겹치면 pinky2는
pinky1이 끝날 때까지 대기하며, 안전한 대기 장소가 없으면 둘 다 출발하지 않는다.
시뮬레이션 위치 가드는 한 번 `READY`가 된 뒤 정지 중 `/amcl_pose` 갱신만
30초 넘게 뜸해진 경우에는 `READY`를 유지한다. 스캔 손실이나 사용자가
`Set current pose: next click`으로 위치를 다시 지정하는 경우는 예외다.
관제 미션 서버는 출발 전 여전히 최신 AMCL 위치가 필요하므로 오래된 위치만
있으면 주행을 거부한다. 360도 자동 회전 초기화는 사용하지 않는다.

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_description pinky_gz_sim pinky_multi_robot --symlink-install
source install/setup.bash
ros2 launch pinky_multi_robot/launch/gazebo_control.launch.py
```

Gazebo 창이 필요 없으면 마지막 명령에 `gui:=false`를 붙인다. 관제 UI도
빼려면 `show_ui:=false`를 추가한다. 종료는 런치 터미널에서 `Ctrl+C`다.
시뮬레이션 전용 Cyclone DDS 설정은 loopback만 사용하므로 실제 로봇과
발견(discovery)되지 않는다.

관제 UI에서 각 로봇의 처음 위치를 지도에 찍는다. 기본 Gazebo 생성 좌표는
`pinky1=(-0.35, 0.90, 0°)`, `pinky2=(0.65, -0.45, 90°)`다.
`Set current pose: next click`을 누르고 해당 위치를 클릭한 다음, 두 로봇이
`READY`인지 확인한다. `Set goal (stage): next click`으로 각 목표를 저장하고
`Send combined mission`을 누르면 두 주행을 동시에 시험할 수 있다.
시뮬레이션 주차·순찰 저장 파일은 실기 UI 파일과 별개다.

UI 없이 초기 생성 위치에서 두 로봇의 실제 이동을 자동 검사하려면, 위 런치를
`show_ui:=false`로 실행한 채 새 터미널에서 다음을 실행한다.

```bash
python3 ~/ws/pinky_pro/pinky_multi_robot/test/gazebo_drive_smoke.py
```

이 검사는 AMCL 초기 위치를 입력하고 두 로봇이 `READY`가 된 뒤 하나의
`ExecuteMultiRobotMission` 목표를 보낸다. 두 Nav2 결과가 성공이고 각
Gazebo 오도메트리 이동량이 5cm 이상일 때만 `PASS`다. **생성 직후 한 번만**
실행한다. 이미 다른 위치로 이동한 상태에서는 먼저 시뮬레이션을 재시작한다.

다른 터미널에서 수동 확인할 때는 실기용 `~/.ros/cyclonedds_pc.xml` 대신
시뮬레이션 전용 설정을 사용해야 한다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=152
export CYCLONEDDS_URI=file:///home/tory/ws/pinky_pro/pinky_multi_robot/config/cyclonedds_sim.xml
ros2 action list -t
ros2 topic hz /pinky1/scan
```
