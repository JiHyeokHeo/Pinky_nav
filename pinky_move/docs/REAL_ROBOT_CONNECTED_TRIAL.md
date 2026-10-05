# 실차 연결 경로 보완 시험 브랜치

브랜치: `test/robot-connected-geometry-20261005`

기하 코드 기준 커밋: `3b44a87`. 시뮬/실차 모두 같은 MetricLaneTracker와
CornerPolicy를 사용합니다. 실차용 변경은 선택형 저속 프로파일과 이 실행 문서입니다.
실차에 아직 배포하거나 주행하지 않았습니다. Gazebo YOLO 물리 완주는 0/4였으므로
완주 검증본이나 기존 버전 대체본으로 취급하지 마세요.

## 준비

기존 dirty worktree를 강제로 checkout/reset하지 말고 별도 폴더를 사용하세요.
PC와 로봇의 `pinky_move`는 이 브랜치의 같은 커밋이어야 합니다.
비공개 저장소라 로봇에서 GitHub 인증이 없다면 PC에서 받은 소스를 별도
로봇 시험 workspace로 복사하세요. 모델과 실차 calibration.json은 기존 파일을 유지하세요.

```bash
git clone --branch test/robot-connected-geometry-20261005 \
  https://github.com/JiHyeokHeo/Pinky_nav.git ~/pinky_connected_trial
cd ~/pinky_connected_trial
source /opt/ros/jazzy/setup.bash
# PC: 기존 ~/ws/pinky_pro/install/setup.bash
# 로봇: 기존 ~/pinky_pro/install/setup.bash
colcon build --packages-select pinky_interfaces pinky_move --symlink-install
source install/setup.bash
```

기존 overlay의 description/bringup 및 ROS 메시지 패키지가 필요합니다.
Pinky2 Domain 22의 기존 Cyclone DDS 설정을 쓰세요. 시뮬 Domain 172용
loopback DDS 설정이나 시뮬 이상 보정을 실제 장비에 복사하지 마세요.

## 1. 무동력 확인 — 로봇

기존 robot bringup과 cam_stream.py가 각각 한 번만 실행돼 있어야 합니다.
Nav2/teleop/예전 lane_autonomy 등 다른 cmd_vel 발행 노드는 중복 실행하지 마세요.
아래 명령은 카메라나 하드웨어 bringup을 추가 실행하지 않습니다.

```bash
export ROS_DOMAIN_ID=22
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source ~/pinky_connected_trial/install/setup.bash
ros2 run pinky_move lane_autonomy --ros-args \
  --params-file ~/pinky_connected_trial/pinky_move/config/lane_autonomy.yaml \
  --params-file ~/pinky_connected_trial/pinky_move/config/lane_connected_robot_test.yaml \
  -p model_path:=/home/pinky/models/lane_seg_best.pt \
  -p calibration_path:=/home/pinky/pinky_connected_trial/pinky_move/config/robot_floor_calibration.json
```

위 calibration 경로는 브랜치에 저장된 실차 보정 파일입니다. 실제 장착 상태에
맞는 더 최신 보정이 따로 있다면 그 **검증된 실차 파일 경로**로 바꾸세요.
새 보정이나 calibration 검증 해제는 이번 수정에 포함되지 않았습니다.
이 프로파일은 `enabled=false`로 시작합니다.

## 2. PC에서 기존 YOLO/기하 worker 실행

```bash
source /opt/ros/jazzy/setup.bash
source ~/pinky_connected_trial/install/setup.bash
ros2 launch pinky_move lane_inference_pc.launch.py \
  robot:=pinky@192.168.45.22 \
  model_path:=/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt
```

기존 SSH 인증과 모델 SHA256 일치가 필요합니다. 로봇이 보낸
`connected_geometry` 설정을 PC planner가 받아 계산합니다. 좌표 변환과 이력은
기존 PC 쪽 구성, 최종 cmd_vel/watchdog 책임은 로봇 쪽으로 유지합니다.
worker의 기존 640px 무검출 재시도도 그대로 있습니다. 물리 시뮬 비교는 320px
단일 추론이었으므로 동일 처리시간/검출 성능이라고 주장하지 않습니다.

## 3. 출발 전에 확인

로봇 터미널 또는 Domain 22를 볼 수 있는 PC 터미널에서:

```bash
ros2 param get /lane_autonomy enabled
ros2 param get /lane_autonomy connected_geometry
ros2 param get /lane_autonomy simulation_white_lane
ros2 topic list --no-daemon
```

순서대로 false / true / false인지 확인하세요.
카메라·odom 최신성, 보정 오류 없음, debug image의 좌우/목표점 위치를 먼저 봅니다.
PC rqt에서 `/lane_autonomy/debug_image`를 선택할 수 있습니다.
연결 기하도 관측을 새로 만들지는 않으므로 `Lane=0` 정지는 여전히 가능합니다.

## 4. 직접 감시하에 짧은 주행만 수동 시작

로봇 옆에서 즉시 물리 정지가 가능할 때만 진행하세요. 이번 작업에서는 아래
enable 명령을 실제 로봇에 호출하지 않았습니다.

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: true}"
```

정지:

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: false}"
```

각 시험은 먼저 무동력 → 직선 → 한쪽 경계 → 좌/우 코너 → S자 순으로
확인하고, 실제 영상·odom·상태·cmd_vel 로그를 저장하세요. 정지 원인을 검출 소실,
기하 실패, 역할 연결 실패, 입력/odom 만료로 구분합니다.
`connected miter fallback on observed boundary`가 복구 진단입니다.
실제 차선폭과 보정 오차의 검증 없이 이번 5장 목표 생성 개선을 완주 개선으로
해석하지 마세요.

## 되돌리기

프로파일 두 번째 `--params-file`을 빼면 기존 기본값 connected_geometry=false로
시작합니다. 단, model_path/calibration_path와 원격 추론 여부는 기존 실행 설정을
사용하세요. 또는 기존 Pinky2 launch를 `connected_geometry:=false`로 실행합니다.
실행 중 파라미터만 바꾸면 tracker가 재생성되지 않으므로 옵션을 바꿀 때는
명시적으로 비활성화하고 lane_autonomy를 재시작하세요.
