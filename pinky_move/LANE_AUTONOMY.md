# YOLO 차선 자율주행

## 2026-09-29: 기존 코드의 metric 제어 개선

현재 보정 파일을 사용하는 경로는 **픽셀 → base_link 바닥 좌표 → 곡선 fitting
→ 중앙 경로 → Pure Pursuit → Twist**입니다. Nav2는 사용하지 않습니다.
설정된 `lane_width`로 한쪽 차선만 있는 시작도 지원하고, 곡선의 법선 방향으로
폭의 절반을 이동해 중앙 경로를 만듭니다. 모두 미검출이면 직전 목표를 짧게
유지하고 감속하여 0.8초 이내 정지합니다. 모호한 검출/추론 오류는 즉시 정지합니다.

이번 변경은 로컬 구현이며 실기 배포·주행 검증은 하지 않았습니다.
함수별 구조·파라미터·제약은 [실행 문서](docs/CONTINUOUS_LANE_RUN.md)를 참고하세요.
아래 near_y 기반 설명은 기존 픽셀 제어 경로 및 이전 변경 기록입니다.

`lane_autonomy`는 `Lane`/`Crossline` 세그멘테이션 모델을 사용해 다음을 수행합니다.

- Lane 마스크의 가까운 기준선과 먼 기준선으로 차로 중심 및 곡률 계산
- 양쪽 라인 중 하나가 잠시 사라지면 최근 차로 폭으로 중심 추정
- 회전량이 클거나 한쪽 라인만 보이면 자동 감속
- 가까운 Crossline을 두 프레임 연속 확인하면 3초 정지
- 같은 Crossline이 시야에서 사라지기 전에는 재정지하지 않음
- 카메라 또는 추론 결과가 오래되면 정지, 사용 가능한 차선이 없으면 정지
- 라이다 가드를 켠 경우에만 라이다 단절과 전방 장애물로 정지

## 2026-09-24 로컬 개선

Pinky2에는 아직 배포하지 않은 변경입니다. 로컬 모의 검증은 실제 코스 완주를
검증하는 것이 아닙니다.

추론은 단일 백그라운드 스레드에서 수행합니다. ROS 제어 타이머, 카메라 수신,
비활성화 서비스는 추론 완료를 기다리지 않습니다. 처리 중 들어오는 영상은
최신 한 장만 보관하며, enable/disable 이전에 시작한 결과는 폐기합니다.

예전에는 추론 시작 시각을 차선 검출 시각으로 저장했습니다. 0.9초 추론 직후
0.45초 차선 소실 기준에 걸리는 문제를 고쳤습니다. 현재 시간 기준은 다음과 같습니다.

| 설정 | 기준 | 기본값 |
| --- | --- | --- |
| `image_timeout` | 마지막 카메라 콜백 수신 후 경과 | 1.5초 |
| `lane_lost_timeout` | 마지막 유효 차선 추론 완료 후 경과 | 1.5초 |
| `result_timeout` | 제어에 사용하는 영상의 콜백 수신 후 경과(추론 시간 포함) | 2.0초 |

모든 내부 타임아웃은 시스템 시각 조정 영향을 받지 않는 단조 시계를 사용합니다.
추론 완료로 카메라 수신 시각을 갱신하지 않습니다. 너무 오래 걸린 추론 결과는
완료되어도 폐기합니다. 입력 수신 시각 기준이므로 네트워크/카메라 내부의 지연까지
측정하는 것은 아닙니다. 새 결과에 사용 가능한 경계가 없으면 즉시 0 속도를 보냅니다.
추론이 지속적으로 느리면 `inference input expired`로 멈출 수 있으며, 이 경우 실제
추론 시간을 확인하고 해상도/모델 성능을 조정해야 합니다. `inference_frequency: 10.0`은
최대 시작 빈도이며 CPU에서 10 FPS를 보장하는 설정이 아닙니다.

가까운 기준 높이 `0.78 → 0.74 → 0.70 → 0.66`을 검사해 유효한 두 경계 쌍을
우선 선택합니다. 범위 전체에 쌍이 없을 때만 가장 가까운 단일 경계를 사용합니다.
두 마스크가 겹치거나 차로 폭 조건을 만족하지 않으면 두 경계로 억지 처리하지 않습니다.
Crossline은 정지선 판단에만 사용하며 Lane 경계 수에 합치지 않습니다.

디버그 영상과 상태 메시지의 값은 다음과 같습니다.

- `Lane`, `Crossline`: 모델이 출력한 각 클래스 인스턴스 수
- `used_boundaries`: 조향 계산에 선택한 Lane 경계 수(0/1/2)
- `candidates`: 선택한 높이를 지나는 유효 Lane 마스크 수
- `near_y`: 실제 선택한 가까운 기준 높이
- `processing`: 디버그 영상의 입력 수신부터 추론 완료까지 걸린 초

예를 들어 `Lane=2 Crossline=1 used_boundaries=1`은 검출된 Lane은 두 개지만
가까운 기준 높이/차로 폭 조건상 한 경계만 사용했다는 뜻입니다.
`WAITING_FOR_LANE: no usable boundary`와 `result expired`도 구분해 표시합니다.
상태 토픽은 값이 같아도 1초마다 발행하여 뒤늦게 구독해도 확인할 수 있습니다.

기존 카메라 180° 회전과 `use_lidar_guard: false` 설정은 유지합니다.
이 설정에서는 라이다 기반 장애물 정지는 동작하지 않으며, Lane/Crossline 모델 자체가
일반 장애물 회피 기능을 제공하는 것은 아닙니다.

## Pinky2에 추후 적용

연결이 가능해지면 다음 파일을 로봇의 대응 경로에 복사하고 다시 빌드합니다.
로컬 `setup.py` 전체를 덮어쓸 필요는 없습니다.

| 로컬 패키지 내 파일 | Pinky2 대상 |
| --- | --- |
| `pinky_move/lane_autonomy.py` | `/home/pinky/pinky_pro/src/pinky_pro/pinky_move/pinky_move/lane_autonomy.py` |
| `pinky_move/lane_logic.py` | `/home/pinky/pinky_pro/src/pinky_pro/pinky_move/pinky_move/lane_logic.py` |
| `config/lane_autonomy.yaml` | `/home/pinky/pinky_pro/src/pinky_pro/pinky_move/config/lane_autonomy.yaml` |

Pinky2 통합 런치는 설정의 로컬 모델 경로 대신 `/home/pinky/models/lane_seg_best.pt`를
지정합니다. Python 파일만 단독 실행한다면 `-p model_path:=/home/pinky/models/lane_seg_best.pt`를
명시해야 합니다.

로봇에서 기존 런치를 종료한 후 실행합니다.

```bash
cd /home/pinky/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_move --symlink-install
source install/setup.bash
ros2 launch pinky_move lane_autonomy_pinky2.launch.py
```

런치는 `DISABLED`로 시작합니다. 영상과 상태를 확인한 뒤 기존 enable 서비스로 켭니다.
진단 터미널도 런치와 같은 ROS 도메인/RMW 설정을 사용해야 합니다. 앞선 로봇 실행 환경은
`ROS_DOMAIN_ID=22`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`였습니다.

## 빌드 및 실행

학습 모델 기본 경로는 다음과 같습니다.

```text
/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt
```

```bash
cd /home/tory/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_move --symlink-install
source install/setup.bash
ros2 launch pinky_move lane_autonomy_robot.launch.py
```

카메라 장치나 모델 위치가 다르면 launch 인자로 지정합니다.

```bash
ros2 launch pinky_move lane_autonomy_robot.launch.py \
  camera_device:=/dev/video1 \
  model_path:=/absolute/path/to/best.pt
```

노드는 안전을 위해 정지 상태로 시작합니다. 로봇 바퀴를 바닥에서 든 상태에서
`/lane_autonomy/debug_image`와 `/lane_autonomy/status`를 먼저 확인합니다.

```bash
ros2 topic echo /lane_autonomy/status
ros2 run rqt_image_view rqt_image_view /lane_autonomy/debug_image
```

정상일 때만 주행을 허용합니다.

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: true}"
```

즉시 정지/비활성화:

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: false}"
```

`pinky_move`나 Nav2처럼 `/cmd_vel`을 발행하는 다른 노드를 동시에 실행하면 안 됩니다.

## 현장 조정 순서

1. `near_y_ratio`와 `far_y_ratio`가 디버그 영상에서 실제 양쪽 Lane을 가로지르는지 확인합니다.
2. 가까운 두 Lane 사이 폭이 영상 폭의 약 몇 %인지 보고 `initial_lane_width_ratio`를 맞춥니다.
3. Crossline 정지 위치가 이르면 `crossline_trigger_y_ratio`를 높이고, 늦으면 낮춥니다.
4. S자에서 반응이 늦으면 `heading_gain`을 조금 올리고 진동하면 낮춥니다.
5. 직선에서 한쪽으로 치우치면 `lateral_gain`을 조정합니다.
6. 충분히 검증하기 전까지 `linear_speed: 0.06`을 올리지 않습니다.

모든 값은 `config/lane_autonomy.yaml`에 있습니다.
