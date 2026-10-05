# YOLO 연결 골격 / miter 보완 시험

## 범위

기준은 `1579af9`입니다. 실제 로봇의 YOLO segmentation → 바닥 투영 →
MetricLaneTracker → CornerPolicy → Pure Pursuit 구조를 유지합니다.
OpenCV 흰색 검출 전용 시뮬 우회 경로는 사용하지 않았습니다.

- `connected_geometry` 기본값은 `false`입니다.
- 성공한 기존 axis 추출과 parametric 곡선은 교체하지 않습니다.
- axis 피팅이 실패하거나 두 방향이 모호하면 단일 연결 골격의 관측을 검사합니다.
- 기존 normal offset, 잘린 유효 prefix, S자 prefix 재피팅이 모두 실패한
  CORNER/S_BEND에만 RDP + normal/miter offset을 적용합니다.
- 단순 x 감소를 접힘으로 보지 않습니다. 실제 offset 역전·교차는 거부합니다.
- RDP 오차는 최대 8mm이며, 보이지 않은 출구나 마스크 간 다리를 만들지 않습니다.
- 차선 역할·이력·코너 상태·입력/odom 만료·수동 enable을 유지합니다.
- 계산 캐시는 현재 프레임 내부만 유효하며, 임시 마스크의 id 재사용도 방지합니다.

## 검증을 해석하는 법

최종 결과: 동일 캐시 **380/407 → 385/407**, 기존 성공→실패 **0장**,
기존 성공 목표점 변화 **0m**. 전체 기능 검사 **594개 통과** (66.65초),
`pinky_move` colcon 빌드 성공 (2.29초, 기존 setuptools 경고 있음).
Gazebo YOLO 물리 완주는 **기준 0/4, 최종 0/4**입니다.
최종 이동 거리는 좌 90도 0.572m, 우 90도 0.609m,
급 S 우측 차로 0.469m, 좌측 차로 0.509m였습니다.
네 코스 모두 마지막에 `Lane=0`, `no boundaries detected`가 남았습니다.
기하 복구 효과는 있지만, 물리 주행 문제가 해결됐다고 볼 수 없습니다.

`reports/connected_geometry_20261005/dataset_regression.json`은 같은 모델의
동일 407장 segmentation 캐시를 비교합니다. 독립 사진의 목표 생성 수이며
실제 주행 성공률이나 모델 mAP가 아닙니다. 기존 성공 목표의 좌표 변화도 기록합니다.
개선 사진의 Before/After와 좌표는 `offline_gains/`에 있습니다.

물리 시험은 Domain 172의 독립 Gazebo 환경에서 진행합니다.
좌·우 90도와 급 S의 3라인 양쪽 차로, 폭 20cm, 모델 320px/conf 0.55,
같은 카메라/보정/속도 설정을 사용합니다. 각 주행은 최대 65초입니다.
YOLO 검출 0개 상태는 기하 보완으로 해결할 수 없으며 FAIL로 기록합니다.
끝점 10cm 이내 및 중심 이탈 폭/2 이내를 통과 기준으로 사용합니다.
이 기준은 전체 로봇 footprint의 안전 인증이 아닙니다.

- `baseline/`: 기하 옵션 꺼짐, 기준 시험
- `connected/`: 초기 보완 시험. 기존 성공 목표 좌표가 최대 약 10.45cm 바뀌어 정책을 축소
- `connected_final/`: 기존 성공 경로 보존을 강화한 최종 재시험
- `regression.xml`: 최종 전체 기능 회귀 검사 (스타일 3종 제외)
- `report.html`: 사진·영상 포함 단일 HTML
- `report_linked.html`: 사진 내장, 영상 상대 경로 버전

시뮬 local YOLO는 320px 단일 추론입니다. 기존 실차 PC worker는 Lane이 없을 때
640px 재시도 기능도 있습니다. 이번 물리 비교에서 그 추가 재시도는 사용하지 않았습니다.
따라서 실차 전체 입력/시간 특성까지 동일 검증했다고 주장하지 않습니다.

## 재현

ROS Jazzy와 기존 Pinky description/gz 패키지 overlay를 먼저 source합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
colcon build --packages-select pinky_move --symlink-install
source install/setup.bash
ros2 launch pinky_move lane_gazebo.launch.py \
  domain:=172 course:=left90 perception:=yolo connected_geometry:=true
```

시뮬 화면만 띄우면 처음에는 주행 비활성 상태입니다. 별도 터미널에서 같은
overlay와 Domain 172, launch와 동일한 loopback Cyclone DDS 설정을 사용해야 합니다.
자동 시험 도구는 이 설정과 enable/disable을 시뮬 도메인에만 적용합니다.

```bash
/home/tory/venv/omx/bin/python pinky_move/tools/run_opencv_physics_suite.py \
  --perception yolo --connected-geometry --repeats 1 --seconds 65 \
  --output /tmp/pinky-connected-new-trial
```

도구 이름은 기존 OpenCV 시험 도구지만 `--perception yolo`면 YOLO만 사용합니다.
존재하는 출력 폴더는 덮어쓰지 않습니다. 자신이 만든 프로세스 그룹만 종료합니다.

## 실차 시험 준비

실차 테스트용 별도 브랜치는 `test/robot-connected-geometry-20261005`입니다.
같은 코드를 사용하며 별도 시험 절차를 제공합니다. 기존 설치나 사용자 worktree를
강제로 덮어쓰지 않습니다. 실제 카메라 intrinsic/extrinsic과 모델 파일을 유지합니다.
실차 주행·배포는 이번 작업에서 하지 않았습니다. 물리 완주가 실패했다면
완주 검증본이 아니라 선택형 실험 브랜치로만 사용해야 합니다.
