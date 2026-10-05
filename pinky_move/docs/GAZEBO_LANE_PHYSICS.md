# Gazebo 카메라 차선 물리 주행 시험

Nav2 없이 **가상 카메라 → 실제 기존 YOLO segmentation 모델 → 기존
`lane_autonomy` → 가상 `/cmd_vel` → Gazebo 바퀴·접촉 물리**를 연결한다.
저장 사진 반복이나 목표점 숫자 시뮬레이션이 아니라 URDF 물리 모델이 실제로 움직인다.
실제 로봇에 SSH/서비스/속도 명령을 보내지 않는다.

## 새 OpenCV 전용 시험 (2026-10-05)

요청에 따라 **여기 Gazebo에서만** `perception:=opencv`를 추가했다.
기본값 `perception:=yolo`와 실제 로봇 설정은 변경하지 않았다.
OpenCV 모드에서는 모델을 로드하지 않고 가상 카메라의 HSV 흰 픽셀을 검출한다.
카메라 intrinsic + URDF extrinsic으로 바닥에 투영하고, 연결된 선의 법선 방향
offset / 직각 miter 교점으로 중앙 경로를 생성해 기존 Pure Pursuit로 제어한다.
3라인에서 현재 차로에 가까운 좌·우 경계를 선택한다. 코스 좌표나 ground truth는
제어에 쓰지 않고 시험 완주·이탈 측정에만 사용한다.

```bash
ros2 launch pinky_move lane_gazebo.launch.py \
  course:=s_sharp lane_count:=2 target_lane:=0 perception:=opencv gui:=true domain:=172
# 좌/우 직각: course:=left90 / course:=right90 (lane_count:=1)
# 반대 차로에서도 시험: target_lane:=1
```

자동 반복 시험 (다른 Gazebo launch를 먼저 종료하고, 새 출력 폴더 지정):

```bash
source /opt/ros/jazzy/setup.bash
source /home/tory/ws/pinky_pro/install/setup.bash
source /home/tory/pinky_nav_publish_yWXwM0/repo/install/setup.bash
cd /home/tory/pinky_nav_publish_yWXwM0/repo
/usr/bin/python3 pinky_move/tools/run_opencv_physics_suite.py \
  --output pinky_move/reports/my_new_opencv_suite --repeats 2 --seconds 100
```

좌90·우90·급 S자 오른쪽 차로·급 S자 왼쪽 차로를 각 2회씩 **순차** 실행한다.
자동 시험은 가상 로봇만 출발시키고 매 회 종료 시 disable한다. 실패도 보존한다.
회귀 검사·Gazebo 완주는 실제 조명, 카메라 지연, 마운트 오차가 있는 실차 완주
보증이 아니다. 기존 staged-corner 정체성 제한을 우회하는 것은 격리된 OpenCV
시뮬 프로파일만이며 `/home/tory/ws/pinky_pro`에 복사하지 않았다.

개선 과정: 초기 S자 v1~v4는 경계 정체성/코너 진입/전방 경로 제한으로 정지했다.
v5는 47초·2.60m 이동·최대 중심 이탈 6.12cm로 처음 완주했다.
목표점을 더 가깝게 잡은 v6는 반대 방향 도로로 전환해 이탈했으므로 채택하지 않았다.
각 영상과 원문 JSON은 `reports/gazebo_physics_20261005/`에 실패까지 남긴다.
첫 반복 suite는 6회 완주했지만 반대 차로 2회가 코스 생성 전에 실패했다.
기존 60cm 구간에서 세 번째 경계의 두 miter 꼭짓점이 겹쳐 길이가 0이 됐다.
급 S자 직각 네 번은 유지하면서 가운데 구간을 70cm로 바꿔 두 차로 모두
물리적으로 생성되게 수정했다. 폭에 비해 지나치게 좁아 경계가 붕괴/역전되는
코스는 이제 생성 단계에서 명확하게 거부한다. 과거 50/60cm 시험과 새 50/70cm
반복 시험은 `course.json`의 좌표 및 폴더 이름으로 구분한다.

### 최종 반복 검증 결과

`opencv_final_suite_v2/suite.json` 및 각 `trial.json`에 원문을 보존했다.

| 조건 | 완주 | 최대 중심 이탈 |
|---|---:|---:|
| 좌90 / 경계선 2개 | 2/2 | 5.65cm |
| 우90 / 경계선 2개 | 2/2 | 5.44cm |
| 급 S자 / 3라인 / 오른쪽 차로 | 2/2 | 7.53cm |
| 급 S자 / 3라인 / 왼쪽 차로 | 2/2 | 6.00cm |

총 **8/8 완주**, 기록 벽시계 26.4~51.1초, SAFETY_STOP 0회.
일부 시작/재관측 프레임에는 짧은 WAITING_FOR_LANE이 있으므로 무정지 운전을
보증하는 수치가 아니다. 기준은 중심점 + 목표 반경 10cm이며 footprint 전체
포함이나 실제 환경의 안전성은 보증하지 않는다.
전체 기능 회귀 **581 passed / 70.42초** (copyright/flake8/pep257 3종 제외),
pinky_description·pinky_move 빌드 2개 성공(2.31초).
Firefox headless에서 새 시험의 HTML 재생기 **16/16 실제 play() 성공**,
원본·디버그 영상 모두 H.264 / yuv420p / 전체 스트림 decode 검사를 통과했다.
`opencv_regression.xml`, `opencv_browser_validation.json`에도 검증 결과가 있다.

테스트 중 발견한 두 추가 문제도 보완했다. 완만한 `s_bend`는 곡률 반경이
3번째 경계 offset보다 작아 뒤집히므로 비교용 완만한 형상을 조정했다.
디버그 이미지 로컬 전송 테스트는 로봇용 DDS 설정을 상속해 localhost peer를
못 찾았으므로 **테스트만** loopback으로 격리했다. 운영 DDS 파일은 변경하지 않았다.

## 코스

| course | 내용 |
|---|---|
| `two_lines` | 평행한 흰 선 두 개, 2m 직선, 두 경계 중앙 추종 |
| `left90` | 0.75m 앞 좌측 직각, 이후 1.25m 직선 |
| `right90` | 좌측 코스의 좌우 반전 |
| `s_bend` | 비교용 완만한 연속 S자, x=3.0m까지, 3번째 선이 접히지 않는 곡률 |
| `s_sharp` | 좌→우→우→좌 직각 네 번, 코너 사이 50/70cm |
| `single_gap` | 두 선 직선에서 왼쪽 선의 x=0.60–0.95m 부분 소실 |

차선 중심선 간 폭은 기본 **20cm**이며 `lane_width:=0.24` 등으로 바꿀 수 있다.
흰 선 두께는 2cm. 목표점은 원래 코스 끝의 반경 10cm 범위이며, 전방 카메라가
목표 이전에 선 끝을 놓치는 효과를 분리하기 위해 흰 선은 끝점 뒤 50cm까지 연장한다.
실제 코스 폭/실차 파라미터를 변경한 것이 아니다.

## 설치 및 빌드

필요 환경: ROS2 Jazzy, Gazebo Harmonic, ros_gz_bridge, xacro,
기존 segmentation 모델 및 ultralytics가 설치된 Python.
현재 PC에는 이 환경과 `/home/tory/venv/omx/bin/python`이 확인됐다.

현재 작업 저장소에서:

```bash
cd /home/tory/pinky_nav_publish_yWXwM0/repo
source /opt/ros/jazzy/setup.bash
source /home/tory/ws/pinky_pro/install/setup.bash
colcon build --base-paths pinky_description pinky_move \
  --packages-select pinky_description pinky_move --symlink-install
source install/setup.bash
```

다른 PC/새 clone에서는 위 저장소 경로와 Python/model 경로를 해당 환경에 맞춘다.
동일 모델의 저장소 위치는 `project_assets/yolo_runs/segment/train/weights/best.pt`다.
이 실행은 `pinky_bringup`이나 실제 Nav2를 시작하지 않는다.

## 1. 화면을 보면서 수동 시작

```bash
source /opt/ros/jazzy/setup.bash
source /home/tory/pinky_nav_publish_yWXwM0/repo/install/setup.bash
ros2 launch pinky_move lane_gazebo.launch.py \
  course:=two_lines gui:=true domain:=172 \
  output:=/tmp/pinky-lane-user-two-lines
```

`course:=left90`, `right90`, `s_bend`, `single_gap`로 코스를 바꾼다.
코스 변경은 기존 launch를 Ctrl+C로 종료한 다음 새 launch로 실행한다.
노드는 처음 `enabled=false`이며 가상 로봇도 자동 출발하지 않는다.

별도 터미널의 서비스/토픽 도구는 아래 환경을 사용해야 한다. launch 내부의 환경
설정은 다른 터미널에는 전달되지 않는다. 실제 domain20/22/52는 launch에서 거부한다.

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=172
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
ros2 service call /lane_sim/lane_autonomy/enable std_srvs/srv/SetBool '{data: true}'
# 가상 로봇만 정지
ros2 service call /lane_sim/lane_autonomy/enable std_srvs/srv/SetBool '{data: false}'
```

카메라/디버그 화면: rqt Image View의 `/lane_sim/camera/image_raw`, `/lane_sim/debug_image`.
상태: `/lane_sim/lane_autonomy/status`. 명령: `/lane_sim/cmd_vel`.
바퀴 기반 오도메트리: `/lane_sim/odom`, 물리 모델 ground truth: `/lane_sim/ground_truth`.
Gazebo도 `GZ_PARTITION=pinky_lane_sim_172`로 별도 분리된다.
이 환경 export는 시뮬레이션 터미널에만 적용하고 실제 로봇용 터미널에 복사하지 않는다.

## 2. 자동 주행·기록 시험

위 launch가 실행 중일 때:

```bash
cd /home/tory/pinky_nav_publish_yWXwM0/repo/pinky_move
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 tools/gazebo_lane_trial.py \
  --domain 172 --seconds 60 \
  --course-dir /tmp/pinky-lane-user-two-lines \
  --output reports/my_two_lines_trial
```

이 도구는 시뮬레이션 준비/카메라 K를 확인하고 가상 컨트롤러만 enable한다.
완주 또는 지정 시간이 지나면 disable하며 `trial.json`, 원본/디버그 JPG,
원본/디버그 MP4, 코스와 보정 JSON을 저장한다. 출력 폴더가 이미 있으면 거부하므로
매 시험마다 새 이름을 지정한다. 영상은 고정 15fps로 저장되어 재생 시간은 벽시계/시뮬
시간과 다를 수 있다. 정확한 이동·상태 시각은 `trial.json`을 기준으로 한다.

완주 판정은 ground truth가 목표 반경 10cm 이내에 도달했고, 측정된 로봇 중심이
차선 중심선에서 lane_width/2보다 더 벗어나지 않은 경우다. **중심점 기준**이라
로봇 footprint 전체가 차선 안이라는 보장은 아니다. 실패 시 종료 코드 1이며,
단지 로봇이 움직였다고 통과 처리하지 않는다. timeout은 벽시계 초다.

## 카메라와 물리 정확도

- Pinky URDF의 질량·관성·충돌·바퀴 반경(2.8cm)·바퀴 간격(9.61cm)을 재사용한다.
- URDF의 front_camera_link(8도 pitch)에서 optical 축 변환을 계산한다.
- Gazebo 카메라 640×480, fx=fy=566, cx=320, cy=240, 이상적인 0 distortion.
- 실제 카메라 calibration JSON은 수정하지 않는다. 시험 전 CameraInfo K가 생성 보정과
  1pixel tolerance로 맞는지 자동 확인한다. 이는 실제 렌즈/마운트 보정 검증이 아니다.
- 라이다·IR·램프는 이 차선 시뮬레이션 복사본에서 제외한다. 실차 URDF는 그대로다.
- YOLO가 가상 조명/바닥의 가로 코너를 검출하지 못하면 실제와 다른 도메인 차이가
  원인일 수 있다. 완벽한 정답 마스크로 바꿔 놓고 YOLO 주행 성공이라고 주장하지 않는다.

## 결과와 한계

### 급 S자와 실제 2차로(3라인)

차선 페인트는 선분별 box가 아니라 연속 miter-joined OBJ triangle mesh다.
각 코너의 양쪽 가장자리를 교점에서 연결하고 인접 삼각형이 동일 꼭짓점을 공유한다.
선 폭은 2cm이며 물리 충돌은 없다. `single_gap`의 의도적인 소실 구간만 분리한다.
형상 관련 33개 테스트 및 Gazebo 정지 카메라를 확인했다.
`reports/stripe_continuity_20261005/`의 평면도와 카메라 화면은 새 형상이고,
기존 주행 영상은 변경 전 형상이다. 새 형상에서 완주했다는 뜻은 아니다.

연속 mesh의 YOLO 단독 주행도 이후 별도 시험했다:
`left90_mesh_yolo_only_01` 70초 / 약 .601m 이동 / 코너 각도 88.3도,
`right90_mesh_yolo_only_01` 60초 / 약 .581m 이동 / 코너 각도 -89.7도,
`s_sharp_three_mesh_yolo_only_01` 60초 / 약 .472m 이동.
3회 모두 **FAIL**이며 차선 바깥으로 이탈해서가 아니라 진입 중 정지해 완주하지 못했다.
좌/우는 Lane=0, S자는 Lane=1이지만 `reacquisition requires the locked boundary side`
오류로 ENTRY_WAIT였다. 마지막 pose_timing은 모두 interpolated여서 이 시점의
odom 시간 연결 실패와는 구분된다. `mesh_trials_summary.json` 및 HTML의
‘새 연속 차선 시험’ 섹션에 새 영상과 원문 정지 이유를 보존했다.

`two_lines`는 이름 그대로 **경계선 2개 / 차로 1개**다.
차로 2개는 `lane_count:=2`로 실행하며 경계선 3개가 생성된다.
`lane_width`는 도로 전체가 아니라 **차로 하나의 폭**이다.
`target_lane:=0`은 오른쪽 차로, `target_lane:=1`은 왼쪽 차로다.
로봇은 선택한 차로 중심에서 시작하며 완주/이탈 판정도 그 차로를 기준으로 한다.
다른 차로로 변경하거나 3개 선의 전체 평균을 따라가는 기능은 아니다.

```bash
# 연속 직각 굴절: 좌→우→우→좌, 코너 사이 50/70cm
ros2 launch pinky_move lane_gazebo.launch.py course:=s_sharp lane_count:=2 target_lane:=0 gui:=true
```

`s_bend`의 완만한 곡선은 비교용으로 유지한다. `s_sharp`는 저장된 샘플처럼
극단적인 굴절을 시험하기 위한 별도 코스이며 사진의 실제 치수를 복원한 것은 아니다.
3라인 코스 생성 성공과 기존 컨트롤러의 3라인 경계 선택 성공은 별개다.

### 직각 실패 진단 및 제거된 보조 검출의 과거 기록

기존 좌/우 직각은 바퀴가 회전하다 실패한 것이 아니라 YOLO Lane=0에서 정지했다.
종료 프레임을 640 입력으로 재추론했을 때 좌측은 conf=.55/.30/.15/.05 모두 0개,
우측은 conf=.55에서 0개지만 conf=.30에서 점수 약 .436인 1개가 나온다.
따라서 confidence 완화만으로 좌우 모두 해결됐다고 할 수 없다.

흰 픽셀 보조 검출 코드·ROS 파라미터·launch 옵션·전용 테스트는 사용자 요청으로
먼저 제거했다. 이후 새 요청으로 Gazebo에만 별도의 `perception:=opencv` 프로파일을
추가했다. 실제 로봇과 Gazebo 기본 `perception:=yolo`는 YOLO 입력을 유지한다.
과거 `WHITE_PIXEL_AID` 시험 영상/로그는 비교 자료로 남겼으며 현재 버전의
완주 결과로 취급하지 않는다. 그 기능은 흰 픽셀을 HSV로 검출해 기존 기하/제어에
입력했으므로 **YOLO 단독 성공이 아니었다**. 밝은 물체 오검출 우려도 있었다.

물리 재시험 `left90_aid_01`은 약 1.870m 이동하고 최대 중심 이탈 3.99cm로 완주했다.
`right90_aid_01`은 약 1.887m 이동하고 최대 중심 이탈 4.11cm로 완주했다.
위 결과는 Gazebo의 중심점 기준이며 실차 또는 전체 footprint의 통과를 보장하지 않는다.

### 90도 판정과 실제 회전의 차이

`lane_corner.py:classify_boundary()`는 이미지 각도가 아니라 보정된 바닥 좌표에서
관측한 차선의 진입/진출 방향으로 판정한다.

1. 최소 5개 유효 점과 5cm 이상의 관측 길이를 요구하고 자기 교차를 검사한다.
2. 약 5cm 길이의 chord로 방향을 계산한다. 지속적인 반대 방향 굴절이면
   직각 하나보다 먼저 `S_BEND`로 분류한다.
3. 12cm 구간의 방향 변화 50도 이상 또는 전체 방향 변화 65도 이상일 때
   진입/진출 두 직선 후보를 검사한다.
4. 두 다리는 각각 최소 5cm, 직선 피팅 잔차는 각각 최대 8mm여야 한다.
   일반 CORNER 각도 범위는 50~130도다. 국소 변화가 50도 미만인 완만한
   굴절은 두 직선 사이 각도 65~115도인 경우에만 추가 후보로 인정한다.
5. 두 직선 교점이 진입 앞/진출 뒤에 있고 관측선에서 4cm 이내여야 한다.
   신뢰 가능한 두 다리가 없으면 `UNKNOWN`이다.

`CornerPolicy.update()`는 방향·위치·차선 정체성이 이어지는 관측을
기본 3프레임 확인한다. staged_turn=true일 때 실제 제자리 회전 방식의
각도 범위는 **75~115도**다(후보 검사에 쓰는 65~115도와 다르다).
확정 후 ENTRY에서 저장된 진입점까지 이동하고 BRAKE 후 PIVOT_TURN,
이후 진출 방향에 맞춰 차선을 재획득한다. 단순히 90도가 보였다고 즉시
회전하지 않는다. 진입점 오차·관측/odom 신선도·저장 경계 매칭 등도 검사한다.

일반 중앙 경로가 유효하면 Pure Pursuit로 달리면서 회전할 수도 있다.
이번 aid 재시험은 완주했지만 제자리 회전 상태를 거쳤다는 뜻은 아니다.
실제 상태 전이는 각 `trial.json`의 status 이력에서 확인한다.
두 aid 시험의 기록된 상태는 NORMAL(좌측 96회, 우측 89회)이었고
ENTRY/BRAKE/PIVOT_TURN은 0회였다. 즉 검출 보완 후 중앙 경로 추종으로
코너를 통과했으며, 이번 결과로 staged 90도 판정/제자리 회전까지 검증됐다고
주장하지 않는다.

### fresh odometry required 시간 비교 버그 수정

추론 결과 처리 시 `_corner_pose_for_frame(now)`에 전달한 now는 추론 작업자의
완료 시각이었다. 최신 odometry 콜백이 완료 뒤에 처리되면 `now - odom_received`
값이 음수가 된다. 기존의 0~0.3초 검사 때문에 신선한 odometry도 거부되는
콜백 순서 의존 버그를 합성 시계로 재현했다. 실차 로컬 추론에서도 가능한 오류다.
당시 옛 로그에는 거부 분기별 수치가 없어 과거 한 프레임의 직접 원인은 확정할 수 없다.

수정은 현재 steady clock으로 수신 신선도를 판단하고, 영상 촬영 timestamp로
odom 위치 보간을 유지하는 것이다. 0.3초 신선도, 0.3초 이하 보간 간격,
보간 불가능 시 최근접 50ms 기준은 완화하지 않았다. `pose_timing`에
거부 원인/처리 지연/odom 나이/보간 간격을 기록한다. 추론 완료 후 새 odom,
실제 odom 만료, 영상 시각 미포함 상황을 회귀 테스트로 검사한다.

`left90_odomfix_01`은 약 88도 후보를 정상 연결하고 CORNER_STAGED로
진입·회전했다. 약 0.732m 이동, 최종 yaw 약 77.75도였으며 회전 완료 후
`turn complete; waiting for exit lane`에서 멈춰 **완주 실패**였다.
동시 회귀 시험 실행 중 추론 입력 만료 정지도 기록됐다. 이번 시험은 시간 비교
수정과 staged 전환을 확인했으나 진출 차선 재획득/완주를 보증하지 않는다.

급 S자·3라인 `s_sharp_three_aid_02`는 약 0.393m 이동 후
`side association ambiguous`로 실패했다. 첫 `_01`은 CameraInfo 수신을
준비 조건에 포함하지 않았던 시험 도구의 시작 순서 문제로 enable 전에 실패했다.
현재 도구는 CameraInfo 수신도 기다린다. 두 실패 모두 삭제하지 않고 보존한다.

### 실차와 Gazebo 로직의 동일성

보조 제거 후 전체 회귀는 555 passed(105.51초, 스타일 검사 3종 제외)였다.
실제 `/home/tory/ws/pinky_pro`에는 odometry 시간 비교 수정과 원인 진단만 적용,
51개 관련 테스트 및 pinky_move 빌드를 통과하고 설치 모듈에도 반영됐음을 확인했다.
실제 로봇 실행 프로세스를 자동 재시작하거나 로봇에 SSH 배포/주행하지 않았다.
실행 중인 노드는 종료 후 다시 실행해야 변경 코드를 읽는다.
YOLO 단독 재시험 좌측은 약 .577m, 우측은 약 .610m 이동 후 정지해 실패했다.
현재 HTML 상단의 `_yolo_only_` 영상이 보조 제거 후 결과이며,
과거 aid 성공 영상은 참고용이다.

기본 `perception:=yolo`에서는 실차와 동일한 `lane_autonomy.py` →
`metric_lane.py` / `lane_corner.py`를 사용한다.
새 OpenCV 시뮬 프로파일은 `white_lane.py`의 연결 골격/법선/miter 경로와 기존
`_metric_command` Pure Pursuit를 사용하고 기존 staged-corner 거부 조건은 사용하지 않는다.
이 프로파일을 실제 로봇에 배포한 것은 아니다. 정답 경로 추종 컨트롤러도 아니다.
차이는 카메라/보정/odom/토픽, 차선폭(.20m vs 실차 사전값 .154m)이다.
기본 YOLO 모드에는 흰 픽셀 보조 검출이 없다. 새 OpenCV 및 과거 aid 완주를
실차 YOLO 단독 성능으로 취급하면 안 된다. 실제 보정 오차와 지연은 별도 검증해야 한다.

장기적인 YOLO 해결은 가로로 보이는 직각 입구·내측/외측 선·3라인·급 S자
데이터를 라벨링해 재학습하고, 현재 실패 프레임을 고정 회귀 셋으로 검사하는 것이다.
S자의 3조각 모호성은 segmentation만이 아니라 odometry로 옮긴 최근 경계와
관측된 인접 두 선을 매칭하여 선택한 차로의 정체성을 유지해야 한다.
검출 0개인데 방향을 새로 추정하거나 3개 선을 전부 평균내는 방식으로 해결하지 않는다.

실제 이 PC에서 실행한 원문은 `reports/gazebo_physics_20261005/**/trial.json`에 있다.
보고서 HTML의 성공/실패 표를 확인한다. 아래는 초기 YOLO 시험의 과거 기록이다.
첫 직선 시험은 코스 끝에서 1.81m에 정지했고,
끝점 뒤 차선을 연장한 재시험은 약 1.916m에서 목표 도달, 차선 중심 이탈 없이 통과했다.
좌측 직각 시험은 약 0.607m에서 YOLO Lane=0으로 멈춰 **실패**다.
우측 직각 시험은 약 0.580m에서 멈춰 **실패**다. 정지 화면 재추론에서 좌측 코너는
320/640 두 입력 크기 모두 confidence=.55 조건으로 검출이 없었다.
이런 실패도 재현 자료이며 실차 완주를 의미하지 않는다.

S자 60초 시험은 `multiple ambiguous boundaries on same side` / Lane=3으로
출발하지 못해 실패했다. 한쪽 선 소실 시험은 약 1.902m에서 목표에 도달하고 최대
중심 이탈 약 1.15mm로 통과했으며, 상태 로그에 단일 경계 추종 20회가 기록됐다.
초기 직선 실패 포함 6회 시험 중 2회 PASS. 코스별로는 5개 중 2개 PASS이며,
두 숫자를 혼동하지 않는다. 전체 motor-free 회귀 테스트는 531 passed(109.37초),
copyright/flake8/pep257 제외. 빌드는 pinky_description와 pinky_move 2개 성공(4.99초).

HTML: [물리 시험 결과·사진·영상](../reports/gazebo_physics_20261005/gazebo_physics_report.html).
영상과 사진은 렌더링된 가상 카메라이며 실제 Pinky 촬영이 아니다.

### MP4 재생 문제 해결

기존 OpenCV 녹화는 MPEG-4 Part 2(`mp4v`)였다. 현재 8개 영상은
H.264 baseline / yuv420p / faststart MP4로 변환했고, 해상도·프레임 수·길이
유지와 전체 프레임 디코딩을 확인했다(`video_validation.json`). 기존 원본은
각 시험 폴더의 `original_mp4v/`에 보존한다.
Firefox 157 headless에서도 HTML을 직접 열어 8개 재생기의 `play()` 성공,
재생 시각 증가, 640×480 프레임 및 MediaError 없음까지 확인했다
(`browser_playback_validation.json`).

HTML에는 영상 재생기가 포함되며 영상 데이터도 내장되어 있다. HTML 하나만
다운로드해 Firefox/Chrome 등 브라우저에서 열고 재생 버튼을 누르면 된다.
GitHub 파일 미리보기에서는 재생되지 않을 수 있으므로 로컬에서 연다.
MP4 다운로드 링크와 JSON 링크는 시험 폴더를 함께 유지해야 동작한다.

이후 시험은 로봇 disable 및 녹화 종료 뒤 자동 변환한다. `ffmpeg`와 `ffprobe`
(libx264 인코더 포함)가 필요하며, 변환 실패 시 원본을 유지하고
`trial.json`의 `summary.video_exports`에 오류를 남긴다. 기존 영상 수동 복구:

```bash
cd /home/tory/pinky_nav_publish_yWXwM0/repo/pinky_move
/usr/bin/python3 tools/repair_trial_videos.py
/usr/bin/python3 tools/build_gazebo_report.py
```

일부 ros_gz_bridge 버전은 launch SIGINT 종료 때 allocator 오류가 날 수 있다.
시험은 disable 응답 후 종료하며, launch가 자식 프로세스 종료를 기다릴 때까지 기다린다.
다른 로봇/관제 PID를 일괄 kill하지 않는다.

공식 참고: [Gazebo 차동구동](https://gazebosim.org/docs/harmonic/moving_robot/),
[ROS2 bridge](https://gazebosim.org/docs/harmonic/ros2_integration/).
