# Gazebo 카메라 차선 물리 주행 시험

Nav2 없이 **가상 카메라 → 실제 기존 YOLO segmentation 모델 → 기존
`lane_autonomy` → 가상 `/cmd_vel` → Gazebo 바퀴·접촉 물리**를 연결한다.
저장 사진 반복이나 목표점 숫자 시뮬레이션이 아니라 URDF 물리 모델이 실제로 움직인다.
실제 로봇에 SSH/서비스/속도 명령을 보내지 않는다.

## 코스

| course | 내용 |
|---|---|
| `two_lines` | 평행한 흰 선 두 개, 2m 직선, 두 경계 중앙 추종 |
| `left90` | 0.75m 앞 좌측 직각, 이후 1.25m 직선 |
| `right90` | 좌측 코스의 좌우 반전 |
| `s_bend` | 양쪽 흰 선이 있는 연속 S자, x=2.1m까지 |
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

실제 이 PC에서 실행한 원문은 `reports/gazebo_physics_20261005/*/trial.json`에 있다.
보고서 HTML의 성공/실패 표를 확인한다. 첫 직선 시험은 코스 끝에서 1.81m에 정지했고,
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

일부 ros_gz_bridge 버전은 launch SIGINT 종료 때 allocator 오류가 날 수 있다.
시험은 disable 응답 후 종료하며, launch가 자식 프로세스 종료를 기다릴 때까지 기다린다.
다른 로봇/관제 PID를 일괄 kill하지 않는다.

공식 참고: [Gazebo 차동구동](https://gazebosim.org/docs/harmonic/moving_robot/),
[ROS2 bridge](https://gazebosim.org/docs/harmonic/ros2_integration/).
