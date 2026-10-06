# Pinky Pro 책상 위 좌우 왕복 주행

> 웹 워크페이지: [docs/table_patrol_workpage.html](docs/table_patrol_workpage.html)

pinky_move는 기존 모터 드라이버를 수정하지 않고 표준 ROS 2 속도 명령인
geometry_msgs/msg/Twist를 /cmd_vel로 발행하는 응용 패키지입니다.

Pinky Pro는 앞뒤 이동과 회전은 가능하지만 바퀴 구조상 몸체를 옆으로
미끄러뜨릴 수는 없습니다. 여기서 “좌우 왕복”은 다음 동작을 뜻합니다.

1. 직선으로 천천히 이동합니다.
2. 설정한 끝점에서 정지합니다.
3. 제자리에서 180도 회전합니다.
4. 반대쪽 끝까지 이동하고 같은 동작을 반복합니다.

## 왜 별도 패키지로 만들었나

pinky_bringup은 모터, 엔코더, 배터리처럼 하드웨어를 담당합니다. 이 파일을
직접 수정하면 나중에 다른 주행 기능과 섞이고 고장 원인을 찾기 어렵습니다.
pinky_move는 “언제 어느 속도로 움직일지”만 판단하고 /cmd_vel 토픽을
발행합니다. 하드웨어와 행동 로직이 분리되어 시뮬레이션과 실제 로봇에서
같은 제어 노드를 재사용할 수 있습니다.

흐름은 다음과 같습니다.

    Gazebo 내부 Range 3개 ─> ir_adc_sim ─┐
                                         ├─> /ir_sensor/range ─┐
    실물 ADC 드라이버 ───────────────────┘                       │
    카메라 /camera/image_raw ────────────────────────────────────┼─> pinky_move ─> /cmd_vel
    라이다 /scan ────────────────────────────────────────────────┤
    위치 /odom ──────────────────────────────────────────────────┘

## 안전 구조

- IR: 좌·중앙·우 센서 중 하나라도 책상면을 놓치거나 0.3초간 끊기면 정지합니다.
- 카메라: 활성화한 경우 화면 아래쪽에 설정한 책상 색이 부족하면 정지합니다.
- 라이다: 진행 방향에 장애물이 있거나 회전 공간이 좁으면 즉시 정지합니다.
- 오도메트리: 실제 이동 거리를 재서 설정 구간 밖으로 계속 가지 않게 합니다.
- 통신 타임아웃: 필요한 센서 메시지가 0.7초 이상 끊겨도 즉시 정지합니다.
- 모터 watchdog: 실제 모터 드라이버는 /cmd_vel이 0.5초 끊기면 RPM을 0으로 만듭니다.
- 수동 허가: 노드는 항상 enabled=false로 시작하며 서비스 호출 전에는 움직이지
  않습니다.
- 종료 정지: Ctrl+C로 종료할 때 정지 명령을 여러 번 발행합니다.

IR과 카메라를 함께 사용해도 실제 낙상을 완전히 보장할 수는 없습니다.
특히 반사형 IR은 표면의 색·광택·외부광에 영향을 받으므로 실측 보정이 필요합니다.
첫 시험은 바닥에서 하십시오. 책상 첫 시험은 사람이 로봇 바로 옆에서 잡을
준비를 하고 최저 속도로 진행해야 합니다. 실제 patrol_length는 책상 길이에서
양쪽 안전 여백을 뺀 값보다 작아야 합니다.

## 빌드

    cd ~/ws/pinky_pro
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install --packages-up-to pinky_move
    source install/setup.bash

새 터미널을 열 때마다 아래 두 줄을 다시 실행해야 합니다.

    source /opt/ros/jazzy/setup.bash
    source ~/ws/pinky_pro/install/setup.bash

source는 ROS 2와 방금 빌드한 패키지의 위치를 현재 터미널에 알려 주는
과정입니다.

## Gazebo에서 먼저 실행하기

전체 시뮬레이션은 한 명령으로 실행합니다.

    ros2 launch pinky_move table_patrol_sim.launch.py

이 명령은 다음 작업을 함께 합니다.

- 0.70 × 0.60m 테이블 네 개를 X축으로 일렬 연결한 2.80 × 0.60m 월드를 엽니다.
- 로봇을 테이블 중앙의 z=0.90m 위치에 놓습니다.
- 카메라 각도는 실물과 맞추기 위해 0도를 유지합니다.
- 내부적으로 /ir/left, /ir/center, /ir/right Range 브리지를 준비합니다.
- ir_adc_sim이 이를 실물과 같은 /ir_sensor/range ADC 배열로 바꿉니다.
- /camera/image_raw, /scan, /odom, /cmd_vel 브리지를 준비합니다.
- IR을 주 책상 끝 센서로 사용하고 pinky_move를 enabled=false로 실행합니다.

Gazebo가 완전히 뜬 후 새 터미널에서 토픽을 확인합니다.

    ros2 topic list
    ros2 topic echo /ir_sensor/range --once
    ros2 topic echo /pinky_move/ir_safe --once
    ros2 topic hz /scan
    ros2 topic hz /odom

정상 책상 중앙에서는 세 ADC 모사값이 약 3200이고 ir_safe가 true여야 합니다.
/ir/left 등은 ADC 모사 노드 내부 동작을 확인할 때만 보는 거리 토픽입니다.
상태에 센서 누락 문구가 없으면 다른 터미널에서 이동을 허용합니다.

    ros2 service call /pinky_move/enable std_srvs/srv/SetBool "{data: true}"

즉시 멈추려면 다음 명령을 사용합니다.

    ros2 service call /pinky_move/enable std_srvs/srv/SetBool "{data: false}"

기본 월드에서는 전체 상판 길이 2.80m 중 양쪽 0.40m씩을 남기고 2.00m를
왕복합니다. 중앙에서 시작하므로 첫 구간은 절반인 1.00m이고, 180도 회전한
뒤부터 매 구간 2.00m를 이동합니다.

월드만 따로 열고 싶을 때는 다음 명령을 사용합니다.

    ros2 launch pinky_gz_sim four_tables_sim.launch.xml

제어기만 따로 실행하려면 다음과 같이 시뮬레이션 시간을 켭니다.

    ros2 launch pinky_move table_patrol.launch.py use_sim_time:=true

## 실제 로봇에서 실행하기

URDF에는 앞쪽 바닥에 ir_l_link, ir_mid_link, ir_r_link가 이미 정의되어 있습니다.
통합 launch는 기존 pinky_sensor_adc를 실행해 /dev/i2c-1에서 세 ADC 값을 읽습니다.
모터·IR·라이다·USB 카메라·안전 제어기를 한 번에 실행하려면 다음 명령을 사용합니다.
노드는 여전히 enabled=false로 시작하므로 이 명령만으로 로봇이 움직이지 않습니다.

    ros2 launch pinky_move table_patrol_robot.launch.py

카메라가 /dev/video1이면 장치를 지정합니다. 기본 픽셀 형식이 지원되지 않을 때는
pixel_format:=yuyv2rgb도 함께 시험할 수 있습니다.

    ros2 launch pinky_move table_patrol_robot.launch.py camera_device:=/dev/video1

카메라 드라이버를 다른 프로그램이 이미 실행 중이라면 중복 실행하지 마십시오. 이때는
pinky_bringup과 제어기만 각각 실행합니다.

    ros2 launch pinky_bringup bringup_robot.launch.xml
    ros2 run pinky_sensor_adc main_node
    ros2 launch pinky_move table_patrol.launch.py

새 터미널에서 아래 토픽이 모두 계속 들어오는지 확인합니다.

    ros2 topic hz /ir_sensor/range
    ros2 topic echo /ir_sensor/range
    ros2 topic hz /camera/image_raw
    ros2 topic hz /scan
    ros2 topic hz /odom
    ros2 topic echo /pinky_move/status
    ros2 topic echo /pinky_move/ir_safe

실물 기본 설정은 ir_adc_calibrated=false라 보정 전에는 움직이지 않습니다.
아래 IR 보정 절차를 마치고 ir_safe=true를 확인한 뒤에만 이동을 허용합니다.

    ros2 service call /pinky_move/enable std_srvs/srv/SetBool "{data: true}"

다른 터미널에서 teleop_twist_keyboard나 Nav2처럼 /cmd_vel을 발행하는
프로그램을 동시에 실행하지 마십시오. 여러 발행자의 속도 명령이 서로
덮어쓰면 예측할 수 없는 동작이 생깁니다.

## 설정 값 바꾸기

실제 로봇 설정은 config/table_patrol.yaml이고, Gazebo 전용 설정은
config/table_patrol_sim.yaml입니다. 실제 설정의 기본 왕복 거리는 안전한 바닥 첫 시험을
위해 0.30m이며, 시뮬레이션은 안전 여백을 늘린 2.00m를 사용합니다.

- patrol_length: 양 끝점 사이 전체 거리(m)입니다.
- start_at_center: true면 첫 이동만 patrol_length의 절반입니다.
- linear_speed: 직선 속도(m/s)입니다. 실제 첫 시험은 0.04를 유지하십시오.
- linear_acceleration: 직선 속도가 변하는 빠르기(m/s²)입니다.
- angular_speed: 회전 최고 속도(rad/s)입니다.
- minimum_turn_speed: 목표각 근처에서도 유지할 최소 회전속도입니다. 너무 크면 지터가 생길 수 있습니다.
- turn_kp: 남은 각도에 비례해 회전속도를 정하는 비례 게인입니다.
- drive_heading_kp: 직진 중 방향 오차를 고치는 세기입니다.
- drive_max_angular_speed: 직진 중 허용할 최대 보정 각속도입니다.
- cross_track_kp: 원래 중앙선에서 벗어났을 때 돌아오는 세기입니다.
- pause_seconds: 끝점에서 회전 전 정지 시간입니다.
- use_ir_guard: 세 IR 채널을 필수 안전 조건으로 사용합니다.
- ir_adc_calibrated: 실물 보정 완료 전에는 false이며 주행을 잠급니다.
- ir_adc_min/max_values: 실물 세 채널의 책상 위 ADC 허용 범위입니다.
- ir_timeout: IR 토픽이 끊겼다고 판단하는 시간입니다.
- surface_ratio_threshold: 카메라 보조 사용 시 책상색의 최소 비율입니다.
- surface_h_min/max: OpenCV HSV의 색상 범위입니다.
- surface_s_min: 색의 선명도 최솟값입니다.
- surface_v_min/max: 밝기 범위입니다.
- front_stop_distance: 전방 장애물 정지 거리(m)입니다.
- turn_stop_distance: 제자리 회전에 필요한 최소 주변 거리(m)입니다.
- sensor_timeout: 센서 메시지를 기다리는 최대 시간(s)입니다.
Gazebo의 ADC 모사 노드에는 별도의 파라미터가 있습니다.

- tabletop_max_distance: 내부 Range가 책상이라고 인정되는 최대 거리입니다.
- surface_adc_values: 책상 위일 때 만들어 낼 세 ADC 기준값입니다.
- off_surface_adc_values: 책상 밖일 때 만들어 낼 세 ADC 기준값입니다.
- adc_noise_stddev: 실물 ADC처럼 조금 흔들리는 잡음의 표준편차입니다.

### 실행 중 속도를 동적으로 바꾸기

pinky_move는 제어 주기마다 속도 파라미터를 다시 읽으므로 실행 중에도 바뀝니다.
안전을 위해 먼저 주행을 비활성화하고 값을 바꾸십시오.

    ros2 service call /pinky_move/enable std_srvs/srv/SetBool "{data: false}"
    ros2 param set /pinky_move linear_speed 0.03
    ros2 param set /pinky_move linear_acceleration 0.03
    ros2 param set /pinky_move angular_speed 0.30

현재값은 다음처럼 확인합니다.

    ros2 param get /pinky_move linear_speed
    ros2 param list /pinky_move

RQt에서는 Plugins > Configuration > Dynamic Reconfigure에서 /pinky_move를
선택합니다. 실행 중 변경값은 노드를 다시 시작하면 YAML 값으로 돌아갑니다.

### 반복 후 책상 밖으로 빠지던 원인과 수정

이전 코드는 매 회전 목표를 현재 yaw + 180도로 만들었습니다. 회전 허용오차가
4도이면 끝난 각도 오차가 다음 직선 방향에 그대로 들어가고, 직진 중에는 각도를
보정하지 않았습니다. 2m를 4도 기울어 이동하면 이론상 약 14cm 옆으로 이동하므로
여러 왕복 뒤 폭 0.60m 책상의 옆 가장자리에 가까워질 수 있습니다.

- 회전 목표를 최초 yaw와 최초 yaw + 180도라는 두 절대 방향으로 고정했습니다.
- 직진 중 heading 오차와 원래 중앙선으로부터의 cross-track 오차를 보정합니다.
- 옆으로 미끄러진 거리는 전진 완료 거리로 세지 않습니다.
- Gazebo 왕복 길이를 2.00m로 줄여 양 끝에 0.40m 여유를 남겼습니다.

진단용 0.20m/s로 75초간 회전 4회를 계측했을 때 회전 명령의 부호 반전은
0회, 최대 Y 이탈은 약 1.5mm, IR false와 안전정지는 0회였습니다. 책상 끝으로
강제 이동한 시험에서는 ADC가 약 [100, 116, 90]으로 내려가고
STOPPED: one or more IR sensors do not see tabletop으로 주행이 차단됐습니다.

파일을 바꾼 뒤에는 실행 중인 노드를 Ctrl+C로 끝내고 다시 실행하십시오.
--symlink-install로 빌드했기 때문에 Python 코드와 설정 수정은 보통 전체 재빌드
없이 반영되지만, setup.py나 새 파일을 추가했다면 다시 빌드해야 합니다.

## 실제 IR ADC 보정

실물 드라이버는 /ir_sensor/range에 [adc_result[2], adc_result[1],
adc_result[0]] 순서의 0~4095 값을 발행합니다. 배선의 좌·중앙·우 이름은
드라이버 코드에 기록되어 있지 않지만 세 채널을 모두 검사하므로 안전 판정에는
순서가 중요하지 않습니다.

1. 모터를 비활성화하고 책상과 같은 재질의 넓은 판 중앙에서 세 값을 20초 기록합니다.
2. 사람이 로봇을 든 상태에서 센서를 하나씩 가려 배열의 각 채널을 확인합니다.
3. 책상 위 값과 책상 밖 값이 겹치지 않는지 확인합니다.
4. table_patrol.yaml의 ir_adc_min_values와 ir_adc_max_values를 채널 순서대로 씁니다.
5. 바닥 시험에서 센서 하나가 범위를 벗어날 때 ir_safe=false와 STOPPED를 확인합니다.
6. 모든 검증 후에만 ir_adc_calibrated를 true로 변경합니다.

다른 로봇이나 다른 책상의 임계값을 복사하지 마십시오. 두 범위가 겹치면
IR을 단독 낙상 방지 수단으로 사용해서는 안 됩니다.

## 실제 책상색 보정

기본 HSV 값은 갈색 시뮬레이션 테이블용입니다. 실제 책상이 흰색이나 검정색이면
기본 설정에서 움직이지 않는 것이 정상이고 안전한 동작입니다.

1. 로봇을 움직이지 않은 상태로 카메라를 책상 쪽으로 기울입니다.
2. rqt_image_view를 실행해 /camera/image_raw를 선택합니다.
3. /pinky_move/surface_ratio를 보면서 HSV 범위를 조정합니다.
4. 로봇 중심과 안전한 위치에서 비율이 임계값보다 충분히 높아야 합니다.
5. 책상 가장자리를 카메라가 보게 했을 때는 비율이 임계값 아래로 내려가야 합니다.
6. 이 두 조건을 모두 만족한 후에만 실제 이동을 허용합니다.

바닥에서 거리 왕복 로직만 시험할 때 use_surface_guard를 false로 바꿀 수 있지만,
책상 위에서는 이 보호 기능을 끄지 마십시오.

## 상태 메시지 뜻

- DISABLED: 사용자가 이동을 허용하지 않은 정상 대기 상태입니다.
- WAITING: 허용 직후 센서들을 확인하는 중입니다.
- DRIVING: 직선 구간을 이동 중이며 현재/목표 거리가 함께 표시됩니다.
- PAUSING: 끝점에서 정지 중입니다.
- TURNING: 180도 회전 중이며 남은 각도가 표시됩니다.
- STOPPED: IR, 카메라, 라이다, 오도메트리 또는 안전 조건 때문에 정지했습니다.

STOPPED가 나오면 억지로 보호 기능을 끄지 말고 뒤에 표시된 원인을 먼저
해결하십시오.
