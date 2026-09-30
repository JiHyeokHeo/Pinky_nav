# Domain 52 관제 통합 실행

로봇 1(ROS Domain 20)과 로봇 2(ROS Domain 22)의 하드웨어 및 Nav2는
각 로봇에서 먼저 실행한다. 아래 launch는 **관제 PC에서만** 실행한다.

```bash
cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
colcon build --packages-select pinky_multi_robot --symlink-install
source install/setup.bash
ros2 launch ~/ws/pinky_pro/pinky_multi_robot/launch/central_control.launch.py
```

이 launch 하나가 AMCL pose 브리지, 로봇별 위치 확인 노드, 위치 확인 연동
NavigateToPose 액션 프록시, 미션 서버, PyQt 관제 UI를 실행한다. UI 없이 실행하려면
`show_ui:=false`를 뒤에 붙인다. 기존 `nav2_action_proxy`나 각 관제 프로세스를
별도로 동시에 실행하지 않는다.

위치 확인 노드는 각 로봇 도메인에 접속해 AMCL 전역 위치 찾기를 한 번 요청한다.
시작만으로 로봇이 움직이지 않는다. PyQt에서 로봇 한 대를 선택한 뒤
`Initialize location (360°)`를 누르고 확인하면, 해당 로봇의 AMCL을 재설정하고
Nav2 `spin` 액션으로 90°씩 네 번 회전한다. 회전 전 주변 장애물을 치우고
감독해야 한다. `Cancel init spin`은 선택한 로봇의 회전을 취소한다.
현재 주행 중인 로봇은 초기화를 거부하며, `Both` 선택 시에도 시작할 수 없다.
`/scan`은 2초 안의 최신 메시지가 필요하고, `/amcl_pose` 대기 시간은 30초다.
AMCL 공분산이 여러 번 기준 이하일 때만 `READY`가 된다. 회전이 완료되어도
위치가 확정되지 않으면 관제 목표는 계속 차단된다.
확인되지 않은 로봇의 관제 NavigateToPose 목표는 프록시에서 중단된다.

회전으로 위치를 못 찾으면 UI에서 로봇 한 대와 Heading을 선택한 뒤
`Set current pose: next click`을 누르고 지도의 실제 로봇 위치를 클릭한다.
해당 로봇 AMCL의 `/initialpose`에 추정 위치를 보낼 뿐, 로봇은 움직이지 않는다.
`READY`와 `map match`를 확인하고 실제 위치/방향을 눈으로 대조한 뒤 주행한다.
지도 정합도는 라이다 끝점과 정적 지도 장애물의 일치율을 보여 주는 참고 신호다.

로봇별 목표를 동시에 보내려면 `pinky1`의 `Set goal (stage): next click`으로
목표를 저장하고, `pinky2`도 같은 방식으로 저장한다. 각 로봇 task를
`Staged goal`로 선택하고 `Send combined mission`을 누르면 한 미션으로 전송된다.
`Saved parking`과 `Patrol`도 로봇별로 섞을 수 있다. 일반 지도 클릭과
`Go to saved parking`/`Start patrol` 버튼은 즉시 전송이다.

실제 로봇 도메인과 분리된 액션 수준 시뮬레이션 테스트:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
python3 ~/ws/pinky_pro/pinky_multi_robot/test/mission_sim_smoke.py
```

테스트는 임시 ROS Domain 150–199에서 가짜 Nav2 액션 서버 두 개를 띄워
동시 시작, 미션 완료, 중복 미션 거부를 확인한다. Gazebo 물리 주행이나 실제
지도/라이다 정합까지 검증하지는 않는다.

상태 확인:

```bash
ROS_DOMAIN_ID=52 ros2 topic echo --once /pinky1/localization_status --no-daemon
ROS_DOMAIN_ID=52 ros2 topic echo --once /pinky2/localization_status --no-daemon
ROS_DOMAIN_ID=52 ros2 topic echo --once /pinky2/amcl_pose --no-daemon
```

`NEEDS_MANUAL_POSE`나 `SEARCHING`이 계속되면 로봇별 Nav2 웹 화면의
2D Pose Estimate(기존 `/api/initialpose`)로 대략적인 위치와 방향을 지정한다.
라이다 스캔과 AMCL 추정이 수렴해야 `READY`가 된다. 같은 모양의 복도가 반복되면
낮은 공분산만으로도 잘못된 위치에 수렴할 수 있으므로, 첫 주행 전에는 관제
지도에 표시된 위치를 실제 위치와 눈으로 대조하고 저속 감독 시험을 한다.

이 코드는 중앙 관제 액션 경로만 차단한다. 각 로봇의 Nav2 액션 서버에 직접
보내는 목표나 별도 `cmd_vel` 발행자는 막지 않는다. 실제 긴급 정지는 로봇의
기존 안전 장치로 수행해야 한다.
