"""YOLO 승인 마스크 + 연결 경로 추종의 물리 시험 증거를 HTML로 묶는다."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from build_s_fix_report import asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='s_semantic_path_v2')
    parser.add_argument('--hardware-report', action='store_true')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    root = package/'reports/yolo_white_20261006'
    output = root/args.phase
    rows, sections, results = [], [], []
    files = sorted(output.glob('*/trial.json'))
    if args.hardware_report:
        recheck = root/'hardware_semantic_isolated_recheck/s_sharp_hybrid_lane0_r1/trial.json'
        if recheck.exists():
            files.append(recheck)
    for file in files:
        trial = json.loads(file.read_text())
        summary = trial['summary']
        run = file.parent.name
        display_run = run if file.parent.parent == output else '단독 재검증 / '+run
        before_file = root/'s_fix_final_v29'/run/'trial.json'
        old = json.loads(before_file.read_text())['summary'] if before_file.exists() else None
        white_file = package/'reports/gazebo_physics_20261005/opencv_final_suite_v2'/run.replace('_hybrid_', '_opencv_')/'trial.json'
        white = json.loads(white_file.read_text())['summary'] if white_file.exists() else None
        detection = dict(yolo_present=0, yolo_absent=0, waiting=0)
        for row in trial['observations']['status']:
            text = row['text']
            detection['waiting'] += text.startswith('WAITING_FOR_LANE')
            try:
                supplement = ast.literal_eval(text.split('supplement=',1)[1].split(', metric_target=',1)[0])
            except (IndexError, ValueError, SyntaxError):
                continue
            if isinstance(supplement, dict):
                detection['yolo_present' if supplement.get('yolo',0) else 'yolo_absent'] += 1
        results.append(dict(run=display_run, before=old, opencv=white, after=summary, detection=detection))
        rows.append(f'<tr><td>{display_run}</td><td>{"PASS" if white and white["passed"] else "미시험"}</td>'
                    f'<td>{"FAIL" if old and not old["passed"] else "—"}</td>'
                    f'<td>{"PASS" if summary["passed"] else "FAIL"}</td><td>{summary.get("travelled_m",0):.3f}m</td>'
                    f'<td>{summary.get("max_cross_track_m",0)*100:.2f}cm</td></tr>')
        before = ''
        if old is not None:
            before = ('<h3>BEFORE: 이전 버전 정지</h3><p>같은 코스의 이전 별도 실행입니다. 같은 프레임·촬영 위치가 아닙니다.</p>'+
                '<div class="compare"><div><h4>이전 디버그 화면</h4>'+asset(before_file.parent/'latest_debug.jpg')+
                '</div><div><h4>이번 디버그 화면</h4>'+asset(file.parent/'latest_debug.jpg')+'</div></div>'+
                '<h4>이전 정지 영상</h4>'+asset(before_file.parent/'debug_frames.mp4',True))
        sections.append('<section><h2>'+display_run+'</h2>'+before+'<h3>AFTER: 이번 카메라 영상</h3>'+asset(file.parent/'raw_frames.mp4',True)+
                        '<h3>제어 디버그</h3>'+asset(file.parent/'debug_frames.mp4',True)+
                        '<details><summary>원본 판정·검출 통계</summary><pre>'+html.escape(json.dumps(results[-1],ensure_ascii=False,indent=2))+'</pre></details></section>')
    tests = ET.parse(output/'regression.xml').getroot()
    counts = dict(tests=sum(int(s.get('tests',0)) for s in tests.iter('testsuite')),
                  failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in tests.iter('testsuite')))
    sources = ['lane_autonomy.py','white_lane.py','yolo_white.py','lane_planning_remote.py','lane_inference_worker.py']
    hashes = {name:hashlib.sha256((package/'pinky_move'/name).read_bytes()).hexdigest() for name in sources}
    passed = sum(r['after']['passed'] for r in results)
    deployment = output/'deployment.json'
    deployment = json.loads(deployment.read_text()) if deployment.exists() else {}
    summary = dict(passed=passed,total=len(results),tests=counts,results=results,source_sha256=hashes,deployment=deployment,
                   actual_robot_driven=False,actual_robot_deployed=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>YOLO S자 연결 경로 검증</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#101820;color:#e9eef3}table{border-collapse:collapse;width:100%}td,th{border:1px solid #556;padding:12px}video,img{width:100%;max-height:480px;object-fit:contain;background:black}section{margin-top:40px}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{background:#263440}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px}</style>
<h1>YOLO + 흰 픽셀 보충: S자 문제 분리·수정</h1>
<h2>OpenCV에서 되고 기존 YOLO에서 안 됐던 이유</h2>
<p>두 시험은 검출기만 다른 것이 아니었습니다. OpenCV 성공 시험은 골격/법선/miter 중앙 경로를 직접 추종하며 고정 코너 ENTRY_WAIT 대조를 우회했습니다.
기존 YOLO 보충 모드는 MetricLaneTracker와 CornerPolicy에서 저장 경계의 짧은 대응 구간, fold-back, 초기 좌우 배정 조건 때문에 중단됐습니다.</p>
<p>제어 설정도 달랐습니다: OpenCV lookahead 16cm, angular limit 0.6rad/s, 단일선 속도 상한 0.06m/s;
기존 hybrid 22cm, 0.15rad/s, 0.03m/s. 검출기 자체의 성능 차이라고만 해석하면 잘못된 비교입니다.</p>
<h2>이번 수정</h2><ol><li>simulation_semantic_path 기본값 false인 별도 비교 옵션을 추가했습니다. 실차 도메인20/22/52와 원격 프로파일에서는 실행할 수 없습니다.</li>
<li>YOLO로 확인하거나 유효 기간 내 flow로 이어받은 현재 마스크만 연결 경로의 입력으로 사용합니다. 흰 영상 전체에서 모델 승인 없이 새 선을 선택하지 않습니다.</li>
<li>연결 골격의 traversal을 정할 수 없는 횡방향 차선은 OpenCV 성공 모드와 동일한 행/열 바닥 투영 추출로 복구합니다. 가로 선은 이미지 끝점 높이 차가 작아 골격 전용 추출에서는 탈락했습니다.</li>
<li>선의 법선으로 차선폭 절반을 offset하고 곡선 vertex는 miter join합니다. 관측 순서/호길이를 유지하며 x 정렬하지 않습니다. Pure Pursuit와 근접 횡방향 목표의 회전 명령을 사용합니다.</li>
<li>YOLO·골격 계산은 inference worker에서 수행하고 제어 callback에는 경로만 전달합니다. 카메라/추론 freshness, semantic TTL, enable 서비스, 속도 한계는 유지합니다.</li></ol>
<p>첫 비교 초안(v1)은 골격만 사용해 약 0.448m에서 멈췄습니다. 화면에 횡방향 선이 있어도 골격 끝점 높이 조건 때문에 곡선이 비었고, 이후 YOLO 재확인도 끊겼습니다.
v2에서는 그 현재 마스크의 행/열 투영 fallback을 복원했습니다. 성공 결과는 v2만을 대상으로 하며 v1 실패 기록도 별도 폴더에 보존했습니다.</p>
<h2>물리 시험 결과</h2><p>Domain172, Gazebo 바퀴 물리, 카메라 640×480, 기존 best.pt/imgsz320/conf0.55. 급 S는 3개 선/2개 차로입니다.
판정은 목표 10cm 내 도착 및 중심선 오차≤10cm이며 충돌/전체 차체 여유에 대한 인증은 아닙니다. 한 번의 성공은 실차 보장이 아닙니다.</p>
<p>과거 OpenCV/기존 hybrid 결과와 이번 수정본은 제어 프로파일·시간 예산이 다르므로 순수 검출기 성능 비교가 아닙니다.
영상은 15fps 저장이며 디버그 영상은 추론 프레임만 모으므로 실제 경과 시간과 재생 시간이 다릅니다. 아래 원본 JSON의 로그 표본 통계는 프레임별 recall이 아닙니다.</p>
<table><tr><th>시험</th><th>과거 OpenCV</th><th>직전 hybrid</th><th>이번 수정</th><th>주행 길이</th><th>최대 중심 오차</th></tr>ROWS</table>
<h2>실행 방법</h2><p>실차에는 배포하지 않았습니다. 시험 클론에서 기존 방식은 기본값 그대로 유지됩니다. 새 비교 모드는 다음과 같이 명시적으로 켭니다.</p>
<pre>source /opt/ros/jazzy/setup.bash
source /home/tory/ws/pinky_pro/install/setup.bash
source /tmp/pinky-hybrid-install-20261006/setup.bash
ros2 launch pinky_move lane_gazebo.launch.py course:=s_sharp lane_count:=2 target_lane:=0 perception:=hybrid semantic_path:=true domain:=172
# 시뮬 카메라/모델 준비 확인 후, 같은 격리 DDS 설정의 터미널에서
ros2 service call /lane_sim/lane_autonomy/enable std_srvs/srv/SetBool "{data: true}"</pre>
<p>서비스 터미널은 ROS_DOMAIN_ID=172 및 launch의 loopback CYCLONEDDS_URI와 일치해야 합니다. 자동 검증 runner는 이를 직접 구성합니다.</p>
SECTIONS</html>'''
    page = page.replace('ROWS',''.join(rows)).replace('SECTIONS',''.join(sections))
    if args.hardware_report:
        hardware = '''<h2>실차용 브랜치 적용</h2>
<p>실차 알고리즘은 PC에서 계산합니다: YOLO segmentation → 모델 확인된 현재 흰 픽셀/flow 보충 → 보정된 base_link 좌표 → 연결 골격 또는 행/열 투영 → 법선/miter 중앙 경로 → lookahead.
Pinky2에서는 기존 freshness/odom/stop-line 검사 후 Pure Pursuit 또는 근접 횡방향 목표 회전을 /cmd_vel로 출력합니다. 새 transport나 Nav2를 추가하지 않았습니다.</p>
<p>semantic_lane_following는 노드 기본값 false, 이 브랜치의 Pinky2 런치 기본값 true입니다. enabled는 false로 시작합니다.
실제 intrinsic/URDF 보정 파일과 기존 차선폭 0.154m를 유지하며 시뮬 이상적 보정/0.20m 폭으로 덮지 않았습니다. 실제 보정의 mounting 검증 상태도 임의로 validated로 변경하지 않았습니다.</p>
<p>lookahead 0.16m, 관측 설정 0.08–0.70m, 기본 속도 0.10m/s, 단일선 상한 0.06m/s, angular 상한 0.60rad/s는 성공 비교 프로파일과 같습니다.
640 재추론은 새 모드에서 건너뛰어 320 현재 결과의 TTL을 보존합니다. semantic TTL 10초/flow frame gap 0.8초는 유지하며 모델 없이 흰 바닥으로 시작할 수 없습니다.</p>
<p>Gazebo 기록은 공통 경로 함수 검증이며 실차 원격 PC→로봇 경로는 모터 없는 protocol 테스트입니다. 실제 로봇 성공 영상으로 해석하면 안 됩니다.
이번 재시험에 실패가 있으면 기존 6/6 성공 기록으로 덮지 않습니다. 시뮬 성공은 실차 완주 보장이 아닙니다.</p>
<h3>Pinky2: 소스 업로드 후 필요한 빌드·실행</h3><pre>cd ~/pinky_pro
colcon build --packages-select pinky_move --symlink-install
source install/setup.bash
ROS_DOMAIN_ID=22 ros2 launch pinky_move lane_autonomy_pinky2.launch.py semantic_lane_following:=true</pre>
<h3>PC: 추론·경로 계산</h3><pre>cd ~/ws/pinky_pro
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch pinky_move lane_inference_pc.launch.py robot:=pinky@192.168.45.22</pre>
<p>양쪽 같은 버전의 패키지가 필요합니다. 로봇 카메라/추론/보정 확인 및 현장 감독 후 별도로 enable합니다. Nav2/teleop 등 다른 cmd_vel 발행자를 동시에 실행하지 마세요. 이 작업에서는 enable을 호출하지 않았습니다.</p>
<h3>모드 롤백</h3><pre># 기존 런치를 Ctrl+C로 종료하고 다시 시작
ROS_DOMAIN_ID=22 ros2 launch pinky_move lane_autonomy_pinky2.launch.py semantic_lane_following:=false</pre>
<p>모드 false는 기존 경로 처리·속도 설정을 선택합니다. 정확한 이전 파일 버전 복원은 로컬 패키지 전체 백업 또는 기준 커밋을 사용해야 합니다. 패키지 전체 재빌드 대신 pinky_move만 빌드하면 됩니다.</p>
<details><summary>적용/백업 상태</summary><pre>DEPLOYMENT</pre></details>'''
        hardware = hardware.replace('DEPLOYMENT',html.escape(json.dumps(deployment,ensure_ascii=False,indent=2)))
        # 이전 성공을 최신 수정본의 판정으로 섞지 않는다. 동일 코스의
        # 명확한 성공 Before/After 자료로만 별도 제시한다.
        success = '<h2>이전 검증 성공의 비포·애프터: 기준 커밋 6691f6e</h2><p>이 자료는 앞선 6/6 시험 중 S자 성공 영상입니다. 위 최신 재시험 판정과는 별도입니다.</p>'
        for lane in (0,1):
            run = f's_sharp_hybrid_lane{lane}_r1'
            directory = root/'s_semantic_path_v2'/run
            old_directory = root/'s_fix_final_v29'/run
            if not directory.exists():
                continue
            success += ('<section><h2>성공 Before / After · '+run+'</h2><div class="compare"><div><h3>BEFORE · ENTRY_WAIT 정지</h3>'+
                asset(old_directory/'latest_debug.jpg')+'</div><div><h3>AFTER · 완주</h3>'+asset(directory/'latest_debug.jpg')+
                '</div></div><h3>AFTER 원본 카메라</h3>'+asset(directory/'raw_frames.mp4',True)+
                '<h3>AFTER 디버그</h3>'+asset(directory/'debug_frames.mp4',True)+'</section>')
        hardware += success
        page = page.replace('SECTIONS', '') if 'SECTIONS' in page else page
        page = page.replace('</html>',hardware+'</html>')
    page = page.replace('<h2>물리 시험 결과</h2>',f'<h2>물리 시험 결과: {passed}/{len(results)} PASS</h2><p>회귀 테스트 {counts["tests"]}개, 실패 {counts["failures"]}개. 실차 배포 없음.</p>')
    (output/'report.html').write_text(page)
    print(json.dumps(dict(passed=passed,total=len(results),report=str(output/'report.html'))))


if __name__ == '__main__':
    main()
