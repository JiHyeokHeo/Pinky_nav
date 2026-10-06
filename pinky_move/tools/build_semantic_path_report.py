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
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    root = package/'reports/yolo_white_20261006'
    output = root/args.phase
    rows, sections, results = [], [], []
    for file in sorted(output.glob('*/trial.json')):
        trial = json.loads(file.read_text())
        summary = trial['summary']
        run = file.parent.name
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
        results.append(dict(run=run, before=old, opencv=white, after=summary, detection=detection))
        rows.append(f'<tr><td>{run}</td><td>{"PASS" if white and white["passed"] else "미시험"}</td>'
                    f'<td>{"FAIL" if old and not old["passed"] else "—"}</td>'
                    f'<td>{"PASS" if summary["passed"] else "FAIL"}</td><td>{summary.get("travelled_m",0):.3f}m</td>'
                    f'<td>{summary.get("max_cross_track_m",0)*100:.2f}cm</td></tr>')
        sections.append('<section><h2>'+run+'</h2><h3>카메라</h3>'+asset(file.parent/'raw_frames.mp4',True)+
                        '<h3>제어 디버그</h3>'+asset(file.parent/'debug_frames.mp4',True)+
                        '<details><summary>원본 판정·검출 통계</summary><pre>'+html.escape(json.dumps(results[-1],ensure_ascii=False,indent=2))+'</pre></details></section>')
    tests = ET.parse(output/'regression.xml').getroot()
    counts = dict(tests=sum(int(s.get('tests',0)) for s in tests.iter('testsuite')),
                  failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in tests.iter('testsuite')))
    sources = ['lane_autonomy.py','white_lane.py','yolo_white.py']
    hashes = {name:hashlib.sha256((package/'pinky_move'/name).read_bytes()).hexdigest() for name in sources}
    passed = sum(r['after']['passed'] for r in results)
    summary = dict(passed=passed,total=len(results),tests=counts,results=results,source_sha256=hashes,
                   actual_robot_driven=False,actual_robot_deployed=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>YOLO S자 연결 경로 검증</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#101820;color:#e9eef3}table{border-collapse:collapse;width:100%}td,th{border:1px solid #556;padding:12px}video{width:100%;max-height:480px;background:black}section{margin-top:40px}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{background:#263440}</style>
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
    page = page.replace('<h2>물리 시험 결과</h2>',f'<h2>물리 시험 결과: {passed}/{len(results)} PASS</h2><p>회귀 테스트 {counts["tests"]}개, 실패 {counts["failures"]}개. 실차 배포 없음.</p>')
    (output/'report.html').write_text(page)
    print(json.dumps(dict(passed=passed,total=len(results),report=str(output/'report.html'))))


if __name__ == '__main__':
    main()
