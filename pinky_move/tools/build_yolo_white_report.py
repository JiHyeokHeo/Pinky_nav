"""YOLO + 흰 픽셀 보충의 실제 Gazebo 시험과 영상 보고서를 생성한다."""
import ast
import base64
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def asset(path, mime, video=False):
    if not path.exists():
        return '<p>자료 없음</p>'
    src = 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
    return (f'<video controls preload="metadata" playsinline src="{src}"></video>' if video else
            f'<img alt="가상 카메라 debug" src="{src}">')


def main():
    root = Path(__file__).resolve().parents[1]/'reports/yolo_white_20261006'
    results, sections, table = [], [], []
    for path in sorted((root/'staged').glob('*/trial.json')):
        data = json.loads(path.read_text())
        summary = data['summary']
        modes, supplement, stop_reasons = Counter(), [], set()
        for sample in data['observations']['status']:
            text = sample['text']
            if 'supplement={' in text:
                supplement.append(ast.literal_eval(text.split('supplement=', 1)[1].split(', metric_target=', 1)[0]))
            if 'corner={' in text:
                modes[ast.literal_eval(text.split('corner=', 1)[1]).get('mode')] += 1
            if text.startswith('SAFETY_STOP'):
                stop_reasons.add(text.split(';')[0])
        row = dict(run=path.parent.name, **summary, corner_modes=dict(modes),
            supplementation_status_samples=len(supplement),
            white_assisted_status_samples=sum(s['white_current']+s['white_tracked'] > 0 for s in supplement),
            yolo_absent_white_tracking_status_samples=sum(s['yolo'] == 0 and s['white_tracked'] > 0 for s in supplement),
            stop_reasons=sorted(stop_reasons))
        results.append(row)
        table.append(f'<tr><td>{row["run"]}</td><td>{"PASS" if row["passed"] else "FAIL"}</td>'
            f'<td>{row["travelled_m"]:.3f}m</td><td>{row["max_cross_track_m"]*100:.2f}cm</td>'
            f'<td>{row["yolo_absent_white_tracking_status_samples"]} / {len(supplement)}</td></tr>')
        sections.append('<section><h2>'+row['run']+'</h2>'+asset(path.parent/'latest_debug.jpg', 'image/jpeg')+
            '<h3>원본 카메라</h3>'+
            asset(path.parent/'raw_frames.mp4', 'video/mp4', True)+'<h3>실제 제어 debug</h3>'+
            asset(path.parent/'debug_frames.mp4', 'video/mp4', True)+
            '<details><summary>판정·차선 보충·상태 모드</summary><pre>'+html.escape(json.dumps(
                {k: row[k] for k in ('passed', 'corner_modes', 'stop_reasons',
                 'white_assisted_status_samples', 'yolo_absent_white_tracking_status_samples')},
                ensure_ascii=False, indent=2))+'</pre></details></section>')
    passed = sum(r['passed'] for r in results)
    (root/'summary.json').write_text(json.dumps(dict(passed=passed, runs=len(results), results=results,
        actual_robot_driven=False, pixel_only=False, controller='existing MetricLaneTracker/CornerPolicy/PurePursuit'),
        ensure_ascii=False, indent=2))
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO + 흰 픽셀 보충 Gazebo 주행</title><style>body{font:16px/1.65 system-ui;margin:0;background:#edf1f5;color:#182536}main{max-width:1050px;margin:auto;padding:24px}header,section{background:white;padding:24px;margin:0 0 20px;border-radius:12px}img,video{width:100%;max-width:750px;display:block;margin:12px auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warn{background:#fff0c7;padding:14px}</style><main>
<header><h1>YOLO + 흰 픽셀 보충<br>Gazebo 폐루프 주행 시험</h1><p>2026-10-06 · experiment/yolo-white-supplement-20261006 · 실제 로봇 주행/배포 없음</p></header>
<section><h2>보충 방식</h2><p>YOLO를 매 프레임 계속 실행합니다. 현재 YOLO 마스크와 겹치는 흰 연결 성분으로 같은 차선의 누락 부분을 보충합니다. YOLO가 잠시 없으면 optical flow로 이전 선택 성분과 현재 흰 성분을 대응합니다. 과거 픽셀을 그대로 재발행하지 않습니다.</p><p>마지막 YOLO 확인은 최대 10초, 프레임 간격은 최대 0.8초입니다. YOLO로 확인되지 않은 흰 성분은 차선으로 새로 선택하지 않습니다. 영상의 18%를 넘는 넓은 흰 영역은 보충에서 거부합니다. 흰 바닥 전체는 사용할 수 없습니다.</p><p>흰 픽셀 전용 우회 경로는 사용하지 않았습니다. 기존 MetricLaneTracker, CornerPolicy, Pure Pursuit와 속도/시간 제한을 유지했습니다. CornerPolicy를 호출했어도 NORMAL/UNKNOWN으로 통과한 코스는 제자리 코너 상태 시험 성공으로 해석하지 않습니다.</p><p>Domain 172, 실제 20/22/52 도메인 차단, 동일 코스/카메라/폭 20cm/YOLO 320px/conf 0.55, 각 최대 65초. Ground truth는 평가에만 사용합니다.</p></section>
'''
    document += f'<section><h2>결과: {passed}/{len(results)} 완주</h2><p>이전 동일 코스 YOLO only는 0/4였습니다. 이번 결과는 각각 한 번의 시험이며 반복 재현성과 실차 검증은 별도입니다.</p><table><tr><th>코스</th><th>완주</th><th>이동</th><th>최대 중심 이탈</th><th>YOLO 0인데 흰 픽셀 추적한 상태 표본</th></tr>'+''.join(table)+'</table><p>상태 표본은 프레임 수가 아닙니다. PASS는 끝점 10cm 이내 도달 및 중심 이탈 폭/2 이내 기준입니다. 전체 footprint 안전 인증이나 멈춤 없는 완주를 뜻하지 않습니다.</p><p class="warn">서비스 재시작으로 실행이 중단돼 완료된 좌·우 90도 기록은 보존하고 S 시험만 재개했습니다. 기존 build 폴더에 다른 브랜치의 제거된 시험 프로파일이 남아 첫 패키징은 실패했습니다. 별도 /tmp build/install에서 다시 빌드해 성공했고, 제어에 사용한 모델·기본 config·코드는 동일합니다.</p></section>'
    tests = ET.parse(root/'regression.xml').getroot()
    total = sum(int(s.get('tests', 0)) for s in tests.iter('testsuite'))
    failures = sum(int(s.get('failures', 0))+int(s.get('errors', 0)) for s in tests.iter('testsuite'))
    document += f'<section><h2>회귀 검사</h2><p>{total}개 검사, 실패/오류 {failures}개. 스타일 3종 제외. 별도 build/install에서 pinky_move 빌드 성공. 실제 로봇용 기본값 simulation_yolo_white=false를 유지하고, 실제 도메인 및 원격 경로의 실험 활성화를 거부합니다.</p><h2>남은 S자 문제</h2><p>우측 차로는 좌우 경계 연결 모호함·center path fold·관측 범위 밖 코너 진입 판정으로 중간에 멈춘 뒤 YOLO 확인 이력이 만료됐습니다. 좌측 차로는 projected=1/fitted=1이어도 side association ambiguous가 반복됐습니다. 흰 픽셀이 있다고 해서 경로 계산과 좌우 연결이 자동 해결되지는 않았습니다.</p><h2>시뮬 재현</h2><pre>source /opt/ros/jazzy/setup.bash\nsource ~/ws/pinky_pro/install/setup.bash\nsource /tmp/pinky-hybrid-install-20261006/setup.bash\nros2 launch pinky_move lane_gazebo.launch.py domain:=172 perception:=hybrid connected_geometry:=true course:=left90</pre><p>수동 launch는 비활성 상태로 시작합니다. 자동 시험 도구는 시뮬 Domain 172에만 enable/disable을 호출합니다. 로컬 실험 코드는 실차에 배포하지 않았고 GitHub에 추가 푸시하지 않았습니다.</p></section>'
    document += ''.join(sections)+'</main></html>'
    (root/'report.html').write_text(document)
    print(json.dumps(dict(passed=passed, runs=len(results), report=str(root/'report.html'))))


if __name__ == '__main__':
    main()
