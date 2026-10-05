"""동일 YOLO/물리 코스의 기준·연결 기하 시험을 HTML로 기록한다."""
import argparse
import base64
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def asset(path, video=False, inline=True):
    if not path.exists():
        return '<p>자료 없음</p>'
    mime = 'video/mp4' if video else 'image/jpeg'
    src = ('data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
           if inline else str(path.relative_to(ROOT)))
    return (f'<video controls preload="metadata" playsinline src="{src}"></video>' if video else
            f'<img alt="{html.escape(path.parent.name)} 최종 화면" src="{src}">')


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--linked-videos', action='store_true')
    parser.add_argument('--after', default='connected_final')
    parser.add_argument('--no-videos', action='store_true', help='Notion 첨부용: 사진은 내장, 영상은 Git 자료 안내')
    args = parser.parse_args()
    ROOT = args.root
    regression = json.loads((ROOT/'dataset_regression.json').read_text())['summary']
    test_root = ET.parse(ROOT/'regression.xml').getroot()
    counts = {key: sum(int(s.get(key, 0)) for s in test_root.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    sections, overview = [], []
    for name in ('left90_yolo_lane0_r1', 'right90_yolo_lane0_r1',
                 's_sharp_yolo_lane0_r1', 's_sharp_yolo_lane1_r1'):
        runs = []
        cards = []
        for phase in ('baseline', args.after):
            directory = ROOT/phase/name
            data = json.loads((directory/'trial.json').read_text())
            summary = data['summary']
            statuses = data['observations']['status']
            last = next((r['text'] for r in reversed(statuses) if not r['text'].startswith('DISABLED')), '')
            runs.append(summary)
            cards.append(f'<article><h3>{phase}: {"PASS" if summary["passed"] else "FAIL"}</h3>'
                f'<p>이동 {summary["travelled_m"]:.3f}m · 최대 중심 이탈 {summary["max_cross_track_m"]*100:.2f}cm</p>'
                + asset(directory/'latest_debug.jpg')
                + ('<p>영상 원본: Git 브랜치의 '+phase+'/'+name+'/raw_frames.mp4 및 debug_frames.mp4. 로컬 전체 report.html에서 재생할 수 있습니다.</p>' if args.no_videos else
                   '<h4>가상 카메라 원본</h4>' + asset(directory/'raw_frames.mp4', True, not args.linked_videos)
                   + '<h4>실제 제어 debug</h4>' + asset(directory/'debug_frames.mp4', True, not args.linked_videos))
                + '<details><summary>마지막 상태와 판정</summary><pre>'+html.escape(last)+'</pre></details></article>')
        overview.append(f'<tr><td>{name}</td>'+''.join(
            f'<td>{"PASS" if s["passed"] else "FAIL"} / {s["travelled_m"]:.3f}m</td>' for s in runs)+'</tr>')
        sections.append('<section><h2>'+name+'</h2><div class="grid">'+''.join(cards)+'</div></section>')
    content = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO 연결 경로 보완 검증</title>
<style>body{margin:0;background:#edf1f5;color:#172332;font:16px/1.65 system-ui}main{max-width:1200px;margin:auto;padding:24px}section,header{background:white;padding:24px;margin-bottom:20px;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}img,video{width:100%;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f0f3f5;padding:12px}.warn{border-left:5px solid #dd9200;background:#fff3d8;padding:14px}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:10px}}</style><main>
<header><h1>YOLO + 기존 추종 구조<br>연결 골격 / miter 보완 시험</h1>
<p>기준 커밋 1579af9 · 2026-10-05 · 실제 로봇 주행/배포 없음</p>
<p class="warn">테스트·빌드 통과와 코스 완주는 다릅니다. 이번 결과에서 실패한 물리 시험을 성공으로 바꾸어 표시하지 않습니다. 실차 시험 브랜치는 선택형 실험 코드이며 완주 검증본이 아닙니다.</p></header>
<section><h2>바뀐 것 / 유지한 것</h2><p>일반 경로는 기존 parametric fitting과 normal offset을 그대로 사용합니다. 연결된 코너 관측의 추가 지지가 확인되거나 기존 오프셋이 실제로 실패한 경우에만 연결 골격/miter를 사용합니다. 관측 밖 출구를 만들거나 x순으로 다시 정렬하지 않습니다.</p>
<p>YOLO 모델·confidence·intrinsic/extrinsic·Pure Pursuit·좌우 이력·코너 상태 머신·입력/odometry 만료는 유지했습니다. 연결 계산은 동일 프레임 안에서만 캐시합니다. OpenCV 흰색 검출 우회 및 GT 경로 입력은 사용하지 않았습니다.</p>
<p>기본값 connected_geometry=false. 실제 카메라의 보정은 시뮬 이상 보정으로 덮어쓰지 않았습니다.</p></section>
'''
    content += f'<section><h2>동일 캐시 {regression["images"]}장</h2><p>목표 생성 {regression["before"]} → {regression["after"]}; 기존 성공→실패 {len(regression["regressions"])}장; 신규 성공 {len(regression["improvements"])}장. 기존 성공 목표 최대 변화: {regression["max_existing_target_shift_m"]:.6f}m.</p><p>독립 사진 cold start이며 실제 연속 주행 성공률이 아닙니다. 초기 connected 시험에서 기존 성공 좌표가 달라지는 문제가 있어 적용 순서를 실패 후 fallback으로 좁혔습니다. 초기 기록은 connected 폴더에 보존했고 아래 After는 최종 재시험 connected_final입니다.</p><h2>회귀 검사</h2><pre>{html.escape(json.dumps(counts, ensure_ascii=False))}</pre></section>'
    gains = json.loads((ROOT/'offline_gains/coordinates.json').read_text())
    for row in gains:
        content += '<section><h2>실사진 무동력 재생: '+row['key']+'</h2><p>'+html.escape(row['image'])+'</p><div class="grid">'
        for phase in ('before', 'after'):
            content += '<article><h3>'+phase+'</h3>'+asset(ROOT/'offline_gains'/(row['key']+'_'+phase+'.jpg'))+'<pre>'+html.escape(json.dumps(row[phase], ensure_ascii=False, indent=2))+'</pre></article>'
        content += '</div><p>동일 YOLO segmentation 캐시. 오프라인 목표 생성이며 실주행 After가 아닙니다.</p></section>'
    content += '<section><h2>동일 물리 조건 Before / After</h2><p>Domain 172 · YOLO only · 320px/conf 0.55 · 폭 20cm · left/right 90° 및 3라인 급 S 양쪽 차로 · 각 최대 65초. center checkpoint 10cm 이내, 중심 이탈 폭/2 이내로 판정합니다. 전체 footprint 안전 인증이 아닙니다.</p><table><tr><th>코스</th><th>Before</th><th>After</th></tr>'+''.join(overview)+'</table><p>검출이 0개가 되면 연결 기하만으로 새 관측을 만들 수 없습니다. 남은 과제는 같은 YOLO 입력을 유지한 검출 소실 원인/학습 도메인 분석과 시간 연속 시험입니다. 장시간 맹목 주행으로 가리지 않았습니다.</p></section>'
    content += ''.join(sections)+'</main></html>'
    output = ROOT/('report_notion.html' if args.no_videos else 'report_linked.html' if args.linked_videos else 'report.html')
    output.write_text(content)
    print(str(output), output.stat().st_size)


if __name__ == '__main__':
    main()
