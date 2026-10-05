"""Summarise actual Gazebo/YOLO closed-loop trials, including failures."""
import base64
from collections import Counter
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'reports/gazebo_physics_20261005'


def figure(path, title):
    if not path.exists():
        return ''
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/jpeg;base64,{data}"><figcaption>{html.escape(title)}</figcaption></figure>'


def main():
    summaries, panels = [], []
    for file in sorted(ROOT.glob('*/trial.json')):
        data = json.loads(file.read_text())
        summary = data['summary']
        summary = dict(summary, run=file.parent.name)
        errors = Counter()
        for status in data['observations']['status']:
            error = re.search(r"'ordinary_error': '([^']+)'", status['text'])
            if error:
                errors[error.group(1)] += 1
        summary['ordinary_error_counts'] = dict(errors)
        summaries.append(summary)
        images = figure(file.parent/'first_camera.jpg', '시작: 실제 Gazebo 카메라')
        images += figure(file.parent/'latest_debug.jpg', '최근: 기존 YOLO + 제어 debug')
        images += figure(file.parent/'final_camera.jpg', '시험 종료 카메라')
        videos = ''.join(f'<p><a href="{file.parent.name}/{name}">{name} 다운로드/재생</a></p>'
                        for name in ('raw_frames.mp4','debug_frames.mp4') if (file.parent/name).exists())
        panels.append(f'<section><h2>{html.escape(file.parent.name)} — {"PASS" if summary["passed"] else "FAIL"}</h2>'
            f'<div class="pictures">{images}</div>{videos}<details><summary>원문 수치와 오류</summary>'
            f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'
            f'<p><a href="{file.parent.name}/trial.json">전체 시간 이력 JSON</a></p></section>')
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    rows = ''.join(f'<tr><td>{html.escape(r["run"])}</td><td>{"PASS" if r["passed"] else "FAIL"}</td>'
        f'<td>{r.get("travelled_m",0):.3f}m</td><td>{r.get("max_cross_track_m",0)*100:.2f}cm</td>'
        f'<td>{r["nonzero_commands"]}</td><td>{html.escape(str(r["ordinary_error_counts"]))}</td></tr>' for r in summaries)
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pinky Gazebo 물리 주행 시험</title>
<style>body{{background:#eef2f7;color:#172332;font:16px/1.7 system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}section,header{{background:white;padding:24px;border-radius:14px;margin-bottom:20px}}.pictures{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:13px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5fa;padding:16px}}.notice{{background:#fff2d5;padding:14px}}@media(max-width:760px){{.pictures{{grid-template-columns:1fr}}table{{font-size:11px}}}}@media print{{section{{break-inside:avoid}}body{{background:white}}}}</style>
</head><body><main><header><h1>실제 Gazebo 물리 모델 · 카메라 · YOLO 주행 시험</h1>
<p>2026-10-05 KST / debug/lane-edge-cases-20261005</p>
<div class="notice">실제 Pinky 로봇 주행이 아닙니다. Gazebo 물리 엔진 안에서 기존 URDF 바퀴 모델이 이동한 시험입니다.
YOLO는 가상 카메라 영상을 직접 처리했고 정답 마스크를 주입하지 않았습니다. 성공뿐 아니라 실패 기록도 보존했습니다.</div></header>
<section><h2>실행 결과</h2><table><thead><tr><th>시험</th><th>완주</th><th>이동 거리</th><th>최대 중심 이탈</th><th>비영점 명령</th><th>관측 오류</th></tr></thead><tbody>{rows}</tbody></table>
<p>PASS: 물리 ground truth가 목표 반경 10cm에 도달 + 중심점이 lane_width/2 밖으로 이탈하지 않음.
로봇 전체 footprint 포함 여부와 실제 환경의 안전성/완주를 보증하지 않습니다.</p>
<p>첫 직선 FAIL은 흰 선 끝 소실로 1.81m에서 정지했습니다. 목표 뒤 50cm 차선을 연장한 two_lines_02 재시험을 별도 보존합니다.
좌/우 직각은 가로 차선이 보이는 정지 영상에서 YOLO의 차선 검출이 없고, S자는 다중 경계 모호성으로 출발하지 못했습니다.</p></section>
{''.join(panels)}<section><h2>사용법</h2><p><a href="../../docs/GAZEBO_LANE_PHYSICS.md">GAZEBO_LANE_PHYSICS.md</a></p>
<pre>ros2 launch pinky_move lane_gazebo.launch.py course:=two_lines gui:=true domain:=172
# course: two_lines / left90 / right90 / s_bend / single_gap</pre>
<p>카메라 640×480, fx=fy=566, URDF camera extrinsics, 이상적 0 distortion. 실제 보정은 변경하지 않았습니다.
영상은 15fps로 저장돼 실제 벽시계 시험 시간과 재생 시간은 다를 수 있습니다. JSON 시각이 기준입니다.</p></section></main></body></html>'''
    path = ROOT/'gazebo_physics_report.html'
    path.write_text(document,encoding='utf-8')
    print(json.dumps(dict(report=str(path),runs=len(summaries),passed=sum(r['passed'] for r in summaries)),indent=2))


if __name__ == '__main__':
    main()
