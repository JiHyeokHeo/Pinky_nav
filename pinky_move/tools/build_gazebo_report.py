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
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return f'<figure><img src="data:{mime};base64,{data}"><figcaption>{html.escape(title)}</figcaption></figure>'


def video_player(path):
    """Inline media keeps playback working if the HTML alone is downloaded."""
    data = base64.b64encode(path.read_bytes()).decode()
    relative = html.escape(path.parent.name+'/'+path.name,quote=True)
    label = '가상 카메라 영상' if path.name == 'raw_frames.mp4' else 'YOLO/제어 디버그 영상'
    return (f'<div class="clip"><h3>{label}</h3>'
            f'<video controls playsinline preload="none" aria-label="{label}">'
            f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
            '이 뷰어가 영상을 지원하지 않으면 HTML을 다운로드해 브라우저에서 여세요.'
            f'</video><p><a href="{relative}" download>MP4 다운로드</a></p></div>')


def main():
    summaries, panels, comparisons, current_trials = [], [], [], []
    for title, before, after in [('좌측 90도', 'left90_01', 'left90_aid_01'),
                                  ('우측 90도', 'right90_01', 'right90_aid_01')]:
        columns = []
        for label, run in [('Before — YOLO 단독 실패', before),
                           ('과거 참고 — 현재 제거된 흰 픽셀 보조 검출 사용, 완주', after)]:
            folder = ROOT/run
            if not (folder/'trial.json').exists():
                continue
            clips = ''.join(video_player(folder/name) for name in
                            ('raw_frames.mp4', 'debug_frames.mp4') if (folder/name).exists())
            columns.append(f'<div><h3>{html.escape(label)} · {run}</h3>{clips}</div>')
        comparisons.append(f'<section><h2>{title} 실패 → 성공 영상 비교</h2>'
            '<p>왼쪽은 기존 confidence=.55 YOLO 시험, 오른쪽은 과거 카메라 흰 픽셀 보조 검출 시험입니다. '
            '현재 보조 검출 코드/옵션은 제거됐으며 오른쪽 결과는 현재 버전의 성능이 아닙니다. '
            '보조 검출 성공을 YOLO 단독 성능 개선으로 해석하지 않습니다. 영상의 재생 버튼을 각각 누르세요.</p>'
            f'<div class="comparison">{"".join(columns)}</div></section>')
    for file in sorted(ROOT.glob('*/trial.json')):
        data = json.loads(file.read_text())
        summary = data['summary']
        summary = dict(summary, run=file.parent.name)
        # The first aid trial predates provenance in recorder metadata.
        if file.parent.name == 'left90_aid_01':
            summary['perception'] = 'YOLO + white pixel aid (launch white_lane_aid:=true)'
        summary.setdefault('perception', 'YOLO only')
        errors = Counter()
        for status in data['observations']['status']:
            error = re.search(r"'ordinary_error': '([^']+)'", status['text'])
            if error:
                errors[error.group(1)] += 1
        summary['ordinary_error_counts'] = dict(errors)
        summaries.append(summary)
        if '_yolo_only_' in file.parent.name:
            clips = ''.join(video_player(file.parent/name) for name in
                            ('raw_frames.mp4', 'debug_frames.mp4') if (file.parent/name).exists())
            current_trials.append(f'<section><h2>현재 버전 재시험 — {html.escape(file.parent.name)}</h2>'
                f'<p>YOLO 단독 / odometry 시간 비교 수정 / 흰 픽셀 보조 기능 없음 · '
                f'{"PASS" if summary["passed"] else "FAIL"} · 이동 {summary.get("travelled_m", 0):.3f}m</p>'
                f'<div class="comparison">{clips}</div>'
                f'<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>')
        images = figure(file.parent/'first_camera.jpg', '시작: 실제 Gazebo 카메라')
        images += figure(file.parent/'latest_debug.jpg', '최근: 기존 YOLO + 제어 debug')
        images += figure(file.parent/'final_camera.jpg', '시험 종료 카메라')
        videos = ''.join(video_player(file.parent/name)
                        for name in ('raw_frames.mp4','debug_frames.mp4') if (file.parent/name).exists())
        panels.append(f'<section><h2>{html.escape(file.parent.name)} — {"PASS" if summary["passed"] else "FAIL"}</h2>'
            f'<p>인식 방식: {html.escape(summary["perception"])}</p>'
            f'<div class="pictures">{images}</div><div class="pictures">{videos}</div><details><summary>원문 수치와 오류</summary>'
            f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'
            f'<p><a href="{file.parent.name}/trial.json">전체 시간 이력 JSON</a></p></section>')
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    rows = ''.join(f'<tr><td>{html.escape(r["run"])}</td><td>{"PASS" if r["passed"] else "FAIL"}</td>'
        f'<td>{r.get("travelled_m",0):.3f}m</td><td>{r.get("max_cross_track_m",0)*100:.2f}cm</td>'
        f'<td>{r["nonzero_commands"]}</td><td>{html.escape(r["perception"])}</td>'
        f'<td>{html.escape(str(r["ordinary_error_counts"]))}</td></tr>' for r in summaries)
    shape_root = ROOT.parent/'stripe_continuity_20261005'
    shape_preview = figure(shape_root/'s_sharp_three_lines_top.png', '수정된 연속 mesh의 평면도 — 2차로 / 3라인')
    shape_preview += figure(shape_root/'gazebo_camera.png', '수정된 mesh를 렌더링한 실제 Gazebo 카메라 — 주행 disabled')
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pinky Gazebo 물리 주행 시험</title>
<style>body{{background:#eef2f7;color:#172332;font:16px/1.7 system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}section,header{{background:white;padding:24px;border-radius:14px;margin-bottom:20px}}.pictures{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:13px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5fa;padding:16px}}.notice{{background:#fff2d5;padding:14px}}@media(max-width:760px){{.pictures,.comparison{{grid-template-columns:1fr}}table{{font-size:11px}}}}@media print{{section{{break-inside:avoid}}body{{background:white}}}}</style>
</head><body><main><header><style>video{{display:block;width:100%;background:#111;max-height:480px}}</style><h1>실제 Gazebo 물리 모델 · 카메라 · YOLO 주행 시험</h1>
<p>2026-10-05 KST / debug/lane-edge-cases-20261005</p>
<div class="notice">실제 Pinky 로봇 주행이 아닙니다. Gazebo 물리 엔진 안에서 기존 URDF 바퀴 모델이 이동한 시험입니다.
YOLO는 가상 카메라 영상을 직접 처리했고 정답 마스크를 주입하지 않았습니다.
white pixel aid 시험은 실제 카메라의 흰 픽셀 보조 검출도 사용했으며 YOLO 단독 성공이 아닙니다.
현재 버전은 보조 검출을 제거하고 YOLO만 사용합니다. *_yolo_only_* 행은 제거 후 재시험입니다.
성공뿐 아니라 실패 기록도 보존했습니다.</div></header>
<section><h2>차선 연결 형상 수정</h2><p>선분별 사각형을 하나의 연속 miter mesh로 교체했습니다.
코너의 안쪽/바깥쪽 모서리를 정확히 공유하며 폭 2cm를 유지합니다. 아래는 새 형상의 정지 화면입니다.
기존 주행 영상은 형상 수정 전 촬영 기록이며 새 형상으로 완주했다는 결과가 아닙니다.</p>
<div class="comparison">{shape_preview}</div></section>
{''.join(current_trials)}
<section><h2>실행 결과</h2><table><thead><tr><th>시험</th><th>완주</th><th>이동 거리</th><th>최대 중심 이탈</th><th>비영점 명령</th><th>인식 방식</th><th>관측 오류</th></tr></thead><tbody>{rows}</tbody></table>
<p>PASS: 물리 ground truth가 목표 반경 10cm에 도달 + 중심점이 lane_width/2 밖으로 이탈하지 않음.
로봇 전체 footprint 포함 여부와 실제 환경의 안전성/완주를 보증하지 않습니다.</p>
<p>첫 직선 FAIL은 흰 선 끝 소실로 1.81m에서 정지했습니다. 목표 뒤 50cm 차선을 연장한 two_lines_02 재시험을 별도 보존합니다.
기존 좌/우 직각은 confidence=.55에서 검출이 없어 정지했고, S자는 다중 경계 모호성으로 출발하지 못했습니다.
보조 검출 재시험 결과는 아래 별도 행으로 표시합니다.</p></section>
{''.join(comparisons)}
<section><h2>직선 실패 → 성공 / 이후 남은 실패</h2><p>two_lines_01 → two_lines_02는 차선 끝 연장 후 완주했습니다.
두 시험은 MP4 녹화 기능 추가 전에 수행해 영상이 없으며, 아래 사진과 원문 로그로 확인할 수 있습니다.</p>
<p>left90_odomfix_01은 odometry 수정 후 제자리 회전까지 진행했지만 진출 차선 재획득에서 멈춰 실패했습니다.
앞선 left90_aid_01의 일반 중앙 경로 추종 성공과 다른 시험입니다. 급 S자·3라인도 아직 완주하지 못했습니다.</p></section>
{''.join(panels)}<section><h2>사용법</h2><p><a href="../../docs/GAZEBO_LANE_PHYSICS.md">GAZEBO_LANE_PHYSICS.md</a></p>
<pre>ros2 launch pinky_move lane_gazebo.launch.py course:=two_lines gui:=true domain:=172
# course: two_lines / left90 / right90 / s_bend / s_sharp / single_gap
# 2차로(3라인): lane_count:=2 target_lane:=0
# 현재 입력: YOLO segmentation only (흰 픽셀 보조 기능 제거)</pre>
<p>카메라 640×480, fx=fy=566, URDF camera extrinsics, 이상적 0 distortion. 실제 보정은 변경하지 않았습니다.
영상은 15fps로 저장돼 실제 벽시계 시험 시간과 재생 시간은 다를 수 있습니다. JSON 시각이 기준입니다.</p></section></main></body></html>'''
    path = ROOT/'gazebo_physics_report.html'
    path.write_text(document,encoding='utf-8')
    print(json.dumps(dict(report=str(path),runs=len(summaries),passed=sum(r['passed'] for r in summaries)),indent=2))


if __name__ == '__main__':
    main()
