"""Summarise actual Gazebo/YOLO closed-loop trials, including failures."""
import base64
from collections import Counter
import html
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

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
    relative = html.escape(path.relative_to(ROOT).as_posix(), quote=True)
    label = '가상 카메라 영상' if path.name == 'raw_frames.mp4' else '인식/제어 디버그 영상'
    return (f'<div class="clip"><h3>{label}</h3>'
            f'<video controls playsinline preload="none" aria-label="{label}">'
            f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
            '이 뷰어가 영상을 지원하지 않으면 HTML을 다운로드해 브라우저에서 여세요.'
            f'</video><p><a href="{relative}" download>MP4 다운로드</a></p></div>')


def trajectory_plot(folder, metadata, observations):
    """시험 평가용 ground truth 궤적. 이 좌표는 제어기에 제공하지 않는다."""
    records = observations.get('ground_truth', [])
    if not records or not metadata.get('boundaries'):
        return ''
    stripes = [np.asarray(p, float) for p in metadata['boundaries'].values()]
    actual = np.asarray([[r['x'], r['y']] for r in records], float)
    centre = np.asarray(metadata['centre'], float)
    all_points = np.vstack([actual, centre, *stripes])
    lo, hi = all_points.min(axis=0)-.15, all_points.max(axis=0)+.15
    scale = min(900/(hi[0]-lo[0]), 470/(hi[1]-lo[1]))
    def pixels(points):
        p = np.rint((np.asarray(points)-lo)*scale).astype(np.int32)
        p[:, 0] += 40
        p[:, 1] = 550-p[:, 1]
        return p
    image = np.full((600, 1000, 3), 65, np.uint8)
    for stripe in stripes:
        cv2.polylines(image, [pixels(stripe)], False, (245, 245, 245), max(2, round(.02*scale)))
    cv2.polylines(image, [pixels(centre)], False, (110, 110, 110), 1)
    cv2.polylines(image, [pixels(actual)], False, (255, 200, 30), 3)
    for point, colour in ((actual[0], (50, 255, 50)), (actual[-1], (80, 80, 255))):
        cv2.circle(image, tuple(pixels([point])[0]), 6, colour, -1)
    cv2.putText(image, 'WHITE: stripes / CYAN: physics trajectory / RED: final pose',
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    path = folder/'trajectory.png'
    cv2.imwrite(str(path), image)
    return figure(path, '실제 물리 주행 궤적 — ground truth는 평가에만 사용')


def main():
    summaries, panels, comparisons, current_trials = [], [], [], []
    for title, before, after in [('좌측 90도', 'left90_01', 'left90_aid_01'),
                                  ('우측 90도', 'right90_01', 'right90_aid_01')]:
        columns = []
        for label, run in [('Before — YOLO 단독 실패', before),
                           ('과거 참고 — 이전 흰 픽셀 보조 검출 사용, 완주', after)]:
            folder = ROOT/run
            if not (folder/'trial.json').exists():
                continue
            clips = ''.join(video_player(folder/name) for name in
                            ('raw_frames.mp4', 'debug_frames.mp4') if (folder/name).exists())
            columns.append(f'<div><h3>{html.escape(label)} · {run}</h3>{clips}</div>')
        comparisons.append(f'<section><h2>{title} 실패 → 성공 영상 비교</h2>'
            '<p>왼쪽은 기존 confidence=.55 YOLO 시험, 오른쪽은 과거 카메라 흰 픽셀 보조 검출 시험입니다. '
            '오른쪽은 이전 보조 검출 버전이며 새 OpenCV 전용 시험과 구분합니다. '
            '보조 검출 성공을 YOLO 단독 성능 개선으로 해석하지 않습니다. 영상의 재생 버튼을 각각 누르세요.</p>'
            f'<div class="comparison">{"".join(columns)}</div></section>')
    for file in sorted(ROOT.rglob('trial.json')):
        data = json.loads(file.read_text())
        summary = data['summary']
        summary = dict(summary, run=file.parent.relative_to(ROOT).as_posix())
        summary.setdefault('controller_enabled', summary.get('yolo_enabled', False))
        if 'opencv' in file.parent.name:
            # Old recorder called the enable flag yolo_enabled. Preserve raw
            # JSON; correct its interpretation only in this derived report.
            summary['perception'] = 'OpenCV white pixels'
            summary['yolo_enabled'] = file.parent.name == 'left90_opencv_v1'
            summary['perception_note'] = ('v1 모델 추론도 실행했지만 차선은 흰 픽셀로 대체'
                if summary['yolo_enabled'] else 'YOLO 모델 추론 없이 카메라 흰 픽셀 사용')
        course_file = file.parent/'course.json'
        course_data = json.loads(course_file.read_text()) if course_file.exists() else {}
        summary['stripe_geometry'] = course_data.get('stripe_geometry', 'legacy segmented boxes')
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
        summary['final_status'] = next((r['text'] for r in reversed(data['observations']['status'])
                                        if not r['text'].startswith('DISABLED')), '')
        summaries.append(summary)
        if '_yolo_only_' in file.parent.name:
            clips = ''.join(video_player(file.parent/name) for name in
                            ('raw_frames.mp4', 'debug_frames.mp4') if (file.parent/name).exists())
            geometry_label = ('새 연속 차선 시험' if 'mesh' in summary['stripe_geometry']
                              else '차선 형상 수정 전 YOLO 단독 시험')
            current_trials.append(f'<section><h2>{geometry_label} — {html.escape(file.parent.name)}</h2>'
                f'<p>YOLO 단독 / odometry 시간 비교 수정 / 흰 픽셀 보조 기능 없음 · '
                f'{"PASS" if summary["passed"] else "FAIL"} · 이동 {summary.get("travelled_m", 0):.3f}m</p>'
                f'<div class="comparison">{clips}</div>'
                f'<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>')
        images = figure(file.parent/'first_camera.jpg', '시작: 실제 Gazebo 카메라')
        images += figure(file.parent/'latest_debug.jpg', '최근: 인식 + 제어 debug')
        images += figure(file.parent/'final_camera.jpg', '시험 종료 카메라')
        images += trajectory_plot(file.parent, course_data, data['observations'])
        videos = ''.join(video_player(file.parent/name)
                        for name in ('raw_frames.mp4','debug_frames.mp4') if (file.parent/name).exists())
        panels.append(f'<section><h2>{html.escape(summary["run"])} — {"PASS" if summary["passed"] else "FAIL"}</h2>'
            f'<p>인식 방식: {html.escape(summary["perception"])} / 차선 형상: {html.escape(summary["stripe_geometry"])}</p>'
            f'<div class="pictures">{images}</div><div class="pictures">{videos}</div><details><summary>원문 수치와 오류</summary>'
            f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'
            f'<p><a href="{file.relative_to(ROOT).as_posix()}">전체 시간 이력 JSON</a></p></section>')
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    rows = ''.join(f'<tr><td>{html.escape(r["run"])}</td><td>{"PASS" if r["passed"] else "FAIL"}</td>'
        f'<td>{r.get("travelled_m",0):.3f}m</td><td>{r.get("max_cross_track_m",0)*100:.2f}cm</td>'
        f'<td>{r["nonzero_commands"]}</td><td>{html.escape(r["perception"])}</td>'
        f'<td>{html.escape(str(r["ordinary_error_counts"]))}</td></tr>' for r in summaries)
    shape_root = ROOT.parent/'stripe_continuity_20261005'
    shape_preview = figure(shape_root/'s_sharp_three_lines_top.png', '수정된 연속 mesh의 평면도 — 2차로 / 3라인')
    shape_preview += figure(shape_root/'gazebo_camera.png', '수정된 mesh를 렌더링한 실제 Gazebo 카메라 — 주행 disabled')
    mesh_summary = ROOT/'mesh_trials_summary.json'
    mesh_notice = ''
    if mesh_summary.exists():
        measured = json.loads(mesh_summary.read_text())
        mesh_notice = (f'<section><h2>새 연속 차선 시험 요약</h2><p>YOLO 단독 · '
            f'{measured["tests"]}회 중 {measured["passed"]}회 완주. '
            '좌·우는 코너 진입 중 검출 0개로 정지했고, 급 S자·3라인은 검출 1개를 '
            '저장된 경계 정체성과 연결하지 못해 정지했습니다. '
            '정지 시점의 odometry 보간은 세 시험 모두 정상이었습니다. '
            '차선 연결 형상 수정만으로 완주 문제가 해결된 것은 아닙니다.</p></section>')
    opencv_runs = [r for r in summaries if r['run'].startswith('opencv_final_suite_v2/')]
    opencv_notice = ''
    if opencv_runs:
        opencv_notice = ('<section><h2>새 OpenCV 전용 반복 시험</h2>'
            f'<p>{len(opencv_runs)}회 중 {sum(r["passed"] for r in opencv_runs)}회 완주. '
            '좌·우 90도 각각 2회, 급 S자·3라인은 오른쪽/왼쪽 차로 각각 2회를 시험합니다. '
            '완주 판정은 중심점 기준이고 실제 로봇의 완주나 footprint 전체 포함을 보증하지 않습니다.</p>'
            '<p>HSV 흰 픽셀 → 카메라 보정 기반 바닥 좌표 → 연결 골격 → 법선/miter 중앙 경로 → Pure Pursuit. '
            'CPU 골격 연산은 작업 스레드로 옮겨 odometry/control 콜백을 막지 않도록 했습니다. '
            '격리된 OpenCV 프로파일만 기존 staged-corner 거부 조건을 우회합니다. 실차 기본 YOLO 설정은 그대로입니다.</p>'
            '<p>v1~v4: 진입/경계 연결 제한으로 정지. v5: 첫 급 S자 완주. '
            'v6: 짧은 preview 변경 후 반대쪽 도로로 진입해 FAIL → 해당 변경 되돌림. '
            '최초 suite는 6회 완주 + 2회 반대 차로 코스 생성 실패였습니다. '
            '50/60cm S 구간에서 바깥 경계가 길이 0으로 접혀, 50/70cm로 수정하고 '
            '경계 붕괴·역방향을 생성 시 거부하는 회귀 검사를 추가했습니다. '
            '아래 opencv_final_suite_v2 행이 수정된 형상에서의 반복 검증입니다.</p></section>')
    regression = ROOT/'opencv_regression.xml'
    if regression.exists():
        suite = ET.parse(regression).getroot().find('testsuite')
        opencv_notice += ('<section><h2>기능 회귀와 영상 재생 검증</h2>'
            f'<p>기능 테스트 {suite.get("tests")}개 / failures {suite.get("failures")} / errors {suite.get("errors")} '
            f'/ {float(suite.get("time")):.2f}초. copyright/flake8/pep257 3종은 별도 스타일 검사로 제외했습니다.</p>'
            '<p>빌드 2개 성공. 새 영상 16개는 H.264 전체 decode 및 Firefox 실제 재생을 확인했습니다. '
            'opencv_regression.xml / opencv_browser_validation.json에 검증 원문이 있습니다.</p></section>')
    document = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pinky Gazebo 물리 주행 시험</title>
<style>body{{background:#eef2f7;color:#172332;font:16px/1.7 system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}section,header{{background:white;padding:24px;border-radius:14px;margin-bottom:20px}}.pictures{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:13px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5fa;padding:16px}}.notice{{background:#fff2d5;padding:14px}}@media(max-width:760px){{.pictures,.comparison{{grid-template-columns:1fr}}table{{font-size:11px}}}}@media print{{section{{break-inside:avoid}}body{{background:white}}}}</style>
</head><body><main><header><style>video{{display:block;width:100%;background:#111;max-height:480px}}</style><h1>Gazebo 물리 모델 · YOLO/OpenCV 카메라 주행 시험</h1>
<p>2026-10-05 KST / debug/lane-edge-cases-20261005</p>
<div class="notice">실제 Pinky 로봇 주행이 아닙니다. Gazebo 물리 엔진 안에서 기존 URDF 바퀴 모델이 이동한 시험입니다.
YOLO 시험과 새 OpenCV 전용 시험을 분리합니다. 정답 마스크·코스 좌표를 제어기에 주입하지 않았습니다.
사용자 요청으로 perception:=opencv인 격리된 시뮬에서만 흰 픽셀 검출을 사용합니다.
실차 기본값과 perception:=yolo는 기존 YOLO 입력입니다. *_yolo_only_*는 이전 YOLO 단독 결과입니다.
성공뿐 아니라 실패 기록도 보존했습니다.</div></header>
{mesh_notice}
{opencv_notice}
<section><h2>차선 연결 형상 수정</h2><p>선분별 사각형을 하나의 연속 miter mesh로 교체했습니다.
코너의 안쪽/바깥쪽 모서리를 정확히 공유하며 폭 2cm를 유지합니다. 아래는 새 형상의 정지 화면입니다.
새 형상 주행 영상은 ‘새 연속 차선 시험’ 섹션에 표시합니다. 이전 영상과 구분해 보세요.</p>
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
앞선 left90_aid_01의 일반 중앙 경로 추종 성공과 다른 시험입니다. 이후 OpenCV 개선 시험 결과는 아래 별도 기록을 확인하세요.</p></section>
{''.join(panels)}<section><h2>사용법</h2><p><a href="../../docs/GAZEBO_LANE_PHYSICS.md">GAZEBO_LANE_PHYSICS.md</a></p>
<pre>ros2 launch pinky_move lane_gazebo.launch.py course:=two_lines gui:=true domain:=172
# course: two_lines / left90 / right90 / s_bend / s_sharp / single_gap
# 2차로(3라인): lane_count:=2 target_lane:=0
# 기본 입력: perception:=yolo
# 새 OpenCV 물리 시험: perception:=opencv
# 급 S자·3라인: course:=s_sharp lane_count:=2 target_lane:=0 perception:=opencv</pre>
<p>카메라 640×480, fx=fy=566, URDF camera extrinsics, 이상적 0 distortion. 실제 보정은 변경하지 않았습니다.
영상은 15fps로 저장돼 실제 벽시계 시험 시간과 재생 시간은 다를 수 있습니다. JSON 시각이 기준입니다.</p></section></main></body></html>'''
    path = ROOT/'gazebo_physics_report.html'
    path.write_text(document,encoding='utf-8')
    print(json.dumps(dict(report=str(path),runs=len(summaries),passed=sum(r['passed'] for r in summaries)),indent=2))


if __name__ == '__main__':
    main()
