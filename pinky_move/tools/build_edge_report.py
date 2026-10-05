"""Self-contained HTML + synthetic-clock lane regression simulator.

Uses actual geometry/controller functions, but no ROS initialization, publishers,
camera, network or robot connection. This is not a Gazebo physics simulation.
"""
import base64
import html
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'test')]
from pinky_move.metric_lane import MetricLaneTracker, pursuit
from test_metric_lane import calibration, masks_for_lines


def simulate(factory):
    """Reproduce an old target latch and a perception gap on a virtual clock.

    Masks stay fixed: geometry identity is known, but these are NOT independent
    physical observations. v/w are computed numbers, never sent to cmd_vel.
    """
    tracker = factory(tracking_gap_s=2.)
    mask = masks_for_lines()[1]
    first = tracker.update([mask], calibration(), 1., lane_width=.16)
    tracker.previous = dict(first, y_m=first['y_m']+.10)
    records = []
    for index in range(1, 10):
        now = 1.+index*.2
        # A later missing observation tests that it is not blindly replayed.
        visible = index != 7
        try:
            target = tracker.update([mask] if visible else [], calibration(),
                                    now, lane_width=.16)
            records.append(dict(t=round(now, 1), visible=visible, accepted=True,
                v=.03, w=pursuit(target, .03), x=target['x_m'], y=target['y_m'],
                reason='current observed path'))
        except ValueError as exc:
            records.append(dict(t=round(now, 1), visible=visible, accepted=False,
                                v=0., w=0., reason=str(exc)))
    return records


def picture(path, caption):
    src = 'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(caption)}" src="{src}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def timeline(rows):
    lines = ''.join(f'<tr><td>{r["t"]:.1f}s</td><td>{"검출" if r["visible"] else "소실"}</td>'
        f'<td class="{"ok" if r["accepted"] else "wait"}">{"목표 생성" if r["accepted"] else "정지"}</td>'
        f'<td>{r["v"]:.2f}</td><td>{r["w"]:.3f}</td><td>{html.escape(r["reason"])}</td></tr>' for r in rows)
    return '<table><thead><tr><th>가상 시각</th><th>입력</th><th>판정</th><th>계산 v(m/s)</th><th>계산 w(rad/s)</th><th>이유</th></tr></thead><tbody>'+lines+'</tbody></table>'


def main():
    source = subprocess.check_output(['git', 'show', '1ceed93:pinky_move/pinky_move/metric_lane.py'], cwd=ROOT, text=True)
    baseline = types.ModuleType('pinky_move._report_baseline')
    baseline.__package__ = 'pinky_move'
    exec(compile(source, '<baseline 1ceed93>', 'exec'), baseline.__dict__)
    before, after = simulate(baseline.MetricLaneTracker), simulate(MetricLaneTracker)
    assert not any(r['accepted'] for r in before)
    assert not after[0]['accepted'] and not after[1]['accepted']
    assert all(r['accepted'] for r in after[2:6])
    assert not after[6]['accepted'] and after[6]['v'] == after[6]['w'] == 0.
    assert all(r['accepted'] for r in after[7:])
    cases = ROOT/'reports/cases'
    simulation = dict(kind='synthetic-clock mask/history regression; no physics or ROS publishers',
                      baseline_commit='1ceed93', assertions_passed=True, before=before, after=after)
    (cases/'simulation_results.json').write_text(json.dumps(simulation, indent=2))
    first = cases/'20261005_stopped_side_ambiguous'
    second = cases/'20261005_005730_edge02'
    before_replay = json.loads((cases/'replay_before.json').read_text())
    after_replay = json.loads((cases/'replay_after.json').read_text())
    checks = json.loads((cases/'fix_validation.json').read_text())
    summary = json.loads((cases/'dataset_regression.json').read_text())['summary']
    version = subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, text=True).strip()
    document = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pinky 차선 추종 — 두 엣지 케이스 수정 보고서</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172332;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 24px}}header,section{{background:white;border-radius:16px;padding:28px;margin-bottom:22px;box-shadow:0 3px 16px #192c4210}}
h1{{line-height:1.3;font-size:32px}}h2{{font-size:24px}}h3{{font-size:19px}}.small{{font-size:14px;color:#4b6177}}
.warning{{padding:16px;background:#fff2d5;border-left:5px solid #d99b17;border-radius:6px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
figure{{margin:0}}img{{width:100%;height:auto;display:block;border-radius:8px}}figcaption{{font-size:14px;color:#4b6177;margin:8px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce3ec;padding:10px}}th{{background:#f2f5f9}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:16px;border-radius:8px;font-size:13px}}code{{font-size:.9em}}.ok{{color:#14794c}}.wait{{color:#ad6500}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}section,header{{padding:18px}}table{{font-size:12px}}th,td{{padding:5px}}}}
@media print{{body{{background:white}}section,header{{box-shadow:none;break-inside:avoid}}main{{max-width:none;padding:0}}h1{{font-size:26px}}img{{max-height:320px;object-fit:contain}}}}
</style></head><body><main>
<header><p class="small">2026-10-05 KST · debug/lane-edge-cases-20261005 · 코드 커밋 {version}</p>
<h1>Pinky 차선 추종<br>두 정지 엣지 케이스의 원인·수정·검증</h1>
<p>기존 YOLO segmentation → 바닥 투영 → 좌우 경계 → 중앙 경로 → Pure Pursuit 구조를 유지하고, 중복 경계 거부와 오래된 목표점 고착만 보완했습니다.</p>
<div class="warning">로봇 연결·배포·실제 주행 없음. 아래 시뮬레이션은 가상 시간과 마스크 이력을 사용한 소프트웨어 재현이며 Gazebo/물리 주행이 아닙니다. 계산된 속도는 어디에도 발행하지 않습니다.</div></header>
<section><h2>검증 요약</h2><table><tr><th>검사</th><th>결과</th><th>의미</th></tr>
<tr><td>전체 테스트</td><td class="ok">{checks['tests']['passed']} passed / 70.10초</td><td>스타일 3종 제외, 테스트 프로세스만 DDS URI 제외</td></tr>
<tr><td>빌드</td><td class="ok">pinky_move 성공 / 1.84초</td><td>브랜치 소스, 격리된 임시 install 경로</td></tr>
<tr><td>기존 동일 이미지 캐시</td><td>{summary['images']}장 · {summary['before']} → {summary['after']} 목표 생성</td><td>기존 성공→실패 {len(summary['regressions'])}장. 실주행 성공률 아님</td></tr>
<tr><td>가상 시간 이력</td><td class="ok">assertion 통과</td><td>3회 재확정, 소실 시 정지, 복귀 확인</td></tr></table></section>
<section><h2>01. 두 인스턴스가 같은 왼쪽 바닥 경계였음</h2>
<p><code>no unambiguous left/right boundary pair (projected=1, fitted=1; side association ambiguous)</code></p>
<div class="grid">{picture(first/'debug_image.jpg','Before — 현장 정지 화면')}{picture(first/'replay_after.jpg','After — 원본 JPEG를 재추론한 무동력 재생')}</div>
<h3>왜 멈췄나</h3><p>두 YOLO 인스턴스의 먼 가지는 달라도 14–48cm 바닥 관측 경계는 거의 동일했습니다. 두 차선 쌍으로 만들 수도 없고, 단일 경계 fallback은 후보의 전방 거리 차이가 6cm 미만이라 탈락했습니다.</p>
<h3>바꾼 부분</h3><p><code>equivalent_boundary()</code>는 전체 관측 곡선을 호 길이로 재표본화해 끝까지 8mm 이내로 일치하는 경계만 하나로 취급합니다. 동일 역할·연속 3회 확인 후 현재 경계로 정상 추종합니다. 가까운 몸 쪽 선을 무조건 고르는 정책을 재도입하거나 마스크를 합치지 않았습니다. 입구만 같고 출구가 다른 갈림길은 중복으로 인정하지 않습니다.</p>
<p class="ok">반복 재생: 기존 4회 모두 실패 → 수정 후 확인 2회 대기, 3·4회 목표 생성·유지.</p>
<p>재생 목표 x=14.77cm / y=-12.67cm, visible_side=left. 이 목표가 실제 코스를 통과한다는 증거는 아닙니다.</p></section>
<section><h2>02. 오래된 목표점과 계속 비교해 거부</h2><p><code>inferred target discontinuity</code></p>
<div class="grid">{picture(second/'debug_image.jpg','Before — 현장 목표점 거부 화면')}{picture(second/'replay_after.jpg','참고 재생 — 이 사진은 기존 cold start도 성공')}</div>
<h3>확인한 구조</h3><p>현재 목표 y가 previous와 max(5cm, 폭의 25%) 이상 달라지면 거부하지만 previous는 바뀌지 않습니다. 동일한 현재 경계가 계속 보여도 반복 거부가 가능했습니다. 당시 previous와 전체 마스크 이력이 없으므로 현장 시퀀스의 완전 재현은 아닙니다.</p>
<h3>바꾼 부분</h3><p>경계 정체성·피팅·normal offset 검사를 통과한 새 경로가 다른 시각에 3회 연속 일치하면 현재 목표로 재확정합니다. 새 목표끼리 2cm 이내, 전체 경로 8mm 이내 및 tracking gap 조건을 유지합니다. 빈 관측/다른 정체성/기하 오류는 확인 이력을 초기화하며 동일 시각과 긴 간격은 누적하지 않습니다.</p>
<div class="warning">원본과 debug 프레임은 약 5.33초 차이입니다. 사진이 새로 시작하면 성공하는 것과 과거 목표 이력 오류가 해결된 것은 별개의 검증입니다.</div></section>
<section><h2>무동력 시뮬레이션: 이전 목표 고착 + 관측 소실</h2>
<p>합성 오른쪽 경계로 실제 <code>MetricLaneTracker.update()</code>를 호출하고, previous 목표에 y=10cm 차이를 주입했습니다. 가상 시각은 프레임마다 0.2초 증가합니다. 성공한 경우에만 실제 <code>pursuit()</code> 함수로 v=0.03m/s와 w를 계산합니다. 이후 한 프레임을 비워 소실/복귀를 검사했습니다.</p>
<h3>Before — 기준 커밋 1ceed93</h3>{timeline(before)}
<h3>After — 현재 브랜치</h3>{timeline(after)}
<p>실제 로봇 위치를 적분하거나 타이어/카메라/충돌을 모델링하지 않았습니다. 이 실험은 오래된 목표 고착이 풀리는지 검사하는 재현 환경이며 완주 검증은 아닙니다.</p></section>
<section><h2>변경 파일 및 재실행</h2><p><code>metric_lane.py</code> / <code>test_saved_cases_20261005.py</code> / <code>test_metric_lane.py</code>. 기존 주행 구조와 로봇용 DDS 파일은 변경하지 않았습니다.</p>
<pre>cd /home/tory/pinky_nav_publish_yWXwM0/repo/pinky_move
/usr/bin/python3 tools/build_edge_report.py
/home/tory/venv/omx/bin/python tools/replay_saved_cases.py --output reports/cases/replay_after.json
/home/tory/venv/omx/bin/python tools/compare_cached_geometry.py --output reports/cases/dataset_regression.json</pre>
<p>HTML은 사진이 포함된 단일 파일입니다. 오프라인 브라우저에서 열거나 인쇄 → PDF로 저장할 수 있습니다.</p></section>
<section><h2>원문 재생 결과</h2><details><summary>Before JSON</summary><pre>{html.escape(json.dumps([{k:v for k,v in row.items() if k != 'diagnostics'} for row in before_replay], ensure_ascii=False, indent=2))}</pre></details>
<details><summary>After JSON</summary><pre>{html.escape(json.dumps([{k:v for k,v in row.items() if k != 'diagnostics'} for row in after_replay], ensure_ascii=False, indent=2))}</pre></details>
<h3>남은 검증</h3><p>실제 연속 프레임/odometry를 포함한 S자 및 급코너 주행, 목표 위치·조향의 적합성, 추론 만료/통신 지연, 쌍 중앙 경로의 별도 discontinuity는 아직 검증해야 합니다.</p></section>
</main></body></html>'''
    output = cases/'edge_case_report_20261005.html'
    output.write_text(document, encoding='utf-8')
    print(json.dumps(dict(html=str(output), simulation_assertions=True,
                         before_accepted=sum(r['accepted'] for r in before),
                         after_accepted=sum(r['accepted'] for r in after),
                         bytes=output.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
