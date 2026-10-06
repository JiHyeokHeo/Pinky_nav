"""S자 버그 수정 전후의 원본 물리 시험 기록으로 HTML을 만든다."""
import argparse
import ast
import base64
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def asset(path, video=False):
    mime = 'video/mp4' if video else 'image/jpeg'
    uri = 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
    return (f'<video controls preload="metadata" playsinline src="{uri}"></video>' if video
            else f'<img alt="실제 시험 디버그 이미지" src="{uri}">')


def main():
    root = Path(__file__).resolve().parents[1]/'reports/yolo_white_20261006'
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='s_fix_final_v29')
    args = parser.parse_args()
    output = root/args.phase
    records = []
    rows, sections = [], []
    for path in sorted(output.glob('*/trial.json')):
        data = json.loads(path.read_text())
        summary = data['summary']
        baseline = json.loads((root/'staged'/path.parent.name/'trial.json').read_text())['summary']
        short = root/'s_fix_v2'/path.parent.name/'trial.json'
        equal_time = json.loads(short.read_text())['summary'] if short.exists() else None
        statuses = data.get('observations', {}).get('status', [])
        geometry_white = geometry_without_yolo = 0
        for sample in statuses:
            try:
                stats = ast.literal_eval(sample['text'].split('supplement=', 1)[1].split(', metric_target=', 1)[0])
            except (ValueError, SyntaxError, IndexError):
                continue
            if not isinstance(stats,dict):
                continue  # 초기 DISABLED/첫 추론 대기에는 supplement=None.
            if stats.get('white_stage_geometry') or stats.get('white_map_geometry'):
                geometry_white += 1
                geometry_without_yolo += stats.get('yolo', 0) == 0
        diagnostics = dict(status_samples=len(statuses), geometry_white_samples=geometry_white,
                           geometry_without_yolo_samples=geometry_without_yolo,
                           inference_failure_samples=sum('inference failed' in s['text'] for s in statuses))
        records.append(dict(run=path.parent.name, before=baseline,
                            after_65s=equal_time, after_200s=summary, diagnostics=diagnostics))
        rows.append(f'<tr><td>{path.parent.name}</td><td>{baseline["travelled_m"]:.3f}m</td>'
            f'<td>{equal_time["travelled_m"]:.3f}m</td>' if equal_time else
            f'<tr><td>{path.parent.name}</td><td>{baseline["travelled_m"]:.3f}m</td><td>—</td>')
        rows[-1] += (f'<td>{"PASS" if summary["passed"] else "FAIL"} · '
                     f'{summary["travelled_m"]:.3f}m</td><td>{100*summary["max_cross_track_m"]:.2f}cm</td></tr>')
        section = '<section><h2>'+path.parent.name+'</h2>'
        if equal_time:
            section += '<h3>수정 전</h3>'
            section += asset(root/'staged'/path.parent.name/'latest_debug.jpg')
        section += '<h3>수정 후</h3>'+asset(path.parent/'latest_debug.jpg')
        section += '<h3>원본 카메라</h3>'+asset(path.parent/'raw_frames.mp4', True)
        section += '<h3>제어 디버그</h3>'+asset(path.parent/'debug_frames.mp4', True)
        section += (f'<p>상태 로그 표본 {len(statuses)}개 중 현재 코너 흰 픽셀 대응 보완 '
                    f'{geometry_white}회, 그중 YOLO 검출 0인 표본 {geometry_without_yolo}회. '
                    '이는 로그 표본 수이며 전체 카메라 프레임 수가 아닙니다.</p>')
        section += '<details><summary>원본 판정 결과</summary><pre>'+html.escape(
            json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></details></section>'
        sections.append(section)
    tests = ET.parse(output/'regression.xml').getroot()
    count = sum(int(s.get('tests', 0)) for s in tests.iter('testsuite'))
    failed = sum(int(s.get('failures', 0))+int(s.get('errors', 0)) for s in tests.iter('testsuite'))
    passed = sum(r['after_200s']['passed'] for r in records)
    (output/'summary.json').write_text(json.dumps(dict(passed=passed, total=len(records),
        actual_robot_driven=False, test_count=count, failed_tests=failed, results=records),
        ensure_ascii=False, indent=2))
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S자 경로 생성 버그 수정 및 물리 시험</title><style>body{font:16px/1.65 system-ui;background:#edf1f5;color:#182536}main{max-width:1050px;margin:auto;padding:24px}section{background:white;padding:24px;margin-bottom:20px;border-radius:12px}img,video{width:100%;max-width:750px;display:block;margin:12px auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><main>
<section><h1>S자 경로 생성 버그 수정</h1><p>2026-10-06 · YOLO + 현재 흰 픽셀 보충 · Gazebo 물리 시험 · 실제 장비 움직임/배포 없음</p>
<p>{{verification_status}}</p>
<h2>왜 멈췄나?</h2><ol><li>과거에 보았던 먼 출구의 흰 차선 조각이 현재 경계 후보에 섞여, 좌우 경계 대응이 모호해졌습니다.</li>
<li>두 차선의 경로 생성에는 connected_geometry 설정이 전달되지 않아, 한 차선과 다른 복구 로직을 사용했습니다.</li>
<li>카메라에서 잘린 현재 진입선만 검사해 회전 중심이 관측 시작점보다 약 2cm 뒤에 있으면 거부했습니다. 이전에 관측했던 진입선의 지지를 활용하지 않았습니다.</li>
<li>일반 Pure Pursuit 목표 범위 밖으로 진입선이 이동하면, 현재 코너의 실관측까지 무효 처리했습니다.</li></ol>
<h2>수정</h2><p>현재 제어 범위보다 먼 독립 흰 조각을 바닥 좌표로 제외합니다. 가까운 부분이 있는 연결 선의 먼 출구는 보존합니다. PP의 최소 목표 거리와 코너 관측 최소 거리를 분리해 가까운 실제 선을 버리지 않습니다.</p>
<p>YOLO 인스턴스별 유일한 흰 성분을 대응시키고 optical-flow 대응도 양방향 일대일로 제한했습니다. 한 모델 인스턴스나 과거 추적선이 여러 경계 후보로 증식하지 않습니다. 화면 아래쪽 우선 선택 규칙은 추가하지 않았습니다.</p>
<p>두 차선 처리에도 기존 연결 골격 복구 설정을 전달합니다. 현재 코너와 일치하는 두 번의 10초 지도 내 odom 변환 관측(8mm/10도 대응)만 잘린 진입선 지지로 사용합니다. 회전각·출구·코너는 현재 관측 그대로입니다. 지도 보관 시간이 늘어난 것이며 빈 영상 주행 시간은 늘리지 않았습니다.</p>
<p>연결 골격 우선 추출도 시험했지만 기존 90도 코스에 회귀가 발생해 최종 실행에서는 비활성화했습니다. 기존 축 추출/실패 시 연결 복구 방식을 유지합니다. 현재 픽셀 선이 10초 odom 지도에서 두 번의 독립 실관측과 5cm/15mm/접선 대응을 만족할 때만 좌우 역할 힌트를 제공합니다. 이 힌트는 과거 목표 재생이나 빈 영상 주행이 아닙니다.</p>
<p>확정된 코너에서는 CURRENT 연결 골격 한 개가 고정된 odom 경계에 정확히 대응하면 행·열 피팅보다 먼저 관측으로 사용합니다. 빈 영상이나 모호한 두 후보는 승인하지 않습니다. 회전 중심을 매 프레임 새로 만들지 않습니다. 일반 전방 목표 생성이 실패해도 현재 실제 선의 코너 분류는 수행하며, 여러 프레임에서 확인된 코너만 회전을 승인합니다.</p>
<p>새 출구가 저장된 끝점을 넘어 같은 방향으로 길어진 경우 그 일직선 끝부분만 대응률 계산에서 제외합니다. 끝점에서 15mm/10도 내에 있는 실제 연결 부분만 해당하며, 내부 불일치나 옆으로 어긋난 평행 차선은 제외하지 않습니다. 긴 새 출구 때문에 기존 동일 코너의 대응률이 희석되는 오류를 수정했습니다.</p>
<p>유효한 현재 중앙 경로가 있고 코너가 lookahead + 로봇 전면보다 멀면 그 경로로 접근하며, 코너 진입 피벗을 조기 확정하지 않습니다. 연결 골격 우선 실험의 90도 회귀(v5/v6)는 자료와 재현 패치를 별도 보존하고 후속 시험을 중단했습니다.</p>
<p>안쪽 경계의 L 코너에서는 중앙 miter 교점이 원래 경계 끝점보다 폭/2 · tan(회전각/2)만큼 앞에 생깁니다. 기존 지지 검사에서 이 정상적인 10cm(폭 20cm, 90도)를 외삽으로 거부하던 오류를 수정했습니다. 실제 두 출구 선분의 지지와 원래 10mm 피팅 오차 검사는 유지합니다. 반대쪽 실제 경계가 고정 코너의 측정 경계/폭 기반 miter 경계에 유일하게 대응하면 목표를 바꾸지 않고 관측을 이어받습니다. 다른 선이나 빈 영상은 이를 대신할 수 없습니다.</p>
<p>YOLO가 코너를 확정한 뒤 수평 차선을 놓치고 optical-flow 의미 추적 10초가 만료되는 경우도 확인했습니다. 이때 확정된 경계 또는 차선폭으로 계산한 반대 경계와 유일하게 대응하는 현재 흰 픽셀만 다시 관측합니다. 과거 마스크를 재생하지 않고 새 차선을 흰색만으로 확정하지 않습니다. 사용 여부는 white_stage_geometry로, 실제 YOLO 검출 개수는 yolo로 각각 기록합니다. 따라서 YOLO 단독 성공과 동일한 의미가 아닙니다. 이 보완은 격리된 시뮬레이션 전용이며 실차에 활성화하지 않았습니다.</p>
<p>물리 재시험 중 재탐색 상태의 rclpy.Time를 디버그 스냅샷에서 deepcopy하다 C 핸들 pickle 예외가 발생하는 버그도 확인했습니다. 이는 모델 오류가 아닌 화면 표시 오류인데 기존에는 추론 실패로 처리했습니다. 시간은 nanoseconds/clock_type 값으로 독립 복사하고, 진단 생성 예외는 경고만 남겨 유효한 추론을 무효화하지 않도록 분리했습니다. 실제 ROS 시간 객체와 표시 예외 회귀 테스트를 추가했습니다.</p>
<p>양쪽이 검출된 코너에서도 반대쪽 선이 직선 진입부만 관측되면 저장된 반대 경계에 출구가 없었습니다. 출구 접선과 15도 이내로 일치하는 실제 지지가 5cm 미만인 경우, 확정된 두 다리와 폭으로 반대 경계 대응 기준을 만듭니다. 그 기준은 이동 경로를 대신하지 않으며 반드시 현재 실제 픽셀이 일치해야 관측을 이어받습니다. 폭 경계 생성 실패 시 기존 실제 관측은 보존합니다.</p>
<p>S자 원본 카메라를 바닥으로 재투영한 진단에서 동일 선은 연결 골격으로 S_BEND(58도)였지만 축 피팅에서는 NORMAL(5.5도)이었습니다. 축 피팅이 유효한 한 다리만 택하면 잘못된 가상 중앙선을 먼저 따라 다음 코너를 잘못 확정할 수 있습니다. 기존 추적기가 좌우/폭을 실제 관측으로 배정한 선에 연결 골격이 유일하게 대응할 때만 전체 굽힘을 코너 관측에 전달합니다. S_BEND인 경우 그 현재 실제 선을 normal/miter offset하여 기존 S 경로 추적/Pure Pursuit로 전달합니다. 모호한 다중 경계, 미배정 선, 실차 프로파일은 변경하지 않습니다. 새로운 연결 관측 시험은 먼 굽힘을 보고 조기 회전하지 않는지도 검사합니다.</p>
<p>한 개의 현재 마스크가 기존 추적기에서 좌우 배정된 경우에는 같은 마스크의 연결 골격을 사용합니다. 여러 다리 사이를 가로지른 잘못된 축 보간 곡선에 다시 맞춰야 한다고 요구하지 않습니다. 여러 마스크이면 기존 유일한 대응 검사를 유지하며, 기존 추적기가 관측을 거부한 미배정/모호한 경우에는 승격하지 않습니다.</p>
<p>기하 처리 중 executor가 odom callback을 받을 수 없어 처리 후 신선도 재검사가 촬영 시각 pose까지 없애는 문제를 분리했습니다. 처리 시작 때 검증한 촬영 pose를 같은 카메라 stamp에만 사용하고, 다음 영상에는 재사용하지 않습니다. 실제 명령의 현재 odom freshness·카메라/추론 age·회전 제한은 유지합니다. 처리 시간을 감추거나 odom timeout을 늘리지 않습니다.</p>
<p>두 실제 경계의 폭과 좌우가 확인된 쌍에서는 주 관측뿐 아니라 반대 경계에 보이는 연결 S도 검사합니다. 두 경계 중 유일하게 전체 S가 보이면 그 경계의 기존 역할/폭으로 중앙 경로를 만듭니다. 양쪽 다 모호하면 기존 처리를 유지합니다. 측정되지 않은 폭으로 다른 선을 임의 배정하지 않습니다.</p>
<p>그 경계를 선택한 뒤에는 추적기의 preferred_observation_side도 같은 기존 역할로 유지합니다. 이 선택이 추적기에 전달되지 않으면 다음 쌍에서 이전 주 경계로 되돌아가 S 경로가 identity changed로 거부되는 통합 문제가 있었습니다. 선의 좌우를 바꾸거나 미검출 목표를 승인하는 변경은 아닙니다.</p>
<p>코너 확정 직전 YOLO/optical-flow가 놓친 현재 흰 선도 최근 odom 지도에서 두 독립 실관측과 유일하게 대응하면 관측을 이어받습니다. 최근 실제 YOLO 확인 10초 이내로 제한하며 현재 픽셀 5cm/15mm/접선 대응 조건을 유지합니다. 보관된 목표로 빈 화면을 운전하거나 모델 없이 시작하지 않습니다. 사용 여부는 white_map_geometry로 따로 기록하며 원래 추적기의 역할 힌트에 전달합니다.</p>
<p>지도 보완은 차선 후보가 모두 없을 때에만 적용합니다. 기존 YOLO/flow 결과가 있으면 그대로 유지합니다. v15에서 유효한 현재 검출까지 지도 선 하나로 덮으면 경계 선택이 달라져 90도 회전에도 회귀가 발생할 수 있음을 확인해 적용 순서를 수정하고 통합 테스트를 추가했습니다. v15 오른쪽 시험은 supervisor 종료로 최종 판정 없이 중단돼 부분 자료로만 보존합니다.</p>
<p>v16 로그에서 LOCAL_MAP 복구가 아직 지나지 않은 S 경로를 삭제하는 문제를 확인했습니다. 현재 측정 경로로 복구하되 미완료 S 경로는 보존하도록 했습니다. UNKNOWN 분류도 현재 측정 경로를 갖고 있으면 기존 S 대응 검사로 보내며, 분류 이름만으로 저장 경로를 지우지 않습니다. 기존 실차 프로파일에는 적용하지 않는 시뮬레이션 마커로 구분합니다.</p>
<p>직각 꼭짓점의 수치 접선 한 점 때문에 전체 경계를 거부하는 문제도 분리했습니다. 현재 중앙선 10cm 이상이 저장 경로와 1cm 이내, 접선 cosine 0.98 이상으로 대응하고 실제 경계의 90% 이상이 대응한 경우에만 꼭짓점 접선 이상치를 허용합니다. 실제 위치 불일치, 역순 경계, 현재 관측 없음, odom 신선도 제한은 그대로 유지합니다. 긍정·부정 회귀 테스트를 함께 추가했습니다.</p>
<p>v18에서는 첫 굽힘 직전 약 60cm 이동 뒤 차선이 카메라 아래로 사라져 멈췄습니다. 연속 직각 코스 전체가 S_BEND로 먼저 분류되면서 기존 직각 코너 진입·회전 절차를 거치지 않는 구조 문제를 확인했습니다. 단계 회전을 사용하는 hybrid 프로파일에서 첫 실제 두 다리가 각각 12cm 이상이고 75~115도인 경우만 첫 코너를 분리합니다. 다음 코너 직전 3cm까지만 현재 관측을 사용하고 기존 잔차/출구 지지/3프레임 확정을 그대로 검사합니다. 부드러운 S나 짧은 다리는 이 방식으로 승격하지 않습니다. 일반 진입 중앙 목표는 그대로 유지합니다.</p>
<p>저장한 첫 카메라 프레임의 동일 기하 입력에서 가까운 골격 대응률이 55%여서 같은 우측 경계가 탈락했습니다. 골격의 계단 접선을 관측 형상 오차 8mm 이내로 정리하자 100% 대응, 위치 오차 5.84mm, 지지 길이 8.2cm가 확인됐습니다. 최종 통합 재생에서도 우측 경계/첫 90.07도 코너가 반환됐습니다. 이 결과는 해당 프레임의 기하 재생 성공이며 물리 완주 성공과 구분합니다. 숫자는 first_frame_geometry_trace.json에 보존했습니다.</p>
<p>초기 S 분류가 먼저 생성된 경우 이후 명확한 첫 직각을 다시 S로 덮어쓰는 상태 우선순위도 보완했습니다. 현재 진입 다리에 정렬된 첫 직각을 3프레임 확정하고 기존 코너 진입 조건까지 통과한 뒤에만 연속 S 경로를 교체합니다. 후보 1~2회 때는 이전 경로를 지우지 않습니다. 좌/우 전환 회귀 테스트를 추가했습니다.</p>
<p>실시간 단계별 로그에서는 첫 골격에 약 3cm/17도 조각이 붙어 있어 첫 두 선분 검사에서 탈락했습니다. 이처럼 5cm 미만·20도 이하의 선두 변화만 제외한 뒤 첫 지지 직각을 찾습니다. 실제 급 코너나 지지되는 곡선을 건너뛰지는 않습니다. 초기 코너는 진입/출구 12cm 지지를 요구하고, 이미 검증된 S 경로의 현재 첫 코너를 이어받을 때만 진입 지지를 기존 설정(5cm)으로 검사합니다. 최초 프레임은 PNG도 보관해 JPEG 손실에 따른 재생 차이를 방지합니다. 과거 JPEG 재생 수치는 s_fix_final_v22/first_frame_geometry_trace.json에 있으며 실시간 전체 입력과 동일하다는 증거는 아닙니다.</p>
<p>v28에서는 약 86cm 진행 후 다음 코너에서 정지했습니다. 위치 오차 1.1mm/85% 대응인 현재 조각도 지지 길이 29.78mm가 최소 40mm보다 짧아 거부됐습니다. 또 일반 중앙 목표가 실패하면 약 46cm 앞의 코너 피벗을 너무 일찍 확정하고 있었습니다. 현재 배정된 연결 코너 경계로 유효한 miter 중앙 경로를 만들 수 있을 때에는 그 경로로 먼저 접근합니다. 기존 거리 조건에 도달한 뒤에만 코너를 확정합니다. 단순히 지지 길이를 낮추거나 저장 피벗을 매번 갱신하지 않습니다.</p>
<h2>시험 조건</h2><p>Domain 172 / loopback 통신, 동일 YOLO 모델 320px/conf 0.55, 차로 폭 20cm, 동일 속도(직선 최대 0.1m/s, 한 차선 0.03m/s), 일반 회전 제한 0.15rad/s·코너 회전 제한 0.25rad/s. 흰 픽셀 전용 제어 우회는 사용하지 않습니다. 기존 MetricLaneTracker·CornerPolicy·Pure Pursuit를 사용합니다.</p>
<p>수정 전과 중간 수정(v2) 시험은 최대 65초였습니다. 최종 시험은 최대 200초로 늘렸습니다. 시간이 다르므로 최종 완주율만으로 동일 시간 성능 개선을 주장하지 않습니다. S자의 중간 수정 65초 이동 거리도 별도로 표시합니다. Ground truth는 평가 전용이며 경로 생성에 넣지 않습니다.</p></section>'''
    document = document.replace('{{verification_status}}',
        '이번 단회 시험에서는 네 코스가 완주했습니다. 반복 재현성과 실차 검증은 별도 필요합니다.'
        if passed == 4 and len(records) == 4 else
        '미완료: 확인한 코드 버그는 보완했지만 급 S자 완주 문제는 해결됐다고 판정하지 않습니다. 아래 FAIL 기록과 영상도 그대로 공개합니다. 실차용 성공 버전으로 배포하지 않았습니다.')
    document += f'<section><h2>최종 {passed}/{len(records)} 완주</h2><table><tr><th>코스</th><th>수정 전 65초 이동</th><th>중간 수정 v2 65초 이동</th><th>최종 최대 200초</th><th>최대 중심 이탈</th></tr>'+''.join(rows)+'</table>'
    document += f'<p>회귀 검사 {count}개, 실패/오류 {failed}개. PASS는 끝점 10cm 이내 도달 및 중심 이탈 차로폭/2 이내 조건입니다. 멈춤 없는 주행, 실차 안전성 또는 반복 재현성 인증을 뜻하지 않습니다.</p></section>'
    document += '<section><p>영상은 고정 15fps로 저장하며, 특히 추론 디버그 영상은 원본 카메라보다 프레임이 적어 실제 주행 시간과 재생 시간이 다릅니다. 시간/명령/위치 분석에는 trial.json 로그를 사용합니다. MP4는 H.264/yuv420p/faststart로 변환하고 전체 디코딩을 검사했습니다.</p></section>'
    document += ''.join(sections)+'</main></html>'
    (output/'report.html').write_text(document)
    print(json.dumps(dict(passed=passed, total=len(records), report=str(output/'report.html'))))


if __name__ == '__main__':
    main()
