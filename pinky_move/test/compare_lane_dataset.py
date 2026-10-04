"""Compare two immutable offline dataset audits, not physical drive outcomes."""
import argparse
import csv
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--before', default='baseline')
    parser.add_argument('--after', default='improved')
    parser.add_argument('--output-stem', default='comparison')
    parser.add_argument('--allow-inference-change', action='store_true')
    args = parser.parse_args()
    root = args.root
    names = (args.before, args.after)
    summaries = [json.loads((root/name/'summary.json').read_text()) for name in names]
    rows = [{r['image']: r for r in csv.DictReader((root/name/'results.csv').open())}
            for name in names]
    a, b = rows
    same_inference = summaries[0]['signature'] == summaries[1]['signature']
    if set(a) != set(b) or (not same_inference and not args.allow_inference_change):
        raise SystemExit('Refusing comparison: dataset membership or inference signature differs')
    good = lambda r: r.get('pred_target_available') == 'True'
    gained = [key for key in a if not good(a[key]) and good(b[key])]
    lost = [key for key in a if good(a[key]) and not good(b[key])]
    unchanged = [key for key in a if good(a[key]) and good(b[key])]
    remaining = [key for key in a if not good(b[key])]
    maximum_delta = max((abs(float(a[key][axis])-float(b[key][axis]))
                         for key in unchanged for axis in ('pred_x_m', 'pred_y_m')), default=0.)
    comparison = dict(images=len(a), gained=len(gained), regressed=len(lost),
                      remaining_no_target=len(remaining), max_existing_target_delta_m=maximum_delta,
                      gains=gained, regressions=lost,
                      label_issues=[dict(image=k, reason=v['label_errors']) for k, v in b.items()
                                    if v.get('label_errors')])
    comparison.update(before=args.before, after=args.after, same_inference=same_inference,
                      signatures=[s['signature'] for s in summaries])
    (root/f'{args.output_stem}.json').write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    table = []
    for split in ('train', 'valid', 'test', 'all'):
        before, after = (s['splits'][split] for s in summaries)
        table.append(f'<tr><td>{split}</td><td>{before["images"]}</td>'
                     f'<td>{before["pred_target_available"]}</td><td>{after["pred_target_available"]}</td>'
                     f'<td>{before["pred_geometry_ms"]["p50"]:.1f} → '
                     f'{after["pred_geometry_ms"]["p50"]:.1f} ms</td></tr>')
    panels = []
    for key in gained:
        panels.append(f'<article><h3>{html.escape(key)}</h3><div class="pair">'
                      f'<figure><img loading="lazy" src="{args.before}/{a[key]["overlay"]}">'
                      f'<figcaption>Before: {html.escape(a[key]["pred_reason"])}</figcaption></figure>'
                      f'<figure><img loading="lazy" src="{args.after}/{b[key]["overlay"]}">'
                      '<figcaption>After: 중앙 목표점 생성</figcaption></figure></div></article>')
    failures = ''.join(f'<tr><td><a href="{args.after}/{b[k]["overlay"]}">{html.escape(k)}</a></td>'
                       f'<td>{html.escape(b[k]["pred_reason"])}</td></tr>' for k in remaining)
    doc = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>407장 차선 기능 검증 · Before / After</title>
<style>body{{max-width:1400px;margin:auto;padding:28px;font-family:sans-serif;line-height:1.6;background:#f3f5f8;color:#182432}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border:1px solid #ddd;text-align:left}}
.pair{{display:flex;gap:10px}}figure{{margin:0;width:50%}}img{{width:100%}}article{{background:white;padding:14px;margin:20px 0}}
h3,td{{overflow-wrap:anywhere}}.warning{{background:#fff2d0;padding:14px}}pre{{white-space:pre-wrap}}</style>
<h1>실주행 이미지 407장: 코드 개선 전후</h1>
<p class="warning">정지 이미지의 독립적인 초기화 조건 검사입니다. 실제 주행 성공률이 아니며,
프레임 연속성·오도메트리·모터·회전 공간·보정 거리 정확도를 검증하지 않습니다.
train 결과는 일반화 성능이 아닙니다. 추론 설정 동일 여부: {same_inference}.
False이면 기하 처리뿐 아니라 추론 변경도 포함한 비교입니다. 설정은 비교 JSON에 기록했습니다.</p>
<h2>결과</h2><table><tr><th>분할</th><th>이미지</th><th>이전 목표점 생성</th><th>개선 후</th><th>기하 계산 중앙값 (PC)</th></tr>{''.join(table)}</table>
<p>개선 {len(gained)}장 · 기존 정상 → 실패 {len(lost)}장 · 남은 목표점 없음 {len(remaining)}장.
기존 정상 목표점 최대 좌표 변화: {maximum_delta:.3g} m. 계산 시간은 동일 PC의 순차 실행 측정이며 로봇 지연 감소를 보장하지 않습니다.</p>
<p>비교 실행: {html.escape(args.before)} → {html.escape(args.after)}.
입력 오류 {summaries[1]['splits']['all']['input_errors']}, 예상 밖 코드 예외 {summaries[1]['splits']['all']['unexpected_errors']}.
형식 문제 라벨은 segmentation 평가에서 제외하고 원본은 수정하지 않았습니다.
단일 차선 경로는 설정된 폭에 의존하므로 목표점 생성이 실제 중앙 정답임을 뜻하지 않습니다.</p>
<h2>개선 이미지 Before / After</h2>{''.join(panels)}
<h2>남아 있는 거부 사례</h2><table><tr><th>이미지</th><th>사유</th></tr>{failures}</table>
<h2>라벨 형식 문제</h2><pre>{html.escape(json.dumps(comparison['label_issues'],indent=2,ensure_ascii=False))}</pre>
<p><a href="{args.before}/report.html">이전 전체 407장</a> · <a href="{args.after}/report.html">개선 후 전체 407장</a> ·
<a href="{args.after}/results.csv">이미지별 CSV</a> · <a href="{args.output_stem}.json">전후 비교 JSON</a></p>
<p>로봇은 오프라인 상태입니다. 이 변경은 Pinky2에 배포하거나 실제 주행시키지 않았습니다.</p></html>'''
    (root/f'{args.output_stem}.html').write_text(doc)
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('gains', 'regressions')}, indent=2))


if __name__ == '__main__':
    main()
