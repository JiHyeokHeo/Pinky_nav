"""동일 실사진/캐시로 개선된 목표 좌표와 Before/After 그림을 저장한다."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import cv2
import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from pinky_move.metric_lane import MetricLaneTracker
from pinky_move.robot_projection import draw_metric_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.root/'dataset_regression.json').read_text())
    params = yaml.safe_load((PACKAGE/'config/lane_autonomy.yaml').read_text())['lane_autonomy']['ros__parameters']
    kwargs = dict(lookahead=params['metric_lookahead_m'], lane_width=params['lane_width'],
                  path_min_m=params['metric_path_min_m'], path_max_m=params['metric_path_max_m'])
    calibration = json.loads((PACKAGE/'config/robot_floor_calibration.json').read_text())
    source = subprocess.check_output(['git', 'show', '1579af9:pinky_move/pinky_move/metric_lane.py'], cwd=PACKAGE, text=True)
    baseline = types.ModuleType('pinky_move._connected_before')
    baseline.__package__ = 'pinky_move'
    exec(compile(source, '<baseline 1579af9>', 'exec'), baseline.__dict__)
    files = {hashlib.sha256(str(p.relative_to(args.dataset)).encode()).hexdigest()[:16]: p
             for split in ('train', 'test', 'valid')
             for p in (args.dataset/split/'images').iterdir()}
    directory = args.root/'offline_gains'
    directory.mkdir(exist_ok=True)
    records = []
    for key in comparison['summary']['improvements']:
        cache = json.loads((PACKAGE/'reports/dataset_20261001/prediction_cache'/(key+'.json')).read_text())
        path = files[key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cache['signature']['image_sha256']
        frame = cv2.imread(str(path))
        masks, polygons = [], []
        for item in cache['instances']:
            if item['cls'] != 1:
                continue
            polygon = np.asarray(item['points'], np.int32)
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            masks.append(mask)
            polygons.append(polygon)
        record = dict(key=key, image=str(path.relative_to(args.dataset)))
        for phase, factory in [('before', baseline.MetricLaneTracker), ('after', MetricLaneTracker)]:
            options = dict(minimum_lane_width_m=.15)
            if phase == 'after':
                options['connected_geometry'] = True
            tracker = factory(**options)
            display = frame.copy()
            for polygon in polygons:
                cv2.polylines(display, [polygon], True, (255, 180, 0), 2)
            try:
                target = tracker.update(masks, calibration, 1., **kwargs)
                draw_metric_target(display, target, calibration)
                record[phase] = dict(ok=True, x_m=target['x_m'], y_m=target['y_m'],
                                     side=target.get('visible_side'), warning=target.get('center_path_warning'))
                text = f"OFFLINE {phase}: TARGET x={target['x_m']:.3f} y={target['y_m']:.3f}m"
            except ValueError as exc:
                record[phase] = dict(ok=False, reason=str(exc))
                text = f'OFFLINE {phase}: NO TARGET'
            cv2.putText(display, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
            cv2.imwrite(str(directory/(key+'_'+phase+'.jpg')), display)
        records.append(record)
    (directory/'coordinates.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
