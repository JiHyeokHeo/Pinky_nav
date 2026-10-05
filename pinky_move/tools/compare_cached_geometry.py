"""Compare baseline/current cold-start geometry on identical cached polygons.

No inference, network, ROS nodes or movement. A target is not course completion.
"""
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinky_move.metric_lane import MetricLaneTracker


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default='1ceed93')
    parser.add_argument('--connected-geometry', action='store_true')
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'show',
        args.baseline+':pinky_move/pinky_move/metric_lane.py'], cwd=ROOT, text=True)
    baseline = types.ModuleType('pinky_move._baseline_metric')
    baseline.__package__ = 'pinky_move'
    exec(compile(source, '<baseline '+args.baseline+'>', 'exec'), baseline.__dict__)
    calibration = json.loads((ROOT/'config/robot_floor_calibration.json').read_text())
    params = yaml.safe_load((ROOT/'config/lane_autonomy.yaml').read_text())['lane_autonomy']['ros__parameters']
    kwargs = dict(lookahead=params['metric_lookahead_m'], lane_width=params['lane_width'],
                  path_min_m=params['metric_path_min_m'], path_max_m=params['metric_path_max_m'])
    results = []
    for path in sorted((ROOT/'reports/dataset_20261001/prediction_cache').glob('*.json')):
        data = json.loads(path.read_text())
        masks = []
        for instance in data['instances']:
            if instance['cls'] != 1:
                continue
            mask = np.zeros((480,640), np.uint8)
            cv2.fillPoly(mask, [np.asarray(instance['points'], np.int32)], 1)
            masks.append(mask)
        row = dict(key=path.stem)
        for name, factory in [('before', baseline.MetricLaneTracker), ('after', MetricLaneTracker)]:
            options = dict(minimum_lane_width_m=.15)
            if name == 'after':
                options['connected_geometry'] = args.connected_geometry
            tracker = factory(**options)
            try:
                target = tracker.update(masks, calibration, 1., **kwargs)
                row[name] = dict(ok=True, x=target['x_m'], y=target['y_m'])
            except ValueError as exc:
                row[name] = dict(ok=False, reason=str(exc))
        results.append(row)
    summary = dict(images=len(results), before=sum(r['before']['ok'] for r in results),
                   after=sum(r['after']['ok'] for r in results),
                   regressions=[r['key'] for r in results if r['before']['ok'] and not r['after']['ok']],
                   improvements=[r['key'] for r in results if not r['before']['ok'] and r['after']['ok']])
    shifts = [((r['after']['x']-r['before']['x'])**2+
               (r['after']['y']-r['before']['y'])**2)**.5
              for r in results if r['before']['ok'] and r['after']['ok']]
    summary['max_existing_target_shift_m'] = max(shifts, default=0.)
    args.output.write_text(json.dumps(dict(summary=summary, results=results), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
