"""Motor-free replay of saved photographs (not the original live masks).

Cache regenerated YOLO polygons so before/after geometry uses identical input.
Repeated still frames test confirmation only, not real temporal tracking.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinky_move.metric_lane import MetricLaneTracker, floor_curves, near_pixel_side
from pinky_move.robot_projection import draw_metric_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    params = yaml.safe_load((ROOT/'config/lane_autonomy.yaml').read_text())['lane_autonomy']['ros__parameters']
    calibration = json.loads((ROOT/'config/robot_floor_calibration.json').read_text())
    model = None
    if args.infer:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2)
        model = YOLO(params['model_path'])
    records = []
    for directory in sorted((ROOT/'reports/cases').iterdir()):
        if not directory.is_dir():
            continue
        photo = directory/'camera_raw.jpg'
        cache = directory/'replay_masks.json'
        frame = cv2.imread(str(photo))
        if model is not None:
            result = model.predict(frame, imgsz=params['imgsz'], conf=params['confidence'],
                                   iou=params['iou'], retina_masks=True, verbose=False)[0]
            instances = [] if result.masks is None else [
                dict(cls=int(cls), points=polygon.tolist())
                for cls, polygon in zip(result.boxes.cls.cpu().numpy(), result.masks.xy)]
            cache.write_text(json.dumps(dict(provenance='regenerated from raw JPEG; not live masks',
                                            instances=instances)))
        data = json.loads(cache.read_text())
        masks = []
        for instance in data['instances']:
            if instance['cls'] != 1:
                continue
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(instance['points'], np.int32)], 1)
            masks.append(mask)
        kwargs = dict(lookahead=params['metric_lookahead_m'], lane_width=params['lane_width'],
                      path_min_m=params['metric_path_min_m'], path_max_m=params['metric_path_max_m'])
        tracker = MetricLaneTracker(tracking_gap_s=params['lane_tracking_gap_seconds'])
        steps = []
        last_target = None
        for now in (1., 1.2, 1.4, 1.6):
            try:
                target = tracker.update(masks, calibration, now, **kwargs)
                last_target = target
                steps.append(dict(target_available=True, x_m=target['x_m'], y_m=target['y_m'],
                                  visible_side=target.get('visible_side'),
                                  pair_fallback=target.get('pair_fallback')))
            except ValueError as exc:
                steps.append(dict(target_available=False, reason=str(exc)))
        if args.output.stem.endswith('after'):
            overlay = frame.copy()
            for instance in data['instances']:
                cv2.polylines(overlay, [np.asarray(instance['points'], np.int32)],
                              True, (255, 255, 0), 2)
            if last_target is not None:
                draw_metric_target(overlay, last_target, calibration)
            cv2.putText(overlay, 'OFFLINE REPLAY ONLY - regenerated masks / no driving',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
            cv2.putText(overlay, str(steps[-1])[:110], (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
            cv2.imwrite(str(directory/'replay_after.jpg'), overlay)
        diagnostics = []
        for mask in masks:
            curves = floor_curves([mask], calibration, 2., .14, .48, 3)
            diagnostics.append(dict(pixel_side=near_pixel_side(mask), bottom=int(np.nonzero(mask)[0].max()),
                                    curves=[curve.tolist() for curve in curves]))
        records.append(dict(case=directory.name, lane_instances=len(masks), steps=steps,
                            diagnostics=diagnostics))
    args.output.write_text(json.dumps(records, indent=2))
    print(json.dumps([{k: v for k, v in record.items() if k != 'diagnostics'} for record in records], indent=2))


if __name__ == '__main__':
    main()
