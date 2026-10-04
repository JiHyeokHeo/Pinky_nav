"""Offline image/segmentation/geometry audit. No ROS node or motor publisher.

Each image is an independent cold start. Unknown timing/odometry is NOT
fabricated into a passing drive episode. IoU here is class-union pixel IoU,
not instance mAP or a real-world driving success rate.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from pinky_move.metric_lane import MetricLaneTracker
from pinky_move.lane_corner import CornerConfig, CornerPolicy, classify_boundary
from pinky_move.robot_projection import draw_metric_target, validate_calibration


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ground_truth(path, width, height):
    if not path.exists():
        return [], ['missing_label']
    items, errors = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            cls = int(values[0])
            if values[0] != cls or cls not in (0, 1):
                raise ValueError('invalid_class')
            if len(values) == 5:
                raise ValueError('bounding_box_not_segmentation')
            if len(values) < 7 or len(values) % 2 != 1:
                raise ValueError('invalid_polygon_length')
            points = np.asarray(values[1:]).reshape(-1, 2)
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
                raise ValueError('invalid_normalized_coordinates')
            items.append(dict(cls=cls, points=(points*[width, height]).tolist()))
        except (ValueError, IndexError) as exc:
            errors.append(f'line {number}: {exc}')
    return items, errors


def masks_for(items, shape, class_id=1):
    masks = []
    for item in items:
        if item['cls'] != class_id:
            continue
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(item['points'], np.int32)], 1)
        masks.append(mask)
    return masks


def geometry(masks, calibration, params):
    tracker = MetricLaneTracker(minimum_lane_width_m=2*(params['corner_robot_half_width_m']+
                                                       params['corner_clearance_m']))
    kwargs = dict(lookahead=params['metric_lookahead_m'],
                  timeout_s=params['metric_single_line_timeout'], lane_width=params['lane_width'],
                  degree=params['path_polynomial_degree'], max_forward_m=params['projection_max_forward_m'],
                  path_min_m=params['metric_path_min_m'], path_max_m=params['metric_path_max_m'])
    target = None
    result = dict(target_available=False, reason='', unexpected_error=False,
                  corner_kind='NO_OBSERVATION', pivot_candidate=False, pivot_reason='')
    started = time.perf_counter()
    try:
        target = tracker.update(masks, calibration, 0., **kwargs)
        result.update(target_available=True, reason='target_available',
                      x_m=target['x_m'], y_m=target['y_m'],
                      lane_width_m=target['normal_width_m'], inferred=target['inferred'])
    except ValueError as exc:
        result['reason'] = str(exc)
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}', unexpected_error=True)
    observation = tracker.last_observation
    eligible = observation is not None and (target is not None or
                    (len(masks) == 1 and 'center_path curve folds back' in result['reason']))
    if eligible:
        cfg = CornerConfig(staged_turn=True, clearance_m=params['corner_clearance_m'],
                           half_width_m=params['corner_robot_half_width_m'],
                           front_m=params['corner_robot_front_m'], rear_m=params['corner_robot_rear_m'])
        try:
            feature = classify_boundary(observation['curve'], cfg)
            result['corner_kind'] = feature['kind']
            result['corner_angle_deg'] = feature['angle_deg']
            if feature['kind'] == 'CORNER' and 65 <= abs(feature['angle_deg']) <= 115:
                # Pure construction check in a coordinate frame at the origin.
                # NOT three temporal confirmations or a measured odometry pose.
                policy = CornerPolicy(cfg)
                policy._start_staged(observation, feature, np.zeros(3), 0.)
                result['pivot_candidate'] = True
        except ValueError as exc:
            result['pivot_reason'] = str(exc)
        except Exception as exc:
            result.update(pivot_reason=f'{type(exc).__name__}: {exc}', unexpected_error=True)
    result['geometry_ms'] = 1000*(time.perf_counter()-started)
    return result, target


def reason_group(reason):
    for delimiter in (' (', ':'):
        reason = reason.split(delimiter)[0]
    return reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--retry-imgsz', type=int, default=0)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    calibration_path = package/'config/robot_floor_calibration.json'
    calibration = json.loads(calibration_path.read_text())
    validate_calibration(calibration)
    params = yaml.safe_load((package/'config/lane_autonomy.yaml').read_text())['lane_autonomy']['ros__parameters']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'images').mkdir(exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    model_hash = digest(args.model)
    signature = dict(model_sha256=model_hash, imgsz=320, confidence=.55, iou=.70, retina_masks=True)
    if args.retry_imgsz:
        signature['retry_imgsz'] = args.retry_imgsz
    files = [(split, f) for split in ('train', 'valid', 'test')
             for f in sorted((args.dataset/split/'images').iterdir())
             if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    model = None
    rows, detail, tiles = [], [], []
    started_all = time.perf_counter()
    for index, (split, path) in enumerate(files, 1):
        relative = str(path.relative_to(args.dataset))
        key = hashlib.sha256(relative.encode()).hexdigest()[:16]
        row = dict(split=split, image=relative, label_errors='', inference_ms=0., cache_hit=False)
        frame = cv2.imread(str(path))
        if frame is None:
            row.update(input_error='unreadable_image')
            rows.append(row)
            continue
        height, width = frame.shape[:2]
        row.update(width=width, height=height, input_error='')
        if [width, height] != calibration['image_size']:
            row['input_error'] = 'calibration_resolution_mismatch'
            rows.append(row)
            continue
        gt, label_errors = ground_truth(args.dataset/split/'labels'/(path.stem+'.txt'), width, height)
        row['label_errors'] = '; '.join(label_errors)
        cache_path = args.cache/(key+'.json')
        expected = dict(signature, image_sha256=digest(path))
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
        if cached is not None and cached.get('signature') == expected:
            items = cached['instances']
            row.update(inference_ms=cached['inference_ms'], cache_hit=True)
        else:
            if model is None:
                import torch
                from ultralytics import YOLO
                torch.set_num_threads(args.threads)
                model = YOLO(str(args.model))
                if model.task != 'segment':
                    raise RuntimeError('The deployed model must be a segmentation model')
                model.predict(np.zeros_like(frame), imgsz=320, device='cpu', verbose=False)
            tick = time.perf_counter()
            prediction = model.predict(frame, imgsz=320, conf=.55, iou=.70,
                                       retina_masks=True, device='cpu', verbose=False)[0]
            row['inference_ms'] = 1000*(time.perf_counter()-tick)
            items = []
            if prediction.masks is not None:
                for cls, polygon in zip(prediction.boxes.cls.int().cpu().tolist(), prediction.masks.xy):
                    name = str(model.names[cls]).lower()
                    if name in ('lane', 'crossline'):
                        items.append(dict(cls=1 if name == 'lane' else 0, points=polygon.tolist()))
            if args.retry_imgsz > 320 and not any(item['cls'] == 1 for item in items):
                retry = model.predict(frame, imgsz=args.retry_imgsz, conf=.55, iou=.70,
                                      retina_masks=True, device='cpu', verbose=False)[0]
                if retry.masks is not None:
                    for cls, polygon in zip(retry.boxes.cls.int().cpu().tolist(), retry.masks.xy):
                        if str(model.names[cls]).lower() == 'lane':
                            items.append(dict(cls=1, points=polygon.tolist()))
                row['inference_ms'] = 1000*(time.perf_counter()-tick)
            cache_path.write_text(json.dumps(dict(signature=expected, instances=items,
                                                inference_ms=row['inference_ms'])))
        pred_masks = masks_for(items, (height, width))
        gt_masks = masks_for(gt, (height, width))
        row.update(pred_lanes=len(pred_masks), gt_lanes=len(gt_masks))
        evaluation, target = geometry(pred_masks, calibration, params)
        for name, value in evaluation.items():
            row['pred_'+name] = value
        if not label_errors:
            gt_evaluation, _ = geometry(gt_masks, calibration, params)
            for name, value in gt_evaluation.items():
                row['gt_'+name] = value
            for cls, name in ((1, 'lane'), (0, 'crossline')):
                a = np.zeros((height, width), bool)
                b = a.copy()
                for mask in masks_for(items, (height, width), cls): a |= mask.astype(bool)
                for mask in masks_for(gt, (height, width), cls): b |= mask.astype(bool)
                intersection = int(np.count_nonzero(a & b))
                union = int(np.count_nonzero(a | b))
                row[name+'_intersection'] = intersection
                row[name+'_union'] = union
                row[name+'_pixel_iou'] = intersection/union if union else None
        overlay = frame.copy()
        for item in gt:
            cv2.polylines(overlay, [np.asarray(item['points'], np.int32)], True, (0, 255, 0), 1)
        for item in items:
            cv2.polylines(overlay, [np.asarray(item['points'], np.int32)], True, (255, 255, 0), 2)
        if target is not None:
            draw_metric_target(overlay, target, calibration)
        cv2.putText(overlay, 'GT=green  YOLO=cyan  static geometry only', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
        cv2.putText(overlay, evaluation['reason'][:92], (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
        image_name = f'images/{key}.jpg'
        cv2.imwrite(str(args.output/image_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        row['overlay'] = image_name
        rows.append(row)
        detail.append(dict(image=relative, prediction=evaluation, label_errors=label_errors))
        tiles.append(f'<article><a href="{image_name}"><img loading="lazy" src="{image_name}"></a>'
                     f'<p>{html.escape(relative)}<br>{html.escape(evaluation["reason"])}</p></article>')
        if index % 25 == 0 or index == len(files):
            print(f'{index}/{len(files)} elapsed={time.perf_counter()-started_all:.1f}s', flush=True)
    summary = dict(images=len(files), splits={}, signature=signature,
                   calibration_sha256=digest(calibration_path),
                   metric_code_sha256=digest(package/'pinky_move/metric_lane.py'),
                   corner_code_sha256=digest(package/'pinky_move/lane_corner.py'),
                   elapsed_seconds=time.perf_counter()-started_all,
                   limitations=['Static independent cold starts; no odometry or temporal driving test.',
                                'Same resolution does not prove same camera pose/intrinsics.',
                                'Training split is not a held-out generalization estimate.',
                                'Pixel IoU is class-union overlap, not mAP.',
                                'One-sided geometry uses configured width, not measured free space.'])
    for split in ('train', 'valid', 'test', 'all'):
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        result = dict(images=len(selected),
            input_errors=sum(bool(r.get('input_error')) for r in selected),
            label_error_images=sum(bool(r.get('label_errors')) for r in selected),
            pred_target_available=sum(bool(r.get('pred_target_available')) for r in selected),
            gt_target_available=sum(bool(r.get('gt_target_available')) for r in selected),
            pred_no_lane=sum(r.get('pred_lanes') == 0 for r in selected),
            unexpected_errors=sum(bool(r.get('pred_unexpected_error') or r.get('gt_unexpected_error')) for r in selected),
            pivot_candidates=sum(bool(r.get('pred_pivot_candidate')) for r in selected),
            reasons=dict(Counter(reason_group(r.get('pred_reason', r.get('input_error', ''))) for r in selected)),
            corner_kinds=dict(Counter(r.get('pred_corner_kind', 'SKIPPED') for r in selected)))
        for name in ('lane', 'crossline'):
            union = sum(r.get(name+'_union', 0) for r in selected)
            intersection = sum(r.get(name+'_intersection', 0) for r in selected)
            result[name+'_pixel_iou_global'] = intersection/union if union else None
        for name in ('inference_ms', 'pred_geometry_ms'):
            values = [r[name] for r in selected if name in r]
            result[name] = dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                                maximum=float(max(values))) if values else None
        summary['splits'][split] = result
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output/'details.json').write_text(json.dumps(detail, indent=2, ensure_ascii=False))
    with (args.output/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader(); writer.writerows(rows)
    document = ('<!doctype html><meta charset="utf-8"><title>Lane dataset audit</title>'
                '<style>body{font-family:sans-serif;padding:20px;background:#eee} '
                'section{display:flex;flex-wrap:wrap}article{width:420px;margin:10px;background:white;padding:8px}'
                'img{width:100%}p{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
                '<h1>407-frame offline lane audit</h1><p>Green: labels. Cyan: YOLO. '
                'This is not a physical driving success rate. No ROS commands were sent.</p>'
                f'<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>'
                '<section>'+''.join(tiles)+'</section>')
    (args.output/'report.html').write_text(document)
    print(json.dumps(summary['splits'], indent=2), flush=True)


if __name__ == '__main__':
    main()
