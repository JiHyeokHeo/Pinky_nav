"""PC-side YOLO segmentation ONLY. No ROS publishers or motor commands.

Connects to Pinky2's loopback perception mailbox through authenticated SSH.
Same model weights, image dimensions and class names as the robot deployment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

import numpy as np
from . import lane_wire


def predict_reply(model, model_hash, request, imgsz=320, confidence=.55, iou=.70):
    started = time.monotonic()
    reply = dict(version=1, token=request['token'], width=request['width'],
                 height=request['height'], model_sha256=model_hash)
    try:
        if request['model_sha256'] != model_hash:
            raise ValueError('PC and robot model hashes differ')
        frame = lane_wire.decode_request(request)
        result = model.predict(frame, imgsz=imgsz, conf=confidence, iou=iou,
                               retina_masks=True, device='cpu', verbose=False)[0]
        names = model.names
        instances = []
        if result.masks is not None and result.boxes is not None:
            for class_id, polygon in zip(result.boxes.cls.int().cpu().tolist(), result.masks.xy):
                name = str(names[class_id]).casefold()
                if name not in ('lane', 'crossline'):
                    continue
                # Original-image coordinates, not resized network-input pixels.
                points = np.asarray(polygon, float)
                instances.append(dict(**{'class': name}, points=points.tolist()))
        reply['instances'] = instances
        lane_wire.decode_result(reply, frame.shape[1], frame.shape[0], model_hash)
    except Exception as exc:
        reply['error'] = str(exc)[:200]
    # Duration only; robot never uses PC timestamps to extend input freshness.
    reply['pc_processing_seconds'] = time.monotonic() - started
    return json.dumps(reply).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='pinky@192.168.45.22')
    parser.add_argument('--model', default='/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt')
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--confidence', type=float, default=.55)
    parser.add_argument('--iou', type=float, default=.70)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-results', type=int, default=0,
                        help='0: run until Ctrl+C; otherwise exit after this many replies')
    args = parser.parse_args()
    if (not 1024 <= args.port <= 65535 or not 32 <= args.imgsz <= 1280 or
            not 1 <= args.threads <= 32 or args.max_results < 0 or
            not 0 < args.confidence <= 1 or not 0 < args.iou <= 1 or args.robot.startswith('-')):
        parser.error('invalid inference/connection arguments')
    if not Path(args.model).is_file():
        parser.error('model file does not exist; no automatic download')
    os.environ['YOLO_OFFLINE'] = 'true'
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    model = YOLO(args.model)
    if model.task != 'segment':
        parser.error('a segmentation model is required')
    names = model.names.values() if isinstance(model.names, dict) else model.names
    if not {'lane', 'crossline'} <= {str(n).casefold() for n in names}:
        parser.error('model must contain lane and crossline classes')
    with open(args.model, 'rb') as stream:
        model_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    # Warm up BEFORE pulling a camera frame. Startup cost must not consume its TTL.
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=args.imgsz,
                  device='cpu', verbose=False)
    tunnel = subprocess.Popen([
        'ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=5',
        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
        '-L', f'127.0.0.1:{args.port}:127.0.0.1:{args.port}', args.robot],
        stdin=subprocess.DEVNULL)
    # Ignore HTTP proxy environment variables: all camera data stays in SSH.
    http = build_opener(ProxyHandler({}))
    url = f'http://127.0.0.1:{args.port}'
    count = 0
    last_warning = 0.
    print(f'PC segmentation ready (CPU); robot={args.robot}, model_sha256={model_hash}', flush=True)
    try:
        while tunnel.poll() is None:
            try:
                with http.open(url+'/frame', timeout=2.) as response:
                    if response.status == 204:
                        time.sleep(.02)
                        continue
                    payload = response.read(lane_wire.MAX_MESSAGE+1)
                request = lane_wire.unpack(payload)
                started = time.monotonic()
                reply = predict_reply(model, model_hash, request, args.imgsz, args.confidence, args.iou)
                with http.open(Request(url+'/result', data=reply,
                                       headers={'Content-Type': 'application/json'}), timeout=2.):
                    pass
                count += 1
                print(f"frame={request['token'].split(':')[-1]} inference={time.monotonic()-started:.3f}s "
                      f"instances={len(json.loads(reply).get('instances', []))} "
                      f"error={json.loads(reply).get('error', '')}", flush=True)
                if args.max_results and count >= args.max_results:
                    break
            except (URLError, OSError, ValueError, KeyError) as exc:
                if time.monotonic()-last_warning > 2.:
                    print(f'Waiting/rejected frame: {exc}', flush=True)
                    last_warning = time.monotonic()
                time.sleep(.1)
        if tunnel.poll() is not None:
            raise RuntimeError(f'SSH tunnel exited ({tunnel.returncode}); robot will time out')
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()


if __name__ == '__main__':
    main()
