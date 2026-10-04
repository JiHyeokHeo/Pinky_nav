"""Bounded segmentation messages and a loopback-only SSH transport.

Only the approved PC pulls JPEG requests through its authenticated SSH tunnel.
There is no public camera endpoint and no motion/enable endpoint. Remote replies
contain pixel polygons and, when requested, a metric plan; never velocity
commands. The robot owns freshness, arrival and motion limits. The PC's clock
is not trusted. Plan identity reuses the single-use frame token and epoch.
"""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock, Thread
from types import SimpleNamespace

import cv2
import numpy as np

MAX_MESSAGE = 2_000_000


def unpack(payload):
    if len(payload) > MAX_MESSAGE:
        raise ValueError('perception message too large')
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('unsupported perception protocol')
    if not isinstance(data.get('token'), str) or not 1 <= len(data['token']) <= 96:
        raise ValueError('invalid frame token')
    return data


def encode_request(frame, token, header, model_sha256):
    height, width = frame.shape[:2]
    if not (16 <= width <= 1920 and 16 <= height <= 1080):
        raise ValueError('unsupported image size')
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('JPEG encode failed')
    payload = json.dumps(dict(version=1, token=token, width=width, height=height,
                              capture_sec=header.stamp.sec,
                              capture_nanosec=header.stamp.nanosec,
                              frame_id=header.frame_id, model_sha256=model_sha256,
                              jpeg=base64.b64encode(jpeg).decode('ascii'))).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError('JPEG request too large')
    return payload


def decode_request(data):
    width, height = data.get('width'), data.get('height')
    if type(width) is not int or type(height) is not int or not (16 <= width <= 1920 and 16 <= height <= 1080):
        raise ValueError('invalid image dimensions')
    jpeg = base64.b64decode(data['jpeg'], validate=True)
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (height, width):
        raise ValueError('JPEG dimensions disagree')
    return frame


class ClassIds(list):
    """Adapter to the existing YOLO-result input, without importing torch."""
    def int(self): return self
    def cpu(self): return self
    def tolist(self): return list(self)


def decode_result(data, width, height, model_sha256):
    if data.get('model_sha256') != model_sha256:
        raise ValueError('remote model differs from robot model')
    if data.get('error'):
        raise ValueError('PC inference failed: '+str(data['error'])[:200])
    if data.get('width') != width or data.get('height') != height:
        raise ValueError('result dimensions disagree')
    instances = data.get('instances')
    if not isinstance(instances, list) or len(instances) > 32:
        raise ValueError('invalid instance count')
    classes, polygons = ClassIds(), []
    for instance in instances:
        if not isinstance(instance, dict) or instance.get('class') not in ('lane', 'crossline'):
            raise ValueError('invalid lane class')
        points = np.asarray(instance.get('points'), dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 4096 or
                not np.isfinite(points).all() or np.any(points < 0) or
                np.any(points[:, 0] > width) or np.any(points[:, 1] > height)):
            raise ValueError('invalid segmentation polygon')
        classes.append(1 if instance['class'] == 'lane' else 0)
        polygons.append(points)
    return SimpleNamespace(boxes=SimpleNamespace(cls=classes),
                           masks=SimpleNamespace(xy=polygons))


class PerceptionMailbox:
    """Single bounded request/reply slots. Socket threads never touch ROS state."""
    def __init__(self):
        self.lock = Lock()
        self.clear()

    def clear(self):
        with self.lock:
            self.token = None
            self.request = None
            self.reply = None

    def offer(self, token, request):
        with self.lock:
            self.token, self.request, self.reply = token, request, None

    def take_request(self):
        with self.lock:
            request, self.request = self.request, None
            return request

    def submit(self, payload):
        data = unpack(payload)
        with self.lock:
            if data['token'] != self.token or self.reply is not None:
                return False
            self.reply = data
            return True

    def take_reply(self):
        with self.lock:
            reply = self.reply
            if reply is not None:
                self.reply, self.token, self.request = None, None, None
            return reply


class LoopbackPerceptionServer:
    """Reachable only on the robot itself or through an authorized SSH tunnel."""
    def __init__(self, mailbox, port=18765):
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(1.)

            def log_message(self, *args):
                pass

            def send(self, code, payload=b''):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def do_GET(self):
                if self.path != '/frame':
                    self.send(404); return
                payload = mailbox.take_request()
                self.send(200 if payload is not None else 204, payload or b'')

            def do_POST(self):
                if self.path != '/result':
                    self.send(404); return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= MAX_MESSAGE:
                        self.send(413); return
                    accepted = mailbox.submit(self.rfile.read(length))
                    self.send(200 if accepted else 409)
                except (ValueError, TypeError, OSError):
                    self.send(400)

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever,
                             kwargs={'poll_interval': .1}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.)
