#!/usr/bin/env python3
"""로봇이 보내는 카메라 영상을 PC 화면에 띄우고 파일로 녹화한다.

    python3 cam_view.py                 # 보기만
    python3 cam_view.py --record        # 실행하자마자 녹화 시작
    python3 cam_view.py --no-window --record --duration 60
                                        # 창 없이 60초 녹화 (SSH 에서 유용)

로봇에서 cam_stream.py 가 돌고 있어야 한다.

창에서:
    q 또는 ESC  종료
    r           녹화 시작/중지
    s           현재 프레임을 사진으로 저장
    f           화면에 겹쳐 보이는 정보 끄기/켜기

영상 방향은 로봇의 cam_stream.py 에서 바로잡는 것이 원칙이다.
그래야 녹화/사진까지 모두 제대로 나온다. 여기 --rotate/--flip 은
로봇 쪽을 고칠 수 없을 때 쓰는 보조 수단이다.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOPIC = '/pinky/camera/image_raw/compressed'
DISCOVERY_SEC = 20.0
STALL_WARN_SEC = 3.0
FALLBACK_FPS = 15.0
WINDOW = 'pinky camera'

# 우선 mp4 로 시도하고, 코덱이 없으면 어디서나 되는 MJPG/avi 로 물러난다
CODECS = [('mp4v', '.mp4'), ('MJPG', '.avi')]

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
FLIPS = {'none': None, 'h': 1, 'v': 0, 'both': -1}


def orient(frame, rotate, flip):
    """회전 -> 뒤집기 순서로 방향을 바로잡는다."""
    code = ROTATIONS[rotate]
    if code is not None:
        frame = cv2.rotate(frame, code)
    code = FLIPS[flip]
    if code is not None:
        frame = cv2.flip(frame, code)
    return frame


class YoloOverlay:
    """선택한 Ultralytics 모델의 검출/세그멘테이션을 화면에 표시한다."""

    def __init__(self, model_path, confidence, image_size, every, calibration_path=None):
        self.model = None
        self.confidence = confidence
        self.image_size = image_size
        self.every = every
        self.frame_count = 0
        self.last_view = None
        self.distance_text = ''
        self.center_hit = None
        self.status = 'YOLO off'
        self.ground_distance = (GroundDistance(calibration_path)
                                if model_path and calibration_path else None)

        if not model_path:
            return

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                'YOLO 표시에는 ultralytics가 필요합니다. '
                '실행 환경에 설치한 뒤 다시 실행하세요.') from error

        self.model = YOLO(model_path)
        classes = ', '.join(str(name) for name in self.model.names.values())
        self.status = f'YOLO: {classes}'
        if self.ground_distance is not None:
            self.status += ' | 바닥 거리 보정 사용'
        print(f'YOLO 모델 로드: {model_path} ({classes})', flush=True)

    def annotate(self, frame):
        """지정 주기마다 추론하고, 중간 프레임은 마지막 결과를 표시한다."""
        if self.model is None:
            return frame

        self.frame_count += 1
        if self.last_view is None or self.frame_count % self.every == 0:
            result = self.model.predict(
                frame, conf=self.confidence, imgsz=self.image_size, verbose=False)[0]
            self.last_view = result.plot()
            if self.ground_distance is not None:
                self.distance_text, self.center_hit = self.ground_distance.from_result(result)
                height, width = self.last_view.shape[:2]
                center_x = width // 2
                cv2.line(self.last_view, (center_x, 0), (center_x, height - 1),
                         (0, 0, 255), 1, cv2.LINE_AA)
                if self.center_hit is not None:
                    point = tuple(np.rint(self.center_hit).astype(int))
                    # 화면 하단(로봇 바로 앞)부터 교점까지가 숫자로 표시한 거리 구간이다.
                    cv2.arrowedLine(self.last_view, (center_x, height - 1), point,
                                    (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.035)
                    cv2.circle(self.last_view, point, 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(self.last_view, point, 10, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                self.distance_text = ''
                self.center_hit = None
        return self.last_view.copy()


class GroundDistance:
    """차선 마스크의 가까운 접점을 보정된 바닥 평면 거리로 변환한다."""

    def __init__(self, calibration_path):
        try:
            with open(calibration_path, encoding='utf-8') as file:
                data = json.load(file)
            self.matrix = np.asarray(data['image_to_ground_homography'], dtype=np.float64)
            self.image_size = tuple(data['image_size'])
            self.range_cm = tuple(data.get('recommended_forward_range_cm', (0, float('inf'))))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f'거리 보정 파일을 읽지 못했습니다: {calibration_path}') from error

        if self.matrix.shape != (3, 3):
            raise RuntimeError(f'거리 보정 행렬 형식이 올바르지 않습니다: {calibration_path}')
        print(f'바닥 거리 보정 로드: {calibration_path} '
              f'(권장 {self.range_cm[0]:.1f}~{self.range_cm[1]:.1f} cm)', flush=True)

    def from_result(self, result):
        """중앙 세로선과 만나는 Lane 마스크의 가장 가까운 바닥 접점 거리."""
        if result.masks is None or result.boxes is None:
            return '중심선 차선 거리: 검출 없음', None

        names = result.names
        source_height, source_width = result.orig_shape
        center_x = source_width / 2.0
        scale = np.array([self.image_size[0] / source_width,
                          self.image_size[1] / source_height])
        closest_cm = None
        closest_pixel = None
        for polygon, cls in zip(result.masks.xy, result.boxes.cls.tolist()):
            if str(names[int(cls)]).lower() not in ('lane', 'left_lane', 'right_lane'):
                continue
            hit = self._centerline_intersection(polygon, center_x)
            if hit is None:
                continue

            calibrated_pixel = hit * scale
            point = cv2.perspectiveTransform(
                calibrated_pixel.reshape(1, 1, 2), self.matrix)[0, 0]
            forward_cm = float(point[1])
            if np.isfinite(forward_cm) and forward_cm > 0:
                if closest_cm is None or forward_cm < closest_cm:
                    closest_cm = forward_cm
                    closest_pixel = hit

        if closest_cm is None:
            return '중심선 차선 거리: 중앙선과 만나는 Lane 없음', None
        low, high = self.range_cm
        suffix = '' if low <= closest_cm <= high else ' (보정 범위 밖)'
        return f'중심선 차선 거리(추정): {closest_cm:.1f} cm{suffix}', closest_pixel

    @staticmethod
    def _centerline_intersection(polygon, center_x):
        """다각형 경계와 x=center_x의 교점 중 영상에서 가장 아래 점을 반환한다."""
        polygon = np.asarray(polygon, dtype=np.float64)
        if len(polygon) < 2:
            return None
        hits = []
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            x1, y1 = first
            x2, y2 = second
            if x1 == x2:
                if np.isclose(x1, center_x):
                    hits.extend((first, second))
            elif min(x1, x2) <= center_x <= max(x1, x2):
                ratio = (center_x - x1) / (x2 - x1)
                hits.append(np.array([center_x, y1 + ratio * (y2 - y1)]))
        return max(hits, key=lambda point: point[1]) if hits else None



class Recorder:
    """들어오는 프레임을 동영상 파일로 저장한다."""

    def __init__(self, out_dir, name=None, fps=None):
        self.out_dir = out_dir
        self.name = name
        self.fps = fps
        self.writer = None
        self.path = None
        self.size = None
        self.frames = 0
        self.started_at = None

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame, measured_fps):
        os.makedirs(self.out_dir, exist_ok=True)
        height, width = frame.shape[:2]
        self.size = (width, height)

        # VideoWriter 는 fps 를 만들 때 고정해야 한다.
        # 실제 수신 속도와 다르면 재생 속도가 어긋나므로 측정값을 우선 쓴다.
        fps = self.fps or (measured_fps if measured_fps >= 1.0 else FALLBACK_FPS)
        self.fps_used = float(fps)

        stem = self.name or datetime.now().strftime('rec_%Y%m%d_%H%M%S')
        if self.name and any(os.path.exists(os.path.join(self.out_dir, stem + ext))
                             for _, ext in CODECS):
            raise RuntimeError(f'같은 이름의 녹화 파일이 이미 있습니다: {stem}')
        suffix = 2
        original_stem = stem
        while any(os.path.exists(os.path.join(self.out_dir, stem + ext))
                  for _, ext in CODECS):
            stem = f'{original_stem}_{suffix}'
            suffix += 1
        for fourcc_name, ext in CODECS:
            path = os.path.join(self.out_dir, stem + ext)
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*fourcc_name), self.fps_used, self.size)
            if writer.isOpened():
                self.writer = writer
                self.path = path
                self.codec = fourcc_name
                break
            writer.release()

        if self.writer is None:
            raise RuntimeError('동영상 파일을 만들지 못했습니다 (코덱 없음).')

        self.frames = 0
        self.started_at = time.time()
        return self.path

    def write(self, frame):
        if self.writer is None:
            return True
        if frame.shape[1::-1] != self.size:
            return False          # 해상도가 바뀌면 이어 쓸 수 없다
        self.writer.write(frame)
        self.frames += 1
        return True

    def stop(self):
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        elapsed = time.time() - self.started_at
        info = (self.path, self.frames, elapsed,
                os.path.getsize(self.path) / 1024 / 1024 if os.path.exists(self.path) else 0)
        self.path = None
        return info


class CamView(Node):
    def __init__(self, topic, rotate=0, flip='none', transport='compressed'):
        super().__init__('cam_view')
        self.rotate = rotate
        self.flip = flip
        self.frame = None
        self.received = 0
        self.last_frame_time = None
        self.fps = 0.0
        self._times = []
        self.transport = transport

        self.create_subscription(
            Image if transport == 'raw' else CompressedImage,
            topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f'{topic} 구독 중...')

    def on_image(self, msg):
        buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if self.transport == 'raw':
            channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(msg.encoding)
            if (channels is None or msg.step < msg.width * channels or
                    buffer.size < msg.height * msg.step):
                self.get_logger().warn('Unsupported or malformed raw Image')
                return
            # Respect row padding; debug_image is bgr8, no second inference.
            frame = buffer[:msg.height*msg.step].reshape(msg.height, msg.step)
            frame = frame[:, :msg.width*channels].reshape(msg.height, msg.width, channels).copy()
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('프레임을 디코딩하지 못했습니다.')
            return

        self.frame = orient(frame, self.rotate, self.flip)
        self.received += 1
        now = time.time()
        self.last_frame_time = now

        self._times.append(now)
        while self._times and now - self._times[0] > 1.0:
            self._times.pop(0)
        self.fps = len(self._times)


def overlay(frame, node, recorder, show_info, vision_status='', distance_text=''):
    if not show_info and not recorder.active:
        return frame

    view = frame.copy()
    height, width = view.shape[:2]

    if show_info:
        text = f'{width}x{height}  {node.fps:.0f} fps  수신 {node.received}'
        header_height = 26 + (20 if vision_status else 0) + (20 if distance_text else 0)
        cv2.rectangle(view, (0, 0), (width, header_height), (0, 0, 0), -1)
        cv2.putText(view, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if vision_status:
            cv2.putText(view, vision_status, (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 255, 255), 1, cv2.LINE_AA)
        if distance_text:
            cv2.putText(view, distance_text, (8, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 0), 1, cv2.LINE_AA)

    if recorder.active:
        elapsed = time.time() - recorder.started_at
        label = f'REC {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}'
        cv2.circle(view, (width - 96, 13), 6, (0, 0, 255), -1)
        cv2.putText(view, label, (width - 82, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return view


def report_stop(info):
    path, frames, elapsed, size_mb = info
    print(f'녹화 종료: {path}', flush=True)
    print(f'  {frames} 프레임, {elapsed:.1f} 초, {size_mb:.1f} MB', flush=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='pinky 카메라 영상 보기 + 녹화 (PC 에서 실행).')
    parser.add_argument('--topic', default=DEFAULT_TOPIC,
                        help=f'구독할 토픽 (기본 {DEFAULT_TOPIC})')
    parser.add_argument('--out', default=HERE, help='사진/동영상을 저장할 폴더')
    parser.add_argument('--transport', choices=['compressed', 'raw'], default='compressed',
                        help='debug_image 토픽은 raw 사용; YOLO 재추론 없이 결과 표시')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='화면에 띄울 때의 배율')
    parser.add_argument('--rotate', type=int, default=0,
                        choices=sorted(ROTATIONS), metavar='{0,90,180,270}',
                        help='받은 영상을 시계 방향으로 회전 (기본 0). '
                             '방향 교정은 로봇 쪽에서 하는 것이 낫다.')
    parser.add_argument('--flip', default='none', choices=sorted(FLIPS),
                        help='회전 뒤 추가로 뒤집기: h 좌우, v 상하, both 둘 다 (기본 none)')
    parser.add_argument('--record', action='store_true',
                        help='실행하자마자 녹화를 시작한다.')
    parser.add_argument('--name', default=None,
                        help='동영상 파일 이름 (확장자 제외). 기본은 시각.')
    parser.add_argument('--record-fps', type=float, default=None,
                        help='녹화 파일의 fps. 기본은 실제 수신 속도를 측정해 쓴다.')
    parser.add_argument('--duration', type=float, default=0,
                        help='이 시간만큼 녹화하고 종료한다 [s]. 0 이면 수동.')
    parser.add_argument('--no-window', action='store_true',
                        help='창을 띄우지 않는다 (SSH 등 화면 없는 환경).')
    parser.add_argument('--model', default=None,
                        help='YOLO 모델(.pt) 경로. 지정하면 검출/세그멘테이션을 화면에 겹친다.')
    parser.add_argument('--model-conf', type=float, default=0.35,
                        help='YOLO 신뢰도 임계값 (기본 0.35)')
    parser.add_argument('--model-imgsz', type=int, default=640,
                        help='YOLO 추론 이미지 크기 (기본 640)')
    parser.add_argument('--model-every', type=int, default=2,
                        help='YOLO를 몇 프레임마다 실행할지 (기본 2)')
    parser.add_argument('--calibration', default=os.path.join(HERE, 'calibration.json'),
                        help='바닥 거리 보정 JSON. 빈 문자열로 지정하면 거리 표시를 끈다.')
    args = parser.parse_args(argv)
    if not 0.0 <= args.model_conf <= 1.0:
        parser.error('--model-conf 는 0~1 범위여야 합니다.')
    if args.model_every < 1:
        parser.error('--model-every 는 1 이상이어야 합니다.')
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.no_window and not args.record:
        print('[실패] --no-window 는 --record 와 함께 써야 의미가 있습니다.',
              file=sys.stderr)
        return 1

    rclpy.init()
    node = CamView(args.topic, args.rotate, args.flip, args.transport)
    recorder = Recorder(args.out, args.name, args.record_fps)
    vision = YoloOverlay(args.model, args.model_conf, args.model_imgsz,
                         args.model_every, args.calibration)
    try:
        if node.count_publishers('/clock') > 0:
            node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        deadline = time.time() + DISCOVERY_SEC
        while time.time() < deadline and node.count_publishers(args.topic) == 0:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.count_publishers(args.topic) == 0:
            domain = os.environ.get('ROS_DOMAIN_ID', '(미설정 = 0)')
            names = sorted(node.get_node_names())
            print(f'\n[실패] {args.topic} 를 보내는 노드가 없습니다.\n'
                  f'  ROS_DOMAIN_ID = {domain}\n'
                  f'  보이는 노드 {len(names)}개: {", ".join(names) or "(없음)"}\n'
                  f'  - 노드가 이것 하나뿐이면 ROS_DOMAIN_ID 를 로봇과 맞추세요.\n'
                  f'  - 로봇에서 cam_stream.py 를 실행했는지 확인하세요.\n', file=sys.stderr)
            return 1

        if not args.no_window:
            print('창이 뜨면  q 종료 / r 녹화 / s 사진 / f 정보표시', flush=True)

        show_info = True
        warned = False
        pending_record = args.record
        last_written_frame = 0

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.frame is None:
                continue

            # 녹화는 첫 프레임이 온 뒤에야 시작할 수 있다 (해상도를 알아야 하므로)
            if pending_record and not recorder.active:
                path = recorder.start(node.frame, node.fps)
                print(f'녹화 시작: {path}  ({recorder.codec}, '
                      f'{recorder.fps_used:.0f} fps)', flush=True)
                pending_record = False

            if recorder.active and node.received != last_written_frame:
                if not recorder.write(node.frame):
                    node.get_logger().warn('해상도가 바뀌어 녹화를 멈춥니다.')
                    report_stop(recorder.stop())
                last_written_frame = node.received

            stalled = time.time() - node.last_frame_time > STALL_WARN_SEC
            if stalled and not warned:
                node.get_logger().warn('영상이 끊겼습니다. 로봇 쪽을 확인하세요.')
                warned = True
            elif not stalled:
                warned = False

            if args.duration > 0 and recorder.active:
                if time.time() - recorder.started_at >= args.duration:
                    report_stop(recorder.stop())
                    break

            if args.no_window:
                continue

            view = vision.annotate(node.frame)
            view = overlay(view, node, recorder, show_info, vision.status,
                           vision.distance_text)
            if args.scale != 1.0:
                view = cv2.resize(view, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('f'):
                show_info = not show_info
            if key == ord('r'):
                if recorder.active:
                    report_stop(recorder.stop())
                else:
                    pending_record = True
            if key == ord('s'):
                os.makedirs(args.out, exist_ok=True)
                path = os.path.join(
                    args.out, datetime.now().strftime('cam_%Y%m%d_%H%M%S.jpg'))
                cv2.imwrite(path, node.frame)
                print(f'사진 저장: {path}', flush=True)

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f'\n[실패] {exc}\n', file=sys.stderr)
        return 1
    finally:
        if recorder.active:          # 어떻게 끝나든 파일은 온전히 닫는다
            report_stop(recorder.stop())
        cv2.destroyAllWindows()
        print(f'총 {node.received} 프레임 수신')
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
