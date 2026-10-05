"""Latest-only diagnostic work; never holds up the control executor."""
from threading import Condition, Thread
import time
from multiprocessing import get_context
from queue import Empty, Full


def _publish_images(requests, timings, domain_id, topics):
    """A separate interpreter/DDS participant; never creates cmd_vel output."""
    from array import array
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    rclpy.init(args=[], domain_id=domain_id)
    node = Node('lane_debug_images', enable_rosout=False,
                start_parameter_services=False)
    pubs = {key: node.create_publisher(Image, topic, QoSProfile(
                depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            for key, topic in topics.items()}
    try:
        while True:
            item = requests.get()
            if item is None:
                break
            key, metadata, payload = item
            started = time.monotonic()
            msg = Image()
            (msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.step,
             msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id) = metadata
            # array assignment avoids per-byte Python validation/conversion.
            data = array('B')
            data.frombytes(payload)
            msg.data = data
            pubs[key].publish(msg)
            try:
                timings.put_nowait(time.monotonic()-started)
            except Full:
                pass
    except KeyboardInterrupt:
        pass  # Launch signals the process group, including this diagnostic child.
    finally:
        node.destroy_node()
        rclpy.try_shutdown()  # The installed ROS signal handler may have shut it down.


class IsolatedImageTransport:
    """Drop diagnostic frames on congestion instead of blocking control.

    A thread alone still shares the Python GIL and DDS participant with camera
    callbacks. The spawned publisher isolates both, with a bounded IPC queue.
    ROS safety checks and perception transport stay in the original process.
    """
    def __init__(self, domain_id, topics, _worker_target=_publish_images):
        ctx = get_context('spawn')
        self.requests, self.timings = ctx.Queue(maxsize=2), ctx.Queue(maxsize=1)
        self.closed, self.dropped, self.last_seconds = False, 0, 0.
        self.process = ctx.Process(target=_worker_target,
            args=(self.requests, self.timings, domain_id, topics), daemon=True,
            name='lane-debug-images')
        self.process.start()

    def submit(self, key, message):
        if self.closed:
            return
        try:
            while True:
                self.last_seconds = self.timings.get_nowait()
        except Empty:
            pass
        h = message.header
        metadata = (message.height, message.width, message.encoding,
                    message.is_bigendian, message.step, h.stamp.sec,
                    h.stamp.nanosec, h.frame_id)
        payload = message.data.tobytes()
        try:
            self.requests.put_nowait((key, metadata, payload))
        except Full:
            self.dropped += 1

    def close(self):
        self.closed = True
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        self.process.join(timeout=.3)
        if self.process.is_alive():
            self.process.terminate()  # Only our isolated diagnostic child.
            self.process.join(timeout=1.)
        # Do not wait for a feeder flushing large images to a stopped child.
        for queue in (self.requests, self.timings):
            queue.cancel_join_thread()
            queue.close()


class DiagnosticPublisher:
    """Keep graph subscriber checks local, but route image writes over IPC."""
    def __init__(self, graph_publisher, transport, key):
        self.graph_publisher, self.transport, self.key = graph_publisher, transport, key

    def get_subscription_count(self):
        return self.graph_publisher.get_subscription_count()

    def publish(self, message):
        self.transport.submit(self.key, message)


class LatestDebugWorker:
    def __init__(self, render):
        self.render = render
        self.condition = Condition()
        self.pending = None
        self.closed = False
        self.last_seconds = 0.
        self.last_error = None
        self.thread = Thread(target=self._run, daemon=True, name='lane-debug')
        self.thread.start()

    def submit(self, job):
        with self.condition:
            if not self.closed:
                self.pending = job  # Replace, never build an unbounded queue.
                self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                job, self.pending = self.pending, None
            started = time.monotonic()
            try:
                self.render(job)
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
            self.last_seconds = time.monotonic()-started

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=2.)
