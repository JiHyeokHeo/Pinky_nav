from threading import Event
from pinky_move.debug_worker import LatestDebugWorker
from pinky_move.debug_worker import IsolatedImageTransport
from multiprocessing import get_context
from types import SimpleNamespace
from array import array
import time


def blocked_image_process(requests, timings, domain_id, topics):
    requests.get()
    topics['entered'].set()
    topics['release'].wait(5.)


def test_blocked_renderer_has_one_latest_slot_and_does_not_block_submit():
    entered, release, newest = Event(), Event(), Event()
    processed = []
    def render(value):
        processed.append(value)
        if value == 1:
            entered.set()
            release.wait(2.)
        if value == 100:
            newest.set()
    worker = LatestDebugWorker(render)
    try:
        worker.submit(1)
        assert entered.wait(1.)
        for value in range(2,101):
            worker.submit(value)
        assert processed == [1]
        assert worker.pending == 100
        release.set()
        assert newest.wait(1.)
        assert processed == [1,100]
    finally:
        release.set()
        worker.close()


def test_renderer_error_does_not_kill_worker():
    done = Event()
    def render(value):
        if value == 'fail':
            done.set()
            raise ValueError('diagnostic only')
    worker = LatestDebugWorker(render)
    try:
        worker.submit('fail')
        assert done.wait(1.)
    finally:
        worker.close()
    assert worker.last_error == 'diagnostic only'


def test_blocked_dds_process_drops_images_without_blocking_producer():
    ctx = get_context('spawn')
    entered, release = ctx.Event(), ctx.Event()
    transport = IsolatedImageTransport(22, dict(entered=entered, release=release),
                                      _worker_target=blocked_image_process)
    message = SimpleNamespace(height=480, width=640, encoding='bgr8',
        is_bigendian=0, step=1920, data=array('B', bytes(640*480*3)),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id='camera'))
    try:
        transport.submit('debug', message)
        assert entered.wait(3.)
        started = time.monotonic()
        for _ in range(100):
            transport.submit('debug', message)
        assert time.monotonic()-started < .5
        assert transport.dropped > 0
    finally:
        release.set()
        transport.close()


def test_isolated_publisher_preserves_image_and_header_on_ros_topic(monkeypatch):
    """Dedicated test domain, diagnostic Image only; never a motor topic."""
    import os
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    # This is a LOCAL two-process transport test. The developer's hardware
    # CycloneDDS profile may list robot peers only and disable multicast;
    # inheriting it makes an unrelated network profile decide this test.
    monkeypatch.setenv('CYCLONEDDS_URI', '<CycloneDDS><Domain><General>'
        '<Interfaces><NetworkInterface name="lo"/></Interfaces>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><Peers>'
        '<Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    context = Context()
    rclpy.init(args=[], domain_id=211, context=context)
    node = Node('lane_image_transport_test', context=context, enable_rosout=False)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    received = []
    topic = '/lane_transport_test/image'
    node.create_subscription(Image, topic, received.append,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    transport = IsolatedImageTransport(211, dict(debug=topic))
    message = Image()
    message.height, message.width, message.step = 4, 5, 15
    message.encoding = 'bgr8'
    message.header.stamp.sec, message.header.stamp.nanosec = 123, 456
    message.header.frame_id = 'base_link'
    message.data = array('B', range(60))
    try:
        assert transport.process.pid != os.getpid()
        deadline = time.monotonic()+8.
        while not received and time.monotonic() < deadline:
            transport.submit('debug', message)
            executor.spin_once(timeout_sec=.05)
        assert received, 'isolated diagnostic publisher did not deliver'
        output = received[-1]
        assert output.header == message.header
        assert output.data == message.data
        assert (output.height, output.width, output.step) == (4,5,15)
    finally:
        transport.close()
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
