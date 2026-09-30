"""Protocol and motor-free controller tests, including stale/replayed replies."""
import json
from types import SimpleNamespace
from urllib.request import build_opener, ProxyHandler, Request
from urllib.error import HTTPError

import numpy as np
import pytest
from std_msgs.msg import Header
from std_srvs.srv import SetBool

from pinky_move import lane_wire
from pinky_move.lane_autonomy import DriveState
from pinky_move.lane_inference_worker import predict_reply
from test_lane_controller import controller


def reply(token='session:1', **updates):
    value = dict(version=1, token=token, width=200, height=100, model_sha256='weights',
                 instances=[{'class': 'lane', 'points': [[48,50],[52,50],[52,99],[48,99]]},
                            {'class': 'lane', 'points': [[148,50],[152,50],[152,99],[148,99]]}])
    value.update(updates)
    return value


def prepare(node):
    node.remote_inference = True
    node.remote_model_sha256 = 'weights'
    node.remote_session = 'session'
    node.remote_sequence = 0
    node.remote_mailbox = lane_wire.PerceptionMailbox()
    node.parameters['result_timeout'] = .8
    node.get_clock = lambda: node.safety_clock
    header = Header(stamp=node.safety_clock.now().to_msg(), frame_id='front_camera_link')
    frame = np.zeros((100,200,3), np.uint8)
    node._image_callback(node._bgr_to_message(frame, header))
    node._start_remote_inference(node.safety_clock.now())
    request = lane_wire.unpack(node.remote_mailbox.take_request())
    return request


def test_jpeg_request_preserves_geometry_and_capture_metadata():
    frame = np.zeros((100,200,3), np.uint8)
    header = Header(frame_id='camera'); header.stamp.sec = 123
    data = lane_wire.unpack(lane_wire.encode_request(frame, 'session:1', header, 'weights'))
    assert data['capture_sec'] == 123 and data['frame_id'] == 'camera'
    assert lane_wire.decode_request(data).shape == frame.shape
    data['width'] = 300
    with pytest.raises(ValueError): lane_wire.decode_request(data)


@pytest.mark.parametrize('updates', [
    {'model_sha256':'other'}, {'width':640}, {'instances':None},
    {'instances':[{'class':'car','points':[[1,1],[2,2],[3,3]]}]},
    {'instances':[{'class':'lane','points':[[1,1],[2,2],[float('nan'),3]]}]},
    {'instances':[{'class':'lane','points':[[1,1],[2,2],[300,3]]}]},
    {'instances':[{'class':'lane','points':[[1,1],[2,2]]}]},
    {'error':'inference failure'},
])
def test_invalid_remote_geometry_rejected(updates):
    with pytest.raises(ValueError): lane_wire.decode_result(reply(**updates),200,100,'weights')


def test_mailbox_rejects_old_duplicate_and_previous_epoch():
    box = lane_wire.PerceptionMailbox(); box.offer('session:1', b'frame')
    assert box.take_request() == b'frame' and box.take_request() is None
    assert not box.submit(json.dumps(reply('wrong:1')))
    assert box.submit(json.dumps(reply()))
    assert not box.submit(json.dumps(reply()))
    assert box.take_reply()['token'] == 'session:1'
    assert not box.submit(json.dumps(reply()))
    box.offer('new:2', b'new'); box.clear()
    assert not box.submit(json.dumps(reply('new:2')))


def test_loopback_http_has_no_motion_endpoint():
    box = lane_wire.PerceptionMailbox()
    server = lane_wire.LoopbackPerceptionServer(box, 0)
    try:
        host, port = server.server.server_address
        assert host == '127.0.0.1'
        http = build_opener(ProxyHandler({})); url = f'http://127.0.0.1:{port}'
        with http.open(url+'/frame', timeout=2) as response: assert response.status == 204
        box.offer('session:1', b'frame')
        with http.open(url+'/frame', timeout=2) as response: assert response.read() == b'frame'
        with http.open(Request(url+'/result',data=json.dumps(reply()).encode()), timeout=2) as response:
            assert response.status == 200
        with pytest.raises(HTTPError) as error: http.open(url+'/enable', timeout=2)
        assert error.value.code == 404
    finally:
        server.close()


def test_fresh_reply_uses_existing_controller_and_disconnect_stops(controller):
    request = prepare(controller)
    controller.safety_clock.seconds += .2
    controller.remote_mailbox.submit(json.dumps(reply(request['token'])))
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.state == DriveState.FOLLOWING
    assert controller.commands[-1].linear.x > 0
    captured = controller.last_result_input_time
    # Repeating the same reply must not extend the original capture TTL.
    assert not controller.remote_mailbox.submit(json.dumps(reply(request['token'])))
    controller.safety_clock.seconds += .61
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.last_result_input_time == captured
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].linear.x == controller.commands[-1].angular.z == 0


def test_expired_reply_never_reaches_control(controller):
    request = prepare(controller)
    controller.remote_mailbox.submit(json.dumps(reply(request['token'])))
    controller.safety_clock.seconds += .81
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.inference_error == 'remote result expired'
    assert controller.commands[-1].linear.x == 0


def test_disable_clears_inflight_frame_and_reply(controller):
    request = prepare(controller)
    controller._enable_callback(SetBool.Request(data=False), SetBool.Response())
    assert controller.remote_pending is None
    assert not controller.remote_mailbox.submit(json.dumps(reply(request['token'])))
    assert controller.commands[-1].linear.x == 0


def test_no_pc_response_stops_and_expires_pending(controller):
    request = prepare(controller)
    controller.safety_clock.seconds += .81
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.remote_pending is None
    assert controller.inference_error == 'remote inference timeout'
    assert not controller.remote_mailbox.submit(json.dumps(reply(request['token'])))
    assert controller.commands[-1].linear.x == controller.commands[-1].angular.z == 0


def test_old_camera_timestamp_cannot_be_renewed_by_new_request(controller):
    prepare(controller)
    controller.remote_pending = None; controller.remote_mailbox.clear()
    controller.last_inference_time = None
    frame = np.zeros((100,200,3),np.uint8)
    header = Header(); header.stamp.sec = 1
    controller._image_callback(controller._bgr_to_message(frame,header))
    controller._start_remote_inference(controller.safety_clock.now())
    assert controller.remote_pending is None
    assert 'timestamp' in controller.inference_error


def test_worker_only_emits_known_segmentation_polygons():
    data = reply(); data.pop('instances')
    frame = np.zeros((100,200,3),np.uint8)
    data = lane_wire.unpack(lane_wire.encode_request(frame,'session:1',Header(),'weights'))
    result = lane_wire.decode_result(reply(),200,100,'weights')
    model = SimpleNamespace(names={0:'crossline',1:'lane'}, predict=lambda *a,**kw:[result])
    response = lane_wire.unpack(predict_reply(model,'weights',data))
    assert len(response['instances']) == 2 and 'cmd_vel' not in response
    response = lane_wire.unpack(predict_reply(model,'wrong',data))
    assert 'error' in response
