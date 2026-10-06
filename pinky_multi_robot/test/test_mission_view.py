"""Read-only telemetry and offscreen Qt route rendering regressions."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from pinky_multi_robot.mission_view import mission_snapshot, parse_view, visible_paths, status_text


def view_server():
    return NS(_task_states={'pinky1': 'parking', 'pinky2': 'parking'},
              _mission_robots={'pinky1', 'pinky2'}, _completed_robots={'pinky1'},
              _safety_fault=None, _route_refresh_pending=False,
              _yielding_robot='pinky2', _reserved_robot=None,
              _active_nav_goals={}, _mission_generation=3, _mission_active=True,
              _diagnostic_state='navigating', _mission_view_reason='pinky2 yields',
              _display_paths={'pinky1': [[0., 0.], [1., 1.]], 'pinky2': [[0., 1.], [1., 0.]]},
              _display_frames={'pinky1': 'map', 'pinky2': 'map'},
              get_parameter=lambda _: NS(value='pinky1'))


def test_snapshot_distinguishes_completed_and_yielding():
    view = mission_snapshot(view_server())
    assert view['robots'] == {'pinky1': 'completed', 'pinky2': 'yielding'}
    assert parse_view(json.dumps(view)) == view
    assert '도착 완료' in status_text(view) and '양보 대기' in status_text(view)


def test_refresh_and_fault_states_override_live_goal():
    s = view_server()
    s._route_refresh_pending = True
    assert mission_snapshot(s)['robots']['pinky2'] == 'replanning'
    s._safety_fault = 'lost pose'
    view = mission_snapshot(s)
    assert view['robots']['pinky2'] == 'stopped'
    assert view['reason'] == 'lost pose'


def test_terminal_failure_keeps_checked_paths_and_reason():
    s = view_server()
    s._mission_active = False
    s._mission_robots = set()
    s._diagnostic_state = 'failed'
    s._mission_view_reason = 'waiting-to-route=0.20m (required 0.35m)'
    view = mission_snapshot(s)
    assert not view['active'] and len(view['paths']) == 2
    assert '0.20m' in status_text(view)


def test_parked_lower_priority_is_shown_as_priority_when_other_yields():
    s = view_server()
    s._yielding_robot = 'pinky1'
    s._completed_robots = {'pinky2'}
    assert mission_snapshot(s)['priority'] == 'pinky2'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), '1', True])
def test_nonfinite_or_wrong_type_coordinates_rejected(value):
    view = mission_snapshot(view_server())
    view['paths']['pinky1']['points'] = [[value, 0.]]
    with pytest.raises(ValueError):
        parse_view(json.dumps(view))


def test_wrong_map_frame_is_not_rendered():
    view = mission_snapshot(view_server())
    view['paths']['pinky2']['frame_id'] = 'other_map'
    assert set(visible_paths(view, 'map')) == {'pinky1'}


@pytest.mark.parametrize('change', [dict(version=2),dict(generation='3'),dict(active=1),
                                   dict(yielding='pinky3'),dict(paths=[]),dict(robots={'bad':'moving'})])
def test_invalid_protocol_rejected(change):
    view = mission_snapshot(view_server())
    view.update(change)
    with pytest.raises(ValueError):
        parse_view(json.dumps(view))


def test_ui_callback_accepts_valid_data_without_motor_command():
    from pinky_multi_robot.nav2_map_ui import MapControlNode
    node = NS(_on_status=Mock(), _on_mission_view=Mock())
    MapControlNode._view_callback(node, NS(data=json.dumps(mission_snapshot(view_server()))))
    node._on_mission_view.assert_called_once()
    node._on_status.assert_not_called()
    MapControlNode._view_callback(node, NS(data='{broken'))
    node._on_status.assert_called_once()


def test_offscreen_canvas_draws_paths_and_yield_ring(tmp_path):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt5.QtWidgets import QApplication
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from pinky_multi_robot.nav2_map_ui import MapCanvas
    app = QApplication.instance() or QApplication([])
    root = Path(__file__).resolve().parents[2]
    canvas = MapCanvas(root/'pinky_navigation/map/my_pinky_map20.yaml')
    canvas.resize(800, 600)
    canvas.set_mission_view(mission_snapshot(view_server()))
    for robot, xy in [('pinky1',(1.,1.)),('pinky2',(0.,1.))]:
        pose = PoseWithCovarianceStamped()
        pose.pose.pose.position.x, pose.pose.pose.position.y = xy
        pose.pose.pose.orientation.w = 1.
        canvas.set_pose(robot, pose)
    canvas.show()
    app.processEvents()
    assert canvas.grab().save(str(tmp_path/'mission_routes.png'))
    canvas.show_routes = False
    canvas.update()
    app.processEvents()
    canvas.close()
