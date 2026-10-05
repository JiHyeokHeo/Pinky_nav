"""실차 시험 프로파일은 수동 활성화 전 모터를 시작하지 않는다."""
from pathlib import Path
import yaml


def test_robot_profile_is_opt_in_disabled_and_not_white_mode():
    p = Path(__file__).parents[1]/'config/lane_connected_robot_test.yaml'
    values = yaml.safe_load(p.read_text())['lane_autonomy']['ros__parameters']
    assert values['enabled'] is False
    assert values['connected_geometry'] is True
    assert values['simulation_white_lane'] is False
    assert values['remote_inference'] and values['remote_geometry']
    assert values['linear_speed'] <= .03
    assert 'calibration_path' not in values
    assert 'corner_odom_timeout' not in values
