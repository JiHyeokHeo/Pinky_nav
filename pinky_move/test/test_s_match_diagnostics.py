"""Expose failed correspondence checks without changing control decisions."""
import numpy as np
import pytest
from test_s_route_progress import seeded, path, target, observation


def test_rejected_distance_is_logged_before_exception():
    p,_=seeded()
    old=p.s_route['path'].copy()
    with pytest.raises(ValueError,match='skips or contradicts'):
        p._track_s_route(target(path()+[0.,.03],np.zeros(3)),observation(),np.zeros(3),1.2,.22)
    assert 'distance_gt_2cm' in p.debug['s_match_failures']
    assert p.debug['s_match_error_m']>.02
    assert p.debug['s_match_required_span_m']>0
    np.testing.assert_array_equal(p.s_route['path'],old)


def test_success_reports_no_failed_checks():
    p,_=seeded()
    result=p._track_s_route(target(path(),np.zeros(3)),observation(),np.zeros(3),1.2,.22)
    assert result['s_route_tracking']
    assert p.debug['s_match_failures']==[]
    assert p.debug['s_match_error_m']<.001
