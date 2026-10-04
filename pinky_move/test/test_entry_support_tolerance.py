"""10mm tolerance is confined to the final observed-entry endpoint gate."""
import numpy as np
import pytest
import pinky_move.lane_corner as lane
from test_lane_corner import observation


@pytest.mark.parametrize('overrun,accepted',[(.005,True),(.00633,True),(.010,True),(.0101,False),(.011,False)])
def test_start_support_tolerance(monkeypatch,overrun,accepted):
    policy=lane.CornerPolicy(lane.CornerConfig(staged_turn=True,clearance_m=0.))
    obs=dict(observation(),width=.154)
    feature=dict(corner=np.array([.5,-.2]),exit=np.array([.5,.2]),
                 tin=np.array([1.,0.]),tout=np.array([0.,1.]))
    feature=dict(feature,entry=feature['corner']-(.077-overrun)*feature['tin'])
    # No new interior candidate can be certified from endpoint-only coverage;
    # isolate the legacy pivot's final support gate, not candidate rescue.
    monkeypatch.setattr(lane,'nearest_on_chain',lambda query,bound:
        (np.repeat(bound[:1],len(query),axis=0),np.zeros(len(query),dtype=int),np.zeros(len(query))))
    if accepted:
        policy._start_staged(obs,feature,np.zeros(3),1.)
        assert policy.staged is not None
        assert policy.debug['entry_support_overrun_m']==pytest.approx(overrun)
    else:
        with pytest.raises(ValueError,match='outside observed'):
            policy._start_staged(obs,feature,np.zeros(3),1.)
        assert policy.staged is None
