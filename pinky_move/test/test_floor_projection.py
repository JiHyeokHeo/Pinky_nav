import json
from pathlib import Path
import pytest
from pinky_move.floor_projection import forward_cm

def calibration():
    return json.loads((Path(__file__).parents[1] / 'config/calibration.json').read_text())

def test_reference_points_forward_fit():
    c = calibration()
    errors = []
    for frame in c['reference_frames']:
        for (u,v), (_,distance) in zip(frame['pixel_points'],frame['ground_points_cm']):
            try: predicted = forward_cm(u,v,c,(640,480))
            except ValueError: continue  # Boundary points may fit outside valid band.
            errors.append((predicted-distance)**2)
    assert len(errors) >= 20
    assert (sum(errors)/len(errors))**.5 < .6

@pytest.mark.parametrize('point,size', [((320,450),(640,480)), ((320,300),(320,240)), ((-1,300),(640,480))])
def test_unsupported_coordinates_rejected(point,size):
    with pytest.raises(ValueError):forward_cm(*point,calibration(),size)
