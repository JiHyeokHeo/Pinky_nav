import cv2
import numpy as np
from pinky_move.lane_autonomy import draw_boundary_labels


def test_labels_preserve_assigned_side_and_report_unknown(monkeypatch):
    texts = []
    monkeypatch.setattr(cv2, 'putText', lambda frame,text,*args: texts.append(text))
    polygons = [(1,np.array([[20,200],[40,200],[40,300]])),
                (0,np.array([[100,200],[150,200],[150,300]])),
                (1,np.array([[400,200],[420,200],[420,300]]))]
    draw_boundary_labels(np.zeros((480,640,3),np.uint8),polygons,1,
                         {0:('right','USED')},dict(mode='ENTRY_WAIT',direction='LEFT'))
    assert '#1 RIGHT [USED]' in texts
    assert '#2 UNKNOWN [UNASSIGNED]' in texts
    assert 'STATE: ENTRY_WAIT LEFT' in texts
