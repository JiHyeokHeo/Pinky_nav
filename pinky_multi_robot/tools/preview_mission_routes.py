"""Render synthetic UI telemetry offscreen. No ROS initialization or motion."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from geometry_msgs.msg import PoseWithCovarianceStamped
from pinky_multi_robot.nav2_map_ui import MapCanvas
from pinky_multi_robot.mission_view import status_text

app = QApplication([])
root = Path(__file__).resolve().parents[2]
canvas = MapCanvas(root/'pinky_navigation/map/my_pinky_map20.yaml')
left = canvas._origin_x+.25*canvas._pixmap.width()*canvas._resolution
right = canvas._origin_x+.75*canvas._pixmap.width()*canvas._resolution
bottom = canvas._origin_y+.25*canvas._pixmap.height()*canvas._resolution
top = canvas._origin_y+.75*canvas._pixmap.height()*canvas._resolution
view = dict(version=1,generation=1,active=True,state='navigating',priority='pinky1',
    yielding='pinky2',robots={'pinky1':'navigating','pinky2':'yielding'},
    reason='예시: 경로 경합 예상 → pinky2 양보, pinky1 통과 후 재개',
    paths={'pinky1':{'frame_id':'map','points':[[left,bottom],[right,top]]},
           'pinky2':{'frame_id':'map','points':[[left,top],[right,bottom]]}})
canvas.set_mission_view(view)
for robot,xy in [('pinky1',(left,bottom)),('pinky2',(left,top))]:
    pose=PoseWithCovarianceStamped()
    pose.pose.pose.position.x,pose.pose.pose.position.y=xy
    pose.pose.pose.orientation.w=1.
    canvas.set_pose(robot,pose)
window=QWidget()
layout=QVBoxLayout(window)
layout.addWidget(QLabel('Nav2 검사 경로 · 양보 UI 예시 / 실제 로봇 위치·주행 결과 아님'))
row=QHBoxLayout()
row.addWidget(canvas,1)
label=QLabel('빨강=pinky1 / 파랑=pinky2\n노란 테두리=양보 대기\n\n'+status_text(view))
label.setWordWrap(True); label.setMinimumWidth(280); label.setMaximumWidth(320)
row.addWidget(label); layout.addLayout(row)
window.resize(1100,800); window.show(); app.processEvents()
target=Path('/home/tory/Downloads/pinky_yolo_presentation/nav2_route_ui_preview.png')
assert window.grab().save(str(target))
window.close()
print(target)
