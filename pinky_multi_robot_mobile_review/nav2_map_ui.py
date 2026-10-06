"""PyQt5 map-click controller for the two Pinky Nav2 action proxies."""
import argparse
import math
import sys
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from PyQt5.QtCore import QPointF, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


DEFAULT_MAP_YAML = '/home/tory/ws/pinky_pro/pinky_navigation/map/my_pinky_map10.yaml'
PARKING_GOALS_FILE = Path.home() / '.config' / 'pinky_map_ui' / 'parking_goals.yaml'


class MapCanvas(QWidget):
    """Draw an occupancy map and convert Qt click positions to map coordinates."""

    clicked = pyqtSignal(float, float)

    def __init__(self, map_yaml):
        super().__init__()
        self.setMinimumSize(640, 480)
        self._pixmap = QPixmap()
        self._image = QImage()
        self._resolution = 0.0
        self._origin_x = self._origin_y = 0.0
        self._poses = {}
        self.load_map(map_yaml)

    def load_map(self, map_yaml):
        yaml_path = Path(map_yaml).expanduser().resolve()
        with yaml_path.open(encoding='utf-8') as file:
            config = yaml.safe_load(file)
        image_path = yaml_path.parent / config['image']
        self._resolution = float(config['resolution'])
        self._origin_x, self._origin_y = map(float, config['origin'][:2])
        self._pixmap = QPixmap(str(image_path))
        self._image = QImage(str(image_path))
        if self._pixmap.isNull():
            raise FileNotFoundError(f'Unable to load map image: {image_path}')

    def find_safe_retreat(self, pose):
        """Return the first free point behind pose, or None.

        Unknown/occupied pixels are rejected. This is a static-map filter; the
        robot's Nav2 local costmap remains the final live-obstacle safety check.
        """
        orientation = pose.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))
        point = pose.pose.pose.position
        for distance in (0.20, 0.30, 0.40, 0.50):
            x = point.x - distance * math.cos(yaw)
            y = point.y - distance * math.sin(yaw)
            if self._is_clear(x, y, safety_radius=0.30):
                return x, y, yaw
        return None

    def _is_clear(self, x, y, safety_radius):
        pixel_x = round((x - self._origin_x) / self._resolution)
        pixel_y = round(self._image.height() - 1 - (y - self._origin_y) / self._resolution)
        radius = math.ceil(safety_radius / self._resolution)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                check_x, check_y = pixel_x + dx, pixel_y + dy
                if (check_x < 0 or check_x >= self._image.width() or
                        check_y < 0 or check_y >= self._image.height() or
                        self._image.pixelColor(check_x, check_y).red() < 250):
                    return False
        return True

    def set_pose(self, robot, pose):
        self._poses[robot] = pose
        self.update()

    def _draw_rect(self):
        if self._pixmap.isNull():
            return QRect()
        scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        width = round(self._pixmap.width() * scale)
        height = round(self._pixmap.height() * scale)
        return QRect((self.width() - width) // 2, (self.height() - height) // 2, width, height)

    def _world_to_widget(self, x, y):
        rect = self._draw_rect()
        image_x = (x - self._origin_x) / self._resolution
        image_y = self._pixmap.height() - 1 - (y - self._origin_y) / self._resolution
        return QPointF(
            rect.x() + image_x * rect.width() / self._pixmap.width(),
            rect.y() + image_y * rect.height() / self._pixmap.height())

    def _widget_to_world(self, point):
        rect = self._draw_rect()
        image_x = (point.x() - rect.x()) * self._pixmap.width() / rect.width()
        image_y = (point.y() - rect.y()) * self._pixmap.height() / rect.height()
        return (
            self._origin_x + image_x * self._resolution,
            self._origin_y + (self._pixmap.height() - 1 - image_y) * self._resolution,
        )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._draw_rect().contains(event.pos()):
            self.clicked.emit(*self._widget_to_world(event.pos()))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#202124'))
        rect = self._draw_rect()
        if not rect.isNull():
            painter.drawPixmap(rect, self._pixmap)
        colors = {'pinky1': QColor('#ef4444'), 'pinky2': QColor('#2563eb')}
        for robot, pose in self._poses.items():
            point = self._world_to_widget(pose.pose.pose.position.x, pose.pose.pose.position.y)
            painter.setPen(QPen(colors.get(robot, Qt.white), 3))
            painter.setBrush(colors.get(robot, Qt.white))
            painter.drawEllipse(point, 7, 7)
            painter.drawText(point + QPointF(10, -10), robot)


class MapControlNode(Node):
    def __init__(self, on_status, on_pose):
        super().__init__('pinky_map_ui')
        self._on_status = on_status
        self._on_pose = on_pose
        # Do not call this _clients: Node already uses that private name for
        # its ROS service clients, which the executor iterates over.
        self._nav_clients = {
            robot: ActionClient(self, NavigateToPose, f'/{robot}/navigate_to_pose')
            for robot in ('pinky1', 'pinky2')
        }
        self._goal_handles = {}
        self._goal_sequences = {robot: 0 for robot in self._nav_clients}
        self._poses = {}
        self._shared_goals = {}
        self._yielding_robot = None
        self._retreat_completed = False
        self._yield_blocked = False
        for robot in self._nav_clients:
            self.create_subscription(
                PoseWithCovarianceStamped, f'/{robot}/amcl_pose',
                lambda message, name=robot: self._pose_callback(name, message), 10)

    def _pose_callback(self, robot, message):
        self._poses[robot] = message
        self._on_pose(robot, message)

    def send_goal(self, robot, x, y, yaw, frame_id, purpose='normal'):
        client = self._nav_clients[robot]
        if not client.server_is_ready():
            self._on_status(f'{robot}: action proxy is not available on Domain 52.')
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = frame_id
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_sequences[robot] += 1
        sequence = self._goal_sequences[robot]
        future = client.send_goal_async(goal, feedback_callback=lambda _msg: None)
        future.add_done_callback(
            lambda done, name=robot, seq=sequence, kind=purpose:
            self._goal_response(name, seq, kind, done))
        self._on_status(f'{robot}: {purpose} goal sent x={x:.2f}, y={y:.2f}.')

    def send_goals(self, robots, x, y, yaw, frame_id):
        """Dispatch independently so two robots begin navigation together."""
        for robot in robots:
            self.send_goal(robot, x, y, yaw, frame_id)
        if set(robots) == {'pinky1', 'pinky2'}:
            self._shared_goals = {
                robot: (x, y, yaw, frame_id) for robot in robots}
            self._yielding_robot = None
            self._retreat_completed = False
            self._yield_blocked = False

    def _goal_response(self, robot, sequence, purpose, future):
        try:
            handle = future.result()
        except Exception as error:
            self._on_status(f'{robot}: goal transport error: {error}')
            return
        if not handle.accepted:
            self._on_status(f'{robot}: goal rejected.')
            return
        self._goal_handles[robot] = (sequence, handle)
        self._on_status(f'{robot}: {purpose} goal accepted.')
        handle.get_result_async().add_done_callback(
            lambda done, name=robot, seq=sequence, kind=purpose:
            self._goal_result(name, seq, kind, done))

    def _goal_result(self, robot, sequence, purpose, future):
        try:
            response = future.result()
            self._on_status(f'{robot}: finished (status {response.status}, '
                            f'error {response.result.error_code}: {response.result.error_msg})')
            if robot == self._yielding_robot and purpose == 'retreat':
                self._retreat_completed = response.status == 4
                if not self._retreat_completed:
                    self._on_status(f'{robot}: retreat did not complete; keeping both goals paused.')
        except Exception as error:
            self._on_status(f'{robot}: result error: {error}')
        current = self._goal_handles.get(robot)
        if current is not None and current[0] == sequence:
            self._goal_handles.pop(robot, None)

    def cancel_goal(self, robot):
        entry = self._goal_handles.get(robot)
        if entry is None:
            self._on_status(f'{robot}: no active goal to cancel.')
            return
        entry[1].cancel_goal_async()
        self._on_status(f'{robot}: cancellation requested.')

    @staticmethod
    def _heading(pose):
        q = pose.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

    def monitor_yield(self, retreat_finder):
        """Yield pinky2 only for a close, head-on shared-goal encounter."""
        if set(self._shared_goals) != {'pinky1', 'pinky2'}:
            return
        first, second = self._poses.get('pinky1'), self._poses.get('pinky2')
        if first is None or second is None:
            return
        a, b = first.pose.pose.position, second.pose.pose.position
        dx, dy = b.x - a.x, b.y - a.y
        distance = math.hypot(dx, dy)
        if distance == 0.0:
            return
        direction = math.atan2(dy, dx)
        head_on = (math.cos(self._heading(first) - direction) > 0.5 and
                   math.cos(self._heading(second) - (direction + math.pi)) > 0.5)
        if self._yielding_robot is None and not self._yield_blocked and distance < 0.60 and head_on:
            retreat = retreat_finder(second)
            if retreat is None:
                self._yield_blocked = True
                self._yielding_robot = 'pinky2'
                self._on_status('No safe retreat for pinky2; cancelling its goal and holding position.')
                self.cancel_goal('pinky2')
                return
            self._yielding_robot = 'pinky2'
            self._retreat_completed = False
            entry = self._goal_handles.get('pinky2')
            self._on_status('Head-on encounter: pinky2 yields to pinky1.')
            if entry is None:
                self.send_goal('pinky2', *retreat, self._shared_goals['pinky2'][3], purpose='retreat')
                return
            cancel_future = entry[1].cancel_goal_async()
            cancel_future.add_done_callback(
                lambda _done, goal=retreat, frame=self._shared_goals['pinky2'][3]:
                self.send_goal('pinky2', *goal, frame, purpose='retreat'))
            return
        if self._yielding_robot == 'pinky2' and self._retreat_completed and distance >= 0.85:
            goal = self._shared_goals['pinky2']
            self._yielding_robot = None
            self._retreat_completed = False
            self._on_status('Clearance restored: resuming pinky2 original goal.')
            self.send_goal('pinky2', *goal, purpose='resume')


class MainWindow(QMainWindow):
    status_changed = pyqtSignal(str)
    pose_changed = pyqtSignal(str, object)

    def __init__(self, map_yaml, frame_id):
        super().__init__()
        self.setWindowTitle('Pinky Nav2 Map Control — Domain 52')
        self._frame_id = frame_id
        self.canvas = MapCanvas(map_yaml)
        self.canvas.clicked.connect(self._map_clicked)
        self.robot = QComboBox()
        self.robot.addItems(['pinky1', 'pinky2', 'Both (simultaneous)'])
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-180.0, 180.0)
        self.yaw.setSuffix('°')
        send_label = QLabel(
            'Select a robot (or both), then click a free point to navigate.')
        cancel = QPushButton('Cancel selected robot goal')
        cancel.clicked.connect(self._cancel_selected)
        self.set_parking = QPushButton('Set parking: next click')
        self.set_parking.setCheckable(True)
        self.park = QPushButton('Go to saved parking')
        self.park.clicked.connect(self._go_to_parking)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Robot:'))
        controls.addWidget(self.robot)
        controls.addWidget(QLabel('Heading:'))
        controls.addWidget(self.yaw)
        controls.addWidget(self.set_parking)
        controls.addWidget(self.park)
        controls.addWidget(cancel)
        controls.addStretch()
        self.status = QLabel('Starting ROS node…')
        self.status.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(send_label)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.node = MapControlNode(self.status_changed.emit, self.pose_changed.emit)
        self._parking_goals = self._load_parking_goals()
        self.status_changed.connect(self.status.setText)
        self.pose_changed.connect(self.canvas.set_pose)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.executor.spin_once(timeout_sec=0.0))
        self.timer.start(20)
        self.yield_timer = QTimer(self)
        self.yield_timer.timeout.connect(
            lambda: self.node.monitor_yield(self.canvas.find_safe_retreat))
        self.yield_timer.start(200)

    def _selected_robots(self):
        return ('pinky1', 'pinky2') if self.robot.currentIndex() == 2 else (self.robot.currentText(),)

    @staticmethod
    def _load_parking_goals():
        if not PARKING_GOALS_FILE.exists():
            return {}
        try:
            with PARKING_GOALS_FILE.open(encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError:
            return {}

    def _save_parking_goals(self):
        PARKING_GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with PARKING_GOALS_FILE.open('w', encoding='utf-8') as file:
            yaml.safe_dump(self._parking_goals, file, sort_keys=True)

    def _map_clicked(self, x, y):
        robots = self._selected_robots()
        yaw = math.radians(self.yaw.value())
        if self.set_parking.isChecked():
            for robot in robots:
                self._parking_goals[robot] = {'x': x, 'y': y, 'yaw': yaw}
            self._save_parking_goals()
            self.set_parking.setChecked(False)
            self.status.setText(
                f'Saved parking for {", ".join(robots)}: x={x:.2f}, y={y:.2f}.')
            return
        self.node.send_goals(robots, x, y, yaw, self._frame_id)

    def _go_to_parking(self):
        robots = self._selected_robots()
        missing = [robot for robot in robots if robot not in self._parking_goals]
        if missing:
            QMessageBox.warning(
                self, 'Parking point missing',
                f'Use “Set parking: next click” for {", ".join(missing)} first.')
            return
        for robot in robots:
            goal = self._parking_goals[robot]
            self.node.send_goal(robot, goal['x'], goal['y'], goal.get('yaw', 0.0), self._frame_id)

    def _cancel_selected(self):
        for robot in self._selected_robots():
            self.node.cancel_goal(robot)

    def closeEvent(self, event):
        self.timer.stop()
        self.yield_timer.stop()
        self.executor.shutdown()
        self.node.destroy_node()
        event.accept()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-yaml', default=DEFAULT_MAP_YAML)
    parser.add_argument('--frame-id', default='map')
    parser.add_argument('--domain-id', type=int, default=52)
    options, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args, domain_id=options.domain_id)
    app = QApplication(sys.argv)
    try:
        window = MainWindow(options.map_yaml, options.frame_id)
    except (FileNotFoundError, KeyError, OSError, yaml.YAMLError) as error:
        QMessageBox.critical(None, 'Map load failed', str(error))
        rclpy.shutdown()
        return 1
    window.resize(1000, 800)
    window.show()
    result = app.exec_()
    if rclpy.ok():
        rclpy.shutdown()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
