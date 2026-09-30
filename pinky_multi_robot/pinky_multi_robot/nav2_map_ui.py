"""PyQt5 map-click controller for the two Pinky Nav2 action proxies."""
import argparse
import math
import sys
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from PyQt5.QtCore import QPointF, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QGroupBox, QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from pinky_multi_robot.localization_command_bus import LocalizationCommandBus


DEFAULT_MAP_YAML = '/home/tory/ws/pinky_pro/pinky_navigation/map/my_pinky_map10.yaml'
PARKING_GOALS_FILE = Path.home() / '.config' / 'pinky_map_ui' / 'parking_goals.yaml'
PATROL_POINTS_FILE = Path.home() / '.config' / 'pinky_map_ui' / 'patrol_points.yaml'


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
            q = pose.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            tip = self._world_to_widget(pose.pose.pose.position.x + 0.16 * math.cos(yaw),
                                        pose.pose.pose.position.y + 0.16 * math.sin(yaw))
            painter.drawLine(point, tip)
            painter.drawText(point + QPointF(10, -10), robot)


class MapControlNode(Node):
    def __init__(self, on_status, on_pose, on_localization=None):
        super().__init__('pinky_map_ui')
        self._on_status = on_status
        self._on_pose = on_pose
        self._on_localization = on_localization
        self._localization_states = {}
        # Do not call this _clients: Node already uses that private name for
        # its ROS service clients, which the executor iterates over.
        self._nav_clients = {
            robot: ActionClient(self, NavigateToPose, f'/{robot}/navigate_to_pose')
            for robot in ('pinky1', 'pinky2')
        }
        self._mission_client = ActionClient(self, ExecuteMultiRobotMission,
                                            '/execute_multi_robot_mission')
        self._mission_handle = None
        self._mission_pending = False
        self._mission_robots = set()
        self._cancel_pending_mission = False
        self._goal_pending = set()
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
            self.create_subscription(
                String, f'/{robot}/localization_status',
                lambda message, name=robot: self._localization_callback(name, message), 10)

    def _localization_callback(self, robot, message):
        if self._localization_states.get(robot) != message.data:
            self._localization_states[robot] = message.data
            if self._on_localization is not None:
                self._on_localization(robot, message.data)

    def _pose_callback(self, robot, message):
        self._poses[robot] = message
        self._on_pose(robot, message)

    def send_goal(self, robot, x, y, yaw, frame_id, purpose='normal'):
        if robot in self._mission_robots and (
                self._mission_pending or self._mission_handle is not None):
            self._on_status(f'{robot}: already controlled by the active mission.')
            return
        if robot in self._goal_pending or robot in self._goal_handles:
            self._on_status(f'{robot}: a direct goal is already active.')
            return

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
        self._goal_pending.add(robot)
        future = client.send_goal_async(goal, feedback_callback=lambda _msg: None)
        future.add_done_callback(
            lambda done, name=robot, seq=sequence, kind=purpose:
            self._goal_response(name, seq, kind, done))
        self._on_status(f'{robot}: {purpose} goal sent x={x:.2f}, y={y:.2f}.')

    def send_goals(self, robots, x, y, yaw, frame_id):
        self.send_park_mission(robots, x, y, yaw, frame_id)

    @staticmethod
    def _pose(x, y, yaw, frame_id):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        return pose

    def _send_mission(self, tasks, label):
        if self._mission_pending or self._mission_handle is not None:
            self._on_status('A mission is already pending or active; cancel it first.')
            return False
        robots = {task.robot_id.strip('/') for task in tasks}
        if any(robot in self._goal_handles or robot in self._goal_pending for robot in robots):
            self._on_status('A selected robot has an active direct goal; cancel it first.')
            return False
        if not self._mission_client.server_is_ready():
            self._on_status('Mission server is unavailable on Domain 52.')
            return False
        goal = ExecuteMultiRobotMission.Goal()
        goal.tasks = tasks
        self._mission_pending = True
        self._mission_robots = robots
        self._cancel_pending_mission = False
        future = self._mission_client.send_goal_async(goal, feedback_callback=self._mission_feedback)
        future.add_done_callback(lambda done: self._mission_response(label, done))
        self._on_status(f'{label}: mission sent.')
        return True

    def send_park_mission(self, robots, x, y, yaw, frame_id):
        tasks = []
        for robot in robots:
            task = RobotTask(); task.robot_id = robot; task.task_type = RobotTask.PARK
            task.parking_goal = self._pose(x, y, yaw, frame_id(robot) if callable(frame_id) else frame_id); tasks.append(task)
        self._send_mission(tasks, 'Parking/move')

    def send_patrol_mission(self, robots, points, laps, frame_id):
        tasks = []
        for robot in robots:
            task = RobotTask(); task.robot_id = robot; task.task_type = RobotTask.PATROL
            task.patrol_waypoints = [self._pose(p['x'], p['y'], p.get('yaw', 0.0), frame_id(robot) if callable(frame_id) else frame_id) for p in points[robot]]
            task.patrol_laps = laps; tasks.append(task)
        self._send_mission(tasks, 'Patrol')

    def _mission_feedback(self, message):
        feedback = message.feedback
        self._on_status(f'{feedback.robot_id}: {feedback.state} ({feedback.completed_waypoints}/{feedback.total_waypoints})')

    def _mission_response(self, label, future):
        self._mission_pending = False
        try:
            handle = future.result()
        except Exception as error:
            self._mission_robots.clear()
            self._on_status(f'{label}: transport error: {error}')
            return
        if not handle.accepted:
            self._mission_robots.clear()
            self._on_status(f'{label}: rejected; another mission may be running.')
            return
        self._mission_handle = handle
        self._on_status(f'{label}: accepted.')
        handle.get_result_async().add_done_callback(
            lambda done, mission=handle: self._mission_result(mission, done))
        if self._cancel_pending_mission:
            handle.cancel_goal_async()
            self._cancel_pending_mission = False

    def _mission_result(self, handle, future):
        try:
            response = future.result()
            self._on_status(f'Mission finished: status {response.status}; {response.result.message}')
        except Exception as error:
            self._on_status(f'Mission result error: {error}')
        if self._mission_handle is handle:
            self._mission_handle = None
            self._mission_robots.clear()

    def cancel_mission(self):
        if self._mission_handle is None:
            if self._mission_pending:
                self._cancel_pending_mission = True
                self._on_status('Mission will be cancelled when accepted.')
            else:
                self._on_status('No active mission to cancel.')
            return
        self._mission_handle.cancel_goal_async()
        self._on_status('Mission cancellation requested.')

    def _goal_response(self, robot, sequence, purpose, future):
        self._goal_pending.discard(robot)
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
    localization_changed = pyqtSignal(str, str)

    def __init__(self, map_yaml, frame_id, simulation=False, simulation_domain=152):
        super().__init__()
        self.setWindowTitle(f'Pinky Nav2 Map Control — Domain {simulation_domain if simulation else 52}')
        self._frame_id = frame_id
        self._simulation = simulation
        config_dir = Path.home() / '.config' / 'pinky_map_ui'
        self._parking_goals_file = config_dir / ('parking_goals_sim.yaml' if simulation else 'parking_goals.yaml')
        self._patrol_points_file = config_dir / ('patrol_points_sim.yaml' if simulation else 'patrol_points.yaml')
        self.canvas = MapCanvas(map_yaml)
        self.canvas.clicked.connect(self._map_clicked)
        self.robot = QComboBox()
        self.robot.addItems(['pinky1', 'pinky2', 'Both (simultaneous)'])
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-180.0, 180.0)
        self.yaw.setSuffix('°')
        send_label = QLabel(
            'Select a robot (or both), then click a free point to navigate.')
        cancel = QPushButton('Cancel active mission / selected goal')
        cancel.clicked.connect(self._cancel_selected)
        self.set_initial_pose_button = QPushButton('Set current pose: next click')
        self.set_initial_pose_button.setCheckable(True)
        self.stage_goal = QPushButton('Set goal (stage): next click')
        self.stage_goal.setCheckable(True)
        self._staged_goals = {}
        self.set_parking = QPushButton('Set parking: next click')
        self.set_parking.setCheckable(True)
        self.add_patrol = QPushButton('Add patrol point: next click')
        self.add_patrol.setCheckable(True)
        self._click_modes = QButtonGroup(self)
        self._click_modes.setExclusive(False)
        for button in (self.set_initial_pose_button, self.stage_goal,
                       self.set_parking, self.add_patrol):
            self._click_modes.addButton(button)
            button.toggled.connect(
                lambda checked, selected=button: [other.setChecked(False)
                                                  for other in self._click_modes.buttons()
                                                  if checked and other is not selected])
        self.park = QPushButton('Go to saved parking')
        self.park.clicked.connect(self._go_to_parking)
        self.start_patrol = QPushButton('Start patrol')
        self.start_patrol.clicked.connect(self._start_patrol)
        self.clear_patrol = QPushButton('Clear selected patrol')
        self.clear_patrol.clicked.connect(self._clear_patrol)
        self.laps = QSpinBox(); self.laps.setRange(1, 100); self.laps.setValue(1)
        self.patrol_list = QListWidget()
        selection_group = QGroupBox('Robot and heading')
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.addWidget(self.robot)
        heading_row = QHBoxLayout()
        heading_row.addWidget(QLabel('Heading:'))
        heading_row.addWidget(self.yaw)
        selection_layout.addLayout(heading_row)

        localization_group = QGroupBox('Localization')
        localization_layout = QVBoxLayout(localization_group)
        localization_layout.addWidget(self.set_initial_pose_button)

        goal_group = QGroupBox('Goals and parking')
        goal_layout = QVBoxLayout(goal_group)
        goal_layout.addWidget(self.stage_goal)
        goal_layout.addWidget(self.set_parking)
        goal_layout.addWidget(self.park)

        patrol_group = QGroupBox('Patrol')
        patrol_layout = QVBoxLayout(patrol_group)
        patrol_layout.addWidget(self.add_patrol)
        laps_row = QHBoxLayout()
        laps_row.addWidget(QLabel('Laps:'))
        laps_row.addWidget(self.laps)
        patrol_layout.addLayout(laps_row)
        patrol_layout.addWidget(self.start_patrol)
        patrol_layout.addWidget(self.clear_patrol)
        patrol_layout.addWidget(QLabel('Saved points (robot | x | y | heading):'))
        self.patrol_list.setMinimumHeight(100)
        patrol_layout.addWidget(self.patrol_list)

        combined_group = QGroupBox('Combined mission')
        combined_layout = QVBoxLayout(combined_group)
        self.combined_modes = {}
        for robot_name in ('pinky1', 'pinky2'):
            mode = QComboBox()
            mode.addItems(['Idle', 'Staged goal', 'Saved parking', 'Patrol'])
            self.combined_modes[robot_name] = mode
            combined_layout.addWidget(QLabel(f'{robot_name} task:'))
            combined_layout.addWidget(mode)
        send_combined = QPushButton('Send combined mission')
        send_combined.clicked.connect(self._start_combined_mission)
        combined_layout.addWidget(send_combined)
        self.status = QLabel('Starting ROS node…')
        self.status.setWordWrap(True)
        self.staged_summary = QLabel('Staged goals: none')
        self.staged_summary.setWordWrap(True)
        self.health_labels = {robot: QLabel(f'{robot}: localization unknown')
                              for robot in ('pinky1', 'pinky2')}

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        for group in (selection_group, localization_group, goal_group,
                      patrol_group, combined_group):
            side_layout.addWidget(group)
        side_layout.addWidget(cancel)
        side_layout.addStretch()
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side_panel)
        side_scroll.setMinimumWidth(290)
        side_scroll.setMaximumWidth(350)

        layout = QVBoxLayout()
        layout.addWidget(send_label)
        main_row = QHBoxLayout()
        main_row.addWidget(self.canvas, 1)
        main_row.addWidget(side_scroll)
        layout.addLayout(main_row, 1)
        layout.addWidget(self.staged_summary)
        for health_label in self.health_labels.values():
            layout.addWidget(health_label)
        layout.addWidget(self.status)
        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.node = MapControlNode(self.status_changed.emit, self.pose_changed.emit,
                                   self.localization_changed.emit)
        if simulation:
            self._localization_bus = LocalizationCommandBus(
                domains={'pinky1': simulation_domain, 'pinky2': simulation_domain},
                namespaces={'pinky1': 'pinky1', 'pinky2': 'pinky2'})
        else:
            self._localization_bus = LocalizationCommandBus()
        self._localization_states = {}
        self._parking_goals = self._load_parking_goals()
        self._patrol_points = self._load_patrol_points()
        self._refresh_patrol_list()
        self.status_changed.connect(self.status.setText)
        self.pose_changed.connect(self.canvas.set_pose)
        self.localization_changed.connect(self._localization_updated)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.executor.spin_once(timeout_sec=0.0))
        self.timer.start(20)

    def _selected_robots(self):
        return ('pinky1', 'pinky2') if self.robot.currentIndex() == 2 else (self.robot.currentText(),)

    def _frame_for(self, robot):
        return f'{robot}/{self._frame_id}' if self._simulation else self._frame_id

    def _selected_single_robot(self):
        if self.robot.currentIndex() == 2:
            QMessageBox.warning(self, 'Select one robot',
                                'Select pinky1 or pinky2 for this map operation.')
            return None
        return self.robot.currentText()

    def _localization_updated(self, robot, state):
        self._localization_states[robot] = state
        self._refresh_health(robot)

    def _refresh_health(self, robot):
        pose = self._localization_states.get(robot, 'unknown')
        self.health_labels[robot].setText(f'{robot}: localization {pose}')

    def _load_parking_goals(self):
        if not self._parking_goals_file.exists():
            return {}
        try:
            with self._parking_goals_file.open(encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError:
            return {}

    def _save_parking_goals(self):
        self._parking_goals_file.parent.mkdir(parents=True, exist_ok=True)
        with self._parking_goals_file.open('w', encoding='utf-8') as file:
            yaml.safe_dump(self._parking_goals, file, sort_keys=True)

    def _load_patrol_points(self):
        if not self._patrol_points_file.exists(): return {'pinky1': [], 'pinky2': []}
        try:
            with self._patrol_points_file.open(encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
                return {robot: data.get(robot, []) for robot in ('pinky1', 'pinky2')}
        except yaml.YAMLError:
            return {'pinky1': [], 'pinky2': []}

    def _save_patrol_points(self):
        self._patrol_points_file.parent.mkdir(parents=True, exist_ok=True)
        with self._patrol_points_file.open('w', encoding='utf-8') as file:
            yaml.safe_dump(self._patrol_points, file, sort_keys=True)

    def _refresh_patrol_list(self):
        self.patrol_list.clear()
        for robot in ('pinky1', 'pinky2'):
            for index, point in enumerate(self._patrol_points[robot], start=1):
                self.patrol_list.addItem(f'{robot} #{index} | x={point["x"]:.3f} | y={point["y"]:.3f} | heading={math.degrees(point.get("yaw", 0.0)):.1f}°')

    def _map_clicked(self, x, y):
        robots = self._selected_robots()
        yaw = math.radians(self.yaw.value())
        if self.set_initial_pose_button.isChecked():
            robot = self._selected_single_robot()
            if robot is None:
                return
            if not self.canvas._is_clear(x, y, safety_radius=0.12):
                QMessageBox.warning(self, 'Pose outside free map area',
                                    'Choose a free area at least 12 cm from mapped obstacles.')
                return
            if ((robot in self.node._mission_robots and
                 (self.node._mission_pending or self.node._mission_handle is not None)) or
                    self.node._goal_handles.get(robot) or robot in self.node._goal_pending):
                QMessageBox.warning(self, 'Navigation active',
                                    'Cancel active navigation before resetting the pose.')
                return
            if not self._localization_bus.set_initial_pose(robot, x, y, yaw, self._frame_for(robot)):
                QMessageBox.warning(self, 'AMCL unavailable',
                                    f'{robot} is not subscribing to /initialpose.')
                return
            self.set_initial_pose_button.setChecked(False)
            self.status.setText(f'{robot}: initial pose sent x={x:.2f}, y={y:.2f}, '
                                f'heading={self.yaw.value():.0f}°. Check map alignment before driving.')
            return
        if self.stage_goal.isChecked():
            robot = self._selected_single_robot()
            if robot is None:
                return
            if not self.canvas._is_clear(x, y, safety_radius=0.12):
                QMessageBox.warning(self, 'Goal outside free map area',
                                    'Choose a free map area for the staged goal.')
                return
            self._staged_goals[robot] = {'x': x, 'y': y, 'yaw': yaw}
            self.stage_goal.setChecked(False)
            self.combined_modes[robot].setCurrentText('Staged goal')
            summary = ', '.join(f'{name}: ({point["x"]:.2f}, {point["y"]:.2f})'
                                for name, point in self._staged_goals.items())
            self.staged_summary.setText(f'Staged goals: {summary}')
            self.status.setText(f'{robot}: goal staged, not sent. Use Send combined mission.')
            return
        if self.set_parking.isChecked():
            for robot in robots:
                self._parking_goals[robot] = {'x': x, 'y': y, 'yaw': yaw}
            self._save_parking_goals()
            self.set_parking.setChecked(False)
            self.status.setText(
                f'Saved parking for {", ".join(robots)}: x={x:.2f}, y={y:.2f}.')
            return
        if self.add_patrol.isChecked():
            for robot in robots:
                self._patrol_points[robot].append({'x': x, 'y': y, 'yaw': yaw})
            self._save_patrol_points(); self._refresh_patrol_list(); self.add_patrol.setChecked(False)
            self.status.setText(f'Added patrol point for {", ".join(robots)}: x={x:.2f}, y={y:.2f}.')
            return
        self.node.send_goals(robots, x, y, yaw, self._frame_for)

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
        goal = self._parking_goals[robots[0]]
        if len(robots) == 2 and self._parking_goals[robots[0]] != self._parking_goals[robots[1]]:
            tasks = []
            for robot in robots:
                point = self._parking_goals[robot]; task = RobotTask(); task.robot_id = robot; task.task_type = RobotTask.PARK
                task.parking_goal = self.node._pose(point['x'], point['y'], point.get('yaw', 0.0), self._frame_for(robot)); tasks.append(task)
            self.node._send_mission(tasks, 'Parking')
        else:
            self.node.send_park_mission(robots, goal['x'], goal['y'], goal.get('yaw', 0.0), self._frame_for)

    def _start_patrol(self):
        robots = self._selected_robots(); missing = [r for r in robots if not self._patrol_points[r]]
        if missing:
            QMessageBox.warning(self, 'Patrol points missing', f'Add patrol points for {", ".join(missing)} first.'); return
        self.node.send_patrol_mission(robots, self._patrol_points, self.laps.value(), self._frame_for)
    def _start_combined_mission(self):
        selections = {robot: combo.currentText()
                      for robot, combo in self.combined_modes.items()
                      if combo.currentText() != 'Idle'}
        if not selections:
            QMessageBox.warning(self, 'No tasks selected',
                                'Choose at least one robot task before sending.')
            return
        not_ready = [robot for robot in selections
                     if self._localization_states.get(robot) != 'READY']
        if not_ready:
            QMessageBox.warning(self, 'Localization not ready',
                                f'{", ".join(not_ready)} is not READY. Set its current pose '
                                'and check map alignment before driving.')
            return
        tasks = []
        for robot, mode in selections.items():
            task = RobotTask()
            task.robot_id = robot
            if mode == 'Patrol':
                points = self._patrol_points.get(robot, [])
                if not points:
                    QMessageBox.warning(self, 'Patrol points missing',
                                        f'Add patrol points for {robot} first.')
                    return
                task.task_type = RobotTask.PATROL
                task.patrol_waypoints = [self.node._pose(
                    point['x'], point['y'], point.get('yaw', 0.0), self._frame_for(robot))
                    for point in points]
                task.patrol_laps = self.laps.value()
            else:
                point = (self._staged_goals if mode == 'Staged goal'
                         else self._parking_goals).get(robot)
                if point is None:
                    QMessageBox.warning(self, 'Goal missing',
                                        f'Set a {mode.lower()} for {robot} first.')
                    return
                task.task_type = RobotTask.PARK
                task.parking_goal = self.node._pose(
                    point['x'], point['y'], point.get('yaw', 0.0), self._frame_for(robot))
            tasks.append(task)
        self.node._send_mission(tasks, 'Combined ' + ', '.join(
            f'{robot}:{mode}' for robot, mode in selections.items()))


    def _clear_patrol(self):
        for robot in self._selected_robots(): self._patrol_points[robot] = []
        self._save_patrol_points(); self._refresh_patrol_list()

    def _cancel_selected(self):
        if self.node._mission_pending or self.node._mission_handle is not None:
            self.node.cancel_mission()
        else:
            for robot in self._selected_robots():
                self.node.cancel_goal(robot)

    def closeEvent(self, event):
        self.timer.stop()
        self.executor.shutdown()
        self.node.destroy_node()
        self._localization_bus.close()
        event.accept()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-yaml', default=DEFAULT_MAP_YAML)
    parser.add_argument('--frame-id', default='map')
    parser.add_argument('--domain-id', type=int, default=52)
    parser.add_argument('--simulation', action='store_true')
    parser.add_argument('--simulation-domain', type=int, default=152)
    options, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args, domain_id=options.domain_id)
    app = QApplication(sys.argv)
    try:
        window = MainWindow(options.map_yaml, options.frame_id,
                            options.simulation, options.simulation_domain)
    except (FileNotFoundError, KeyError, OSError, yaml.YAMLError) as error:
        QMessageBox.critical(None, 'Map load failed', str(error))
        rclpy.shutdown()
        return 1
    window.resize(1100, 800)
    window.show()
    result = app.exec_()
    if rclpy.ok():
        rclpy.shutdown()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
