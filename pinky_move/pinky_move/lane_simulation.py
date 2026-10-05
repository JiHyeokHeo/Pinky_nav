"""Gazebo lane-course construction, separate from production robot settings.

Reuse Pinky's URDF mass/collision/wheel model. Only sensors, topic names and
camera intrinsics change in the generated simulation copy. No hardware launch.
"""
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from .robot_projection import fixed_transform, validate_calibration

COURSES = ('two_lines', 'left90', 'right90', 's_bend', 'single_gap')


def course_points(course):
    """Centreline in world metres; two real painted boundaries surround it."""
    if course not in COURSES:
        raise ValueError('unknown lane course')
    if course in ('two_lines', 'single_gap'):
        return np.array([[-.35, 0.], [2., 0.]])
    if course in ('left90', 'right90'):
        sign = 1. if course == 'left90' else -1.
        return np.array([[-.35, 0.], [.75, 0.], [.75, sign*1.25]])
    x = np.linspace(-.35, 2.1, 125)
    # Smooth zero-slope entry/exit and alternating curvature inside the bend.
    u = np.clip((x-.45)/1.3, 0., 1.)
    y = .24*np.sin(2*np.pi*u)*np.sin(np.pi*u)**2
    return np.column_stack([x, y])


def offset_polyline(points, distance):
    """Miter joins keep the painted pair continuous around right angles."""
    p = np.asarray(points, float)
    d = np.diff(p, axis=0)
    t = d/np.linalg.norm(d, axis=1)[:, None]
    normals = np.column_stack([-t[:, 1], t[:, 0]])
    offsets = [normals[0]*distance]
    for a, b in zip(normals, normals[1:]):
        offsets.append((a+b)*distance/max(1e-6, 1.+float(a@b)))
    offsets.append(normals[-1]*distance)
    return p+np.asarray(offsets)


def element(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def set_text(parent, tag, value):
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = str(value)


def visual_box(parent, name, pose, size, color, collision=False):
    model = element(parent, 'model', name=name)
    element(model, 'static', 'true')
    element(model, 'pose', ' '.join(map(str, pose)))
    link = element(model, 'link', name='body')
    for kind in ('visual', 'collision') if collision else ('visual',):
        shape = element(link, kind, name=kind)
        box = element(element(shape, 'geometry'), 'box')
        element(box, 'size', ' '.join(map(str, size)))
        if kind == 'visual':
            material = element(shape, 'material')
            element(material, 'ambient', color)
            element(material, 'diffuse', color)
    return model


def prepare_simulation(description, output, course='two_lines', lane_width=.20):
    """Generate SDF + matched ideal-camera calibration + course metadata.

    Ideal zero distortion is deliberate: the real lens distortion coefficients
    must not be applied to a Gazebo pinhole camera. Extrinsics come from the
    exact expanded URDF used to spawn the physical robot.
    """
    if course not in COURSES or not .16 <= lane_width <= .60:
        raise ValueError('invalid course/lane width')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import xacro
    document = xacro.process_file(str(Path(description)/'urdf/robot.urdf.xacro'),
        mappings={'is_sim': 'true', 'cam_tilt_deg': '8', 'namespace': ''})
    urdf = ET.fromstring(document.toxml())
    optical = np.eye(4)
    optical[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    transform = fixed_transform(ET.tostring(urdf, encoding='unicode'), 'base_link', 'front_camera_link')@optical
    calibration = dict(method='intrinsics_urdf_floor', frame_id='base_link',
        ground_plane_z_m=-.028, image_size=[640, 480], rotation_degrees=0,
        camera_matrix=[[566., 0., 320.], [0., 566., 240.], [0., 0., 1.]],
        distortion_coefficients=[0.]*5, base_from_upright_optical=transform.tolist(),
        status='simulation_only_ideal_camera', robot_frame_validated=True,
        steering_supported=True, provenance='Expanded simulation URDF + specified ideal camera; NOT real-robot validation')
    validate_calibration(calibration)
    for mesh in urdf.iter('mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith('package://pinky_description/'):
            mesh.set('filename', str(Path(description)/filename.split('pinky_description/', 1)[1]))
    for gazebo in urdf.findall('gazebo'):
        for sensor in list(gazebo.findall('sensor')):
            if sensor.get('type') != 'camera':
                gazebo.remove(sensor)
                continue
            set_text(sensor, 'topic', '/lane_sim/camera/image_raw')
            set_text(sensor, 'update_rate', 15)
            set_text(sensor, 'gz_frame_id', 'front_camera_optical')
            camera = sensor.find('camera')
            set_text(camera, 'horizontal_fov', 2*math.atan(320/566))
            set_text(camera.find('image'), 'width', 640)
            set_text(camera.find('image'), 'height', 480)
            lens = element(camera, 'lens')
            intrinsics = element(lens, 'intrinsics')
            for key, value in dict(fx=566, fy=566, cx=320, cy=240, s=0).items():
                element(intrinsics, key, value)
        for plugin in list(gazebo.findall('plugin')):
            if 'LampControl' in plugin.get('name', ''):
                gazebo.remove(plugin)  # Custom lamp plugin is irrelevant to lane tests.
            elif 'DiffDrive' in plugin.get('name', ''):
                for key, value in dict(topic='/lane_sim/cmd_vel', odom_topic='/lane_sim/odom',
                    tf_topic='/lane_sim/tf', frame_id='odom', child_frame_id='base_link',
                    odom_publish_frequency=30).items():
                    set_text(plugin, key, value)
            elif 'JointStatePublisher' in plugin.get('name', ''):
                set_text(plugin, 'topic', '/lane_sim/joint_states')
    gazebo = element(urdf, 'gazebo')
    plugin = element(gazebo, 'plugin', filename='gz-sim-odometry-publisher-system', name='gz::sim::systems::OdometryPublisher')
    for key, value in dict(odom_topic='/lane_sim/ground_truth', odom_frame='world',
        robot_base_frame='base_link', dimensions=3, odom_publish_frequency=30).items():
        element(plugin, key, value)
    robot_file = output/'pinky_lane.urdf'
    ET.ElementTree(urdf).write(robot_file, encoding='unicode', xml_declaration=True)
    conversion = subprocess.run(['gz', 'sdf', '-p', str(robot_file)], capture_output=True, text=True, check=True)
    sdf_robot = ET.fromstring(conversion.stdout)
    model = sdf_robot.find('model')
    if model is None or not model.findall('joint'):
        raise ValueError('URDF conversion did not preserve wheel joints')
    model.set('name', 'pinky_lane')
    set_text(model, 'pose', '0 0 0.003 0 0 0')
    root = ET.Element('sdf', version='1.10')
    world = element(root, 'world', name='lane_course')
    physics = element(world, 'physics', name='default', type='ignored')
    element(physics, 'max_step_size', .001)
    element(physics, 'real_time_factor', 1.)
    element(world, 'gravity', '0 0 -9.81')
    for filename, name in [('physics', 'Physics'), ('user-commands', 'UserCommands'),
                           ('scene-broadcaster', 'SceneBroadcaster'), ('sensors', 'Sensors')]:
        plugin = element(world, 'plugin', filename=f'gz-sim-{filename}-system', name=f'gz::sim::systems::{name}')
        if name == 'Sensors':
            element(plugin, 'render_engine', 'ogre2')
    scene = element(world, 'scene')
    element(scene, 'ambient', '.7 .7 .7 1')
    element(scene, 'background', '.75 .75 .75 1')
    light = element(world, 'light', name='sun', type='directional')
    element(light, 'pose', '0 0 3 0 0 0')
    element(light, 'diffuse', '.8 .8 .8 1')
    element(light, 'direction', '-.2 .1 -1')
    visual_box(world, 'floor', [0,0,-.025,0,0,0], [8,8,.05], '.18 .18 .18 1', True)
    visual_box(world, 'background_wall', [3,0,1,0,0,0], [.05,8,2], '.75 .75 .75 1', True)
    points = course_points(course)
    extension = points[-1]-points[-2]
    painted_points = np.vstack([points, points[-1]+.50*extension/np.linalg.norm(extension)])
    boundaries = {}
    for side, sign in [('left', 1.), ('right', -1.)]:
        curve = offset_polyline(painted_points, sign*lane_width/2)
        boundaries[side] = curve.tolist()
        # Split a straight stripe so an optional missing segment is real, not
        # a perfect simulator mask substituted for camera/YOLO perception.
        if course == 'single_gap' and side == 'left':
            segments = [(curve[0], [.60, lane_width/2]), ([.95, lane_width/2], curve[-1])]
        else:
            segments = list(zip(curve, curve[1:]))
        for index, (a, b) in enumerate(segments):
            a, b = np.asarray(a), np.asarray(b)
            delta = b-a
            centre = (a+b)/2
            visual_box(world, f'{side}_stripe_{index}', [*centre,.0007,0,0,math.atan2(delta[1],delta[0])],
                       [np.linalg.norm(delta)+.002,.020,.001], '.95 .95 .95 1')
    world.append(model)
    world_file = output/'lane_course.sdf'
    ET.ElementTree(root).write(world_file, encoding='unicode', xml_declaration=True)
    calibration_file = output/'simulation_calibration.json'
    calibration_file.write_text(json.dumps(calibration, indent=2))
    metadata = dict(course=course, lane_width_m=lane_width, centre=points.tolist(),
                    boundaries=boundaries, ground_truth_topic='/lane_sim/ground_truth',
                    calibration=str(calibration_file), description=str(description),
                    notes='Physics model reused from URDF. Camera/paint/lighting are idealised, no physical success claim.')
    (output/'course.json').write_text(json.dumps(metadata, indent=2))
    return str(world_file), str(calibration_file)
