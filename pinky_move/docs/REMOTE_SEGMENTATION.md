# Pinky2: PC segmentation, robot-local motion control

The existing segmentation weights are unchanged. The PC performs ONLY YOLO;
Pinky2 still owns calibration, left/right identity, centre-path generation,
Pure Pursuit, enable/disable, speed limits and timeout stops. No Nav2 is added.

The robot retains its original camera frame and source timestamp. It offers a
JPEG through a **127.0.0.1-only** mailbox. The approved PC connects using SSH to
`pinky@192.168.45.22`; camera data is not exposed on a public HTTP socket or new
additional DDS image topic. The launch reuses the existing
`/pinky/camera/image_raw/compressed` stream from `cam_stream.py`; do not start
a second camera process. The result contains lane/crossline pixel polygons in the
original image size, frame token and model hash; it contains no motor command.
Both model hashes must match. PC clock synchronization is not required.

## 1. Pinky2 terminal

Stop the previous lane launch first. This launch starts hardware and camera,
so do not simultaneously start another hardware bringup or cmd_vel publisher.
`cam_stream.py` uses the established Picamera2 path (640x480, 180-degree
rotation, JPEG at 15 fps). The lane node now subscribes to that compressed
topic directly; it no longer starts `csi_camera_node`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=22
ros2 launch pinky_move lane_autonomy_pinky2.launch.py
```

Defaults now select remote inference, `/usr/bin/python3`, and an end-to-end
result timeout of **0.8 seconds**. The robot does not import YOLO or torch.
The default target is now **22 cm radial lookahead**, with boundary fitting
still limited to the observed **14-48 cm forward interval**. Row/column
candidates must pass fitting before selection. The virtual centre remains a
half-lane-width normal offset (left boundary -> right offset, and vice versa),
not a fixed leftward bias. No unseen near-floor segment is extrapolated.
The existing model file is read only to compute its SHA256 identity.
Driving starts **disabled**. Keep the existing robot CycloneDDS configuration.

### Corner and S-bend geometry update

Boundary samples now retain their connected order and are fitted as
`(x(s), y(s))`, where `s` is distance along the observed boundary. Four centimetres
of **curve length**, not four centimetres of forward-x change, is sufficient
support. This handles a stripe that turns sideways without mistaking small
decreases in x for a self-crossing. No samples are sorted by x or joined across
an out-of-range gap. Initial side uses the first observed 2 cm of the boundary;
subsequent geometry/optical-flow association retains the side through a bend.

Single-line centres use the local normal `(-dy/ds, dx/ds)` times half the lane
width. Actual tangent reversals/cusps and self-intersections are still rejected.
Only a connected forward prefix at least 6 cm long before a distant offset
failure may be used; the debug image reports `center_path truncated ...`.
Invalid near geometry still stops. A shortened visible path uses an adaptive
endpoint; no unseen path is extrapolated to reach the 22 cm lookahead circle.
The 14–48 cm interval bounds **observed boundary samples**, not the normally
offset centre, which can legitimately lie closer to the axle in a corner.

Forward-monotone pairs retain common-x averaging. Lateral pairs require unique,
ordered normal intersections and consistent separation; ambiguous pairs are
not averaged. Pure Pursuit follows the connected centre in order. There is no
new timed “line ends, turn left” rule and no Nav2 dependency.

The measured stopped-corner snapshot produced approximately 18.9 cm from the
wheel axle to the bend centre, compared with the operator's approximate 19 cm
measurement. Camera-to-base translation is already included. This one point
does not validate the entire calibration or actual driving performance.

Motor-free regression coverage includes the actual re-inferred snapshot mask,
mirrored corners, S-bend steering signs, normal offsets, lateral pairs, genuine
cusps, crossings, stale frames and existing controller recovery. On-course
tracking and timing still need supervised low-speed verification. Speeds,
lookahead, calibration and watchdog settings are unchanged by this update.

## 2. Local PC terminal (192.168.45.52)

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
export ROS_DOMAIN_ID=22
ros2 launch pinky_move lane_inference_pc.launch.py
```

This uses `/home/tory/venv/omx/bin/python` and the matching weights at
`/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt`.
Do not substitute `Downloads/lane2/best.pt`: its hash differs from the deployed
robot model. CPU, imgsz=320, four torch CPU threads, JPEG quality=85.
No NVIDIA/CUDA installation is required for this PC.

SSH uses existing keys and strict known-host verification. If authentication
fails, check `ssh pinky@192.168.45.22` interactively; do not disable host checking.
Only one worker can own local port 18765. Ctrl+C closes its owned SSH tunnel.
On tunnel failure the worker exits; restart it after restoring the network.

## 3. Preview, then enable on Pinky2 in another terminal

With the same ROS environment sourced, inspect `/lane_autonomy/debug_image`
and `/lane_autonomy/status`. `DISABLED` remains normal until you enable.
The debug `processing` field in remote mode is source-frame age when the reply
is consumed, including camera-message age, transport, PC inference and waiting;
the worker's `inference` field excludes robot-side path/controller computation.

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: true}"
```

Stop:

```bash
ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: false}"
```

Disable first, then Ctrl+C the PC and robot launches. Never enable as part of
automatic deployment. Lidar stopping remains disabled as previously requested;
this system is not an obstacle-avoidance or safety-certified controller.

## Timeout and replay protection

- One pending frame and one reply slot; no historical image queue.
- Robot-generated session/sequence token; only the matching reply is accepted.
- Robot's steady clock determines age from the same-host camera timestamp.
- A late reply, repeated token, old enable epoch or wrong model cannot renew age.
- Stop when the source frame expires (0.8 s default), even if the PC repeats data.
- Invalid polygons/shape/NaN/model errors cannot reach the controller.
- The local motor bringup also retains its cmd_vel watchdog (default 0.5 s).
  This is separate from perception freshness; a fresh cmd_vel from stale vision
  must still be rejected by the lane controller.
- Timer-based software stops are best-effort, not hard real-time guarantees.

## Return to the previous on-robot inference mode

Stop both launches first. On Pinky2:

```bash
ros2 launch pinky_move lane_autonomy_pinky2.launch.py \
  remote_inference:=false \
  python_executable:=/home/pinky/venv/yolo/bin/python \
  result_timeout:=2.0
```

## Motor-free integration validation

On Pinky2 use `start_hardware:=false cmd_vel_topic:=/lane_autonomy/validation_cmd_vel`.
Do not enable. This starts only the camera and controller, with velocity output
disconnected from `/cmd_vel`. The PC worker also supports `--max-results N` for
bounded tests via `python -m pinky_move.lane_inference_worker`.

### Completed verification

- Local and robot suites: 113 tests passed, including replay, stale replies,
  wrong weights, malformed polygons, disable epochs and timeout zero output.
- Actual Pinky2 camera -> PC CPU -> robot: 27 returned frames, no model errors.
- PC processing plus reply POST: 0.136-0.365 s in this short run. This is NOT
  pure model time or a guaranteed control rate; total delay includes transport
  and robot-side geometry. One live debug frame showed source age 0.38 s.
- Live controller stayed DISABLED. All 228 observed validation velocity
  messages were zero; motor bringup was not started and `/cmd_vel` was not used.
- Test processes and SSH tunnel were stopped after validation. Actual moving
  lane-following quality has NOT been validated by these stationary tests.
