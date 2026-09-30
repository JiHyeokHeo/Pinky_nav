# Continuous lane operation

Run on Pinky2 (192.168.45.22), Domain 22. Source ROS Jazzy and
~/pinky_pro/install/setup.bash before running commands.

    ros2 launch pinky_move lane_autonomy_pinky2.launch.py

Launch includes hardware bringup, the existing Picamera2 cam_stream.py and YOLO. Do not start duplicate
bringup or competing cmd_vel publishers. Starts disabled; enable explicitly:

    ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: true}"

Stop explicitly before shutting down the launch:

    ros2 service call /lane_autonomy/enable std_srvs/srv/SetBool "{data: false}"

Defaults: max_enabled_seconds=0 (continuous), linear_speed=0.03 m/s,
maximum_angular_speed=0.15 rad/s, use_lidar_guard=false.
metric_trial_enabled is retained for compatibility but no longer gates driving.
A positive max_enabled_seconds can still bound a run; configure before enabling.

Configured lane_width permits starting with one identifiable boundary. The
default 0.154 m is a prior estimate: replace it with the actual course width.
Two consecutive real pairs replace the configured width with measured width.
A continuously tracked single boundary is used without an elapsed-time cap
(metric_single_line_timeout=0), at no more than 0.01 m/s. The centre and virtual
boundary are recomputed from each new detection; the last measured width is
retained until a real pair is confirmed again. Empty detections briefly retain
the previous target at at most 0.01 m/s: hold until 0.3 s after the last valid
result, then ramp to zero by 0.8 s. This is not odometry-compensated prediction.
Ambiguous tracking or stale camera/inference causes a stop. Driver command
timeout is 0.5 seconds. This is not general obstacle avoidance or a hardware
emergency stop; driver/process failure is not covered by its software timer.

Nominal URDF optical mounting and floor-distance accuracy remain unverified.
Continuous operation does not imply validated autonomous navigation.
After a tracking gap, continuous mode can reacquire the same locked side using
configured width after three consistent observations (each gap <=0.8 s).
The robot stays stopped during confirmation. Empty/ambiguous observations reset
confirmation; the opposite side cannot silently take over. A failed pair can
fall back only to a unique geometric match of the previously tracked boundary.
An explicitly positive single-line timeout still requires a real pair on expiry.
Reacquisition first matches the last observed geometry, not its current y sign;
a curved boundary near robot centre therefore retains its locked role. A new
unmatched boundary still cannot replace it. Fold errors identify `center_path`
versus `virtual_boundary` and appear on the debug image. A folded centre path
still stops motion. A folded full-width virtual boundary is omitted (including
from matching references) while a valid centre path may continue at the existing
single-line speed cap. Recovery commits only after all control geometry checks
succeed; a failed attempt retains the previous locked side and recovery state.

Recent segmentation images now help preserve left/right identity through small
camera motions. Confirmed metric boundaries seed their mask roles; backward
optical flow aligns each previous 160x120 mask with the next raw frame. Only a
unique overlap within 0.8 seconds supplies a role hint. A stale or ambiguous
match cannot steer the robot by itself: floor projection, centre-path checks,
and the existing three-frame recovery still apply. An empty image clears the
reference instead of carrying a lane label indefinitely.

On the local PC (Domain 22), view the robot's already annotated raw image:

    python3 ~/Downloads/cam/cam_view.py --topic /lane_autonomy/debug_image --transport raw --calibration ''

Do not pass --model: the robot already performs inference. Press r to record,
s for a snapshot, q to close. This viewer sends no motion commands. Frames are
updated at inference completion rate, not necessarily camera frame rate.
After code/default changes, restart the launch; an existing process retains
its loaded code and parameters. Deployment does not enable motion.

## Camera-only metric controller (local update, 2026-09-29)

No Nav2 actions are used. Existing YOLO masks feed `floor_curves`, then
`fit_boundary` and `classify_lanes` in metric_lane.py. robot_projection.py
undistorts each pixel, rotates its optical ray using calibration extrinsics,
and intersects the ground plane directly (no BEV). Coordinates are base_link
x-forward/y-left in metres. Ground z is -0.028 m for this URDF, not zero.

`center_from_pair` averages two boundaries on their common observed interval.
For one boundary, `normal_offset` uses unit normal (-dy/dx, 1): left boundary
offset is -width/2, right boundary +width/2. The curve is rebuilt every result.
`select_lookahead` intersects the forward path with a lookahead circle; a short
visible path uses its observed endpoint instead of extrapolation. `pursuit`
computes omega = v * 2*y/(x*x+y*y), with angular speed saturation. Existing
lane_autonomy.py publishes Twist and owns enable, timing and stop handling.

Parameters in config/lane_autonomy.yaml:

| Parameter | Default | Purpose |
| --- | --- | --- |
| linear_velocity | -1 | -1 uses legacy linear_speed=0.03 m/s |
| lookahead_distance | -1 | -1 uses metric_lookahead_m=0.22 m |
| lane_width | 0.154 | Initial normal width, metres; measure your course |
| metric_path_min_m | 0.14 | First forward distance used for path fitting |
| metric_path_max_m | 0.48 | Last forward distance used for path fitting |
| path_polynomial_degree | 3 | Maximum fit degree; simpler fit preferred for noise |
| projection_max_forward_m | 2.0 | Accepted ground projection range |
| single_line_max_speed | 0.01 | Single-boundary speed cap, m/s |
| lane_loss_hold_seconds | 0.3 | Short empty-detection hold |
| lane_loss_stop_seconds | 0.8 | Zero velocity deadline from last valid detection |
| near_pair_max_m | 0.28 | Both real boundaries must overlap inside 14-28 cm |
| single_line_turn_speed | 0.08 | In-place yaw rate while one near side is tracked |
| single_line_turn_delay_seconds | 3.0 | Continuous same-side detections required before turning |
| single_line_turn_seconds | 20 | Maximum duration of a turn episode |
| single_line_visibility_timeout | 0.6 | Stop yaw if the near side is not refreshed |

The requested example linear_velocity=0.2, lookahead_distance=0.7,
lane_width=0.6 is supported, but is NOT the robot default. Change parameters
with metric_path_max_m >= lookahead_distance when extending lookahead. The
14 to 48 cm interval is independent of the 22 cm target distance; it only limits
the fitted path support. If the observed centre offset itself folds within
this interval, steering remains stopped.

Boundary sampling evaluates BOTH rows and columns, first rejecting candidates
whose polynomial fit fails within the configured 14-48 cm support. It prefers
lower RMS residual, then greater observed forward support among fits within
2 mm RMS of the best. A longer invalid candidate can never displace a valid
shorter one. If both valid candidates have at least 4 cm common support but
disagree by more than 3.5 cm median lateral distance, the mask is rejected.
Column sampling removes <=2-pixel connectors with a 3x3 opening before tracking
disjoint runs. Equally supported branches and broad blobs remain rejected.
Sampling orientation never decides left/right identity. Floor projection, the
14-48 cm support range, path-fold checks and drive limits are unchanged. This
remains a local y(x) model, not support for arbitrary loops or hairpins.

Multiple model instances do not necessarily form a usable left/right pair.
When pair construction fails, a UNIQUE boundary matching the locked side via
geometry or recent image flow can use the normal-offset single-side path. The
source mask index is retained for image tracking. No common pair support is
invented; ambiguous multiple matches and opposite identities still stop motion.

At cold start, left/right identity comes from the nearest observed 2 cm forward
band, independently of steering lookahead. A curve crossing the robot centre
at 28 cm must not invalidate a clearly left/right boundary close to the robot.
The nearest band still needs to stay entirely beyond the +/-1 cm centre margin;
otherwise startup waits instead of guessing. Established identities continue to
use frame-to-frame tracking, not repeated cold-start classification.

After three seconds without a usable centre path, while the same near boundary
remains identifiable, the controller turns in place toward the lane interior:
left boundary -> right turn, right boundary -> left turn. Brief single-line
detections with a valid centre path retain the existing slow path follower.
`single_line_turn_requires_path_loss=true` prevents a healthy single-line path
from being interrupted merely because the second boundary is outside the image.
An empty/ambiguous frame or a usable near pair resets the three-second timer.
This can also run while an offset centre path is temporarily invalid, provided
the fresh boundary matches the previously tracked side. Boundary identity time
is refreshed independently of path validity: a rejected folded centre does not
expire a continuously matched boundary or reset its completed three-frame
reacquisition. This does not refresh the driving target; invalid geometry still
stops forward motion. Empty, ambiguous or stale observations remain rejected.
It stops turning after
two consecutive measured left/right paths overlap inside 14-28 cm. A pair
seen only farther away cannot release the turn. The turn episode lasts at most
20 seconds; an absent near boundary stops yaw after 0.6 seconds, and the
20-second limit stays exhausted until a near pair or an explicit re-enable.
With `single_line_turn_requires_path_loss=true`, a newly VALID single-side
centre path may also release the recovery state after three consecutive fresh
observations of the same side with near support in 14-28 cm and consistent
geometry. This permits the existing slow follower to resume even after turn
timeout. Held targets, distant-only geometry and visible-but-invalid paths
cannot release recovery. This single-side exit keeps the rotation budget
exhausted: another spin needs a confirmed near pair or operator re-enable.

Check the debug view while disabled and validate physical distances and steering before increasing
speed. Nominal extrinsics still require physical validation. A local cubic
handles forward S-shaped paths; this is not an intersection, U-turn or global
route planner. Total loss holds a stale target only briefly, not indefinitely.
