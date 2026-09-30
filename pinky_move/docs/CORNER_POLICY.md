# Camera-only normal, S-bend and sharp-corner policy

**Current default:** see “2026-09-30: staged right-angle entry and pivot turn”
below. The original curve-planner description remains as reference for other
angles and for `corner_staged_turn: false`.

This extends `lane_autonomy`; it is not a new ROS node or a Nav2 integration.
YOLO stays on the PC. All decisions and Twist generation remain on the robot.
No automatic enable or lidar activation is added. A supervised 20-second run
was performed before the final endpoint-support change; that final change has
only been validated without motor output so far.

## Five integrated stages

1. **Observed boundary first.** `MetricLaneTracker.last_observation` contains
   the current trusted real boundary and its role/width, not the offset centre.
   Empty detections, identity ambiguity, target discontinuity and invalid fits
   cannot be overridden by the corner planner. Single-mask centre-offset fold
   failures may use the alternative planner; multi-mask failed associations may
   not. With a successful measured pair the other real boundary is also checked.
2. **Localized heading change.** Ordered samples are resampled along arc length.
   Approximately 5 cm chords suppress pixel-scale derivatives. A change of at
   least 50 degrees within 12 cm is only a candidate. Two TLS line fits must
   establish observed entry and exit legs (each >=5 cm), residual <=8 mm, and a
   nearby intersection with compatible directions. Missing exit is UNKNOWN.
   These are initial tuning thresholds, not experimentally established values.
3. **S-bend distinction.** Significant heading changes of both signs classify
   S_BEND and retain ordinary normal-offset following. Straight and smooth curves
   remain NORMAL. A single endpoint-angle comparison cannot identify an S bend.
4. **Temporal confirmation.** Three distinct fresh observations with consistent
   side, turn sign, odom-frame corner location (5 cm gate) and exit heading
   (20 degree gate) are required. Gap >0.6 s resets the confirmation count;
   duplicates do not count. Odometry is interpolated at camera-header time, with
   <=50 ms nearest-sample fallback; its latest receipt/source must be fresh.
   Frame changes and implausible odometry jumps clear landmark history. No
   odometry means no sharp-corner confirmation, not guessed stationary motion.
5. **Approach/turn/exit.** Confirmed distant corners are APPROACH; those within
   30 cm radial range are TURN. A valid ordinary centre path is retained through
   a far CORNER_CANDIDATE and APPROACH, capped at corner speed. Only when the
   corner reaches lookahead plus the robot's front extent, or the ordinary
   centre path fails, does the confirmed corner use a newly validated local path
   rather than a timed in-place turn. That mode replaces the ordinary centre path:
   one trusted visible boundary supplies the corner shape, and offsets from
   the physical robot limit through lane centre are searched. The closest
   fully valid offset is preferred. Tangent-connected cubic Bezier candidates
   are tested for crossing, forward support, sampled swept-rectangle containment
   and angular-rate feasibility. Speed is capped by `corner_speed_mps` and further reduced
   for curvature. No candidate means stop. Exit requires a fresh ordinary path
   and three frames aligned with the odom-frame exit direction before returning
   to NORMAL. Lost corner/exit evidence is not replaced with a stored path.

Corner candidates wait for confirmation. They do NOT trigger the older blind
single-boundary recovery spin. An active corner also disables empty-mask path
holding. The existing frame/inference watchdogs still apply; corner odometry
has a separate 0.3 s timeout evaluated on control ticks.

## Important current geometry blocker

`lane_width: 0.154` is an old **estimate**, not a tape measurement. The existing
navigation footprint is x=[-0.090,0.060], y=[-0.075,0.075] metres. The painted-line
clearance is now 0 mm per side, so the minimum width is 0.150 m before checking
rotation. The footprint sweep can still reject a 0.154 m sharp corner. A
confirmed pair is sampled every 0.5 s; three consistent samples establish a
stored width, which a one-line observation may reuse. A sudden narrow or wide
pair is held for remeasurement rather than replacing the stored width.
In the supervised 20-second test, the robot followed a single line at 0.03 m/s
on the straight but stopped at the corner with `observed legs too short for
swept footprint`; removing lateral clearance does not remove this independent
entry/exit-support check.
The entry/exit precheck now uses the actual configured rear/front extents,
a 5 mm endpoint pad and a 5 mm nondegenerate residual. Its former extra 1 cm pad plus 2 cm residual
duplicated the subsequent swept-footprint support check. Every sampled robot
footprint still has to remain within the observed line support.

Measure stripe-centre separation and actual robot footprint before changing
these parameters. Do not shrink the footprint or enlarge the lane merely to
make a plan pass. Nominal URDF/footprint geometry and one-point distance checks
do not prove calibrated free space. Narrow or poorly observed corners may be
physically infeasible or require more observations. Entry/exit segments must
also cover the swept front/rear; short 5 cm legs can classify a corner but are
not automatically sufficient to execute it.

Single-boundary containment uses the configured width prior. It is not a
measurement of the missing boundary. Swept checks are discrete and lane-only;
they do not detect obstacles, guarantee collision freedom or certify tracking.
The follower still needs supervised physical validation. Actual motion depends
on latency, slip, calibration and available observations.

## Configuration and debugging

Pinky2 now uses a corner linear cap of 0.05 m/s and a separate
`corner_max_angular_speed` of 0.25 rad/s for corner approach/turn/exit targets.
Ordinary single-line tracking retains its 0.03 m/s cap and ordinary steering
retains its 0.15 rad/s cap. Curvature, target distance, and observed geometry
can still reduce actual corner speed below these upper bounds.

All `corner_*` parameters are in `config/lane_autonomy.yaml`. Restart the node
after changing geometry/classification thresholds (policy constructed on reset).
`corner_enabled: false` returns to the earlier follower; it is not a fix for
insufficient clearance. Defaults do not change the 22 cm ordinary lookahead,
calibration, existing speed limits or lidar setting.

`/lane_autonomy/debug_image` shows mode, left/right turn, heading change, radial
corner distance and confirmation count. Accepted Bezier paths use the existing
centre-path/target overlay. `/lane_autonomy/status` includes the same diagnostics
and a rejection reason. Inspect `CORNER_CANDIDATE`, `APPROACH`, `TURN`, `EXIT`,
`EXIT_WAIT`, `NORMAL`, `S_BEND`, or `UNKNOWN` before any supervised drive.

## Motor-free verification

`test_lane_corner.py` covers left/right corners, straight/S/noisy/short-exit
observations, true footprint conflicts, width and support rejection, curvature
speed limits, temporal confirmation, duplicate/gap rejection, odom compensation,
and exit hysteresis. Controller tests cover centre-fold replacement only after
confirmation, identity-error blocking, no blind recovery spin/empty hold, camera
time odometry interpolation, resets and stale data. These are software tests,
not evidence of course completion.
# 2026-09-30: staged right-angle entry and pivot turn

Confirmed 65–115 degree corners now use `corner_staged_turn: true` in the
robot configuration. Other corners retain the existing curve planner. This
change is camera-based local control, not Nav2.

1. Three consistent observations establish entry/exit directions and width.
2. Intersect the inward half-width offsets of those directions to calculate
   a **base_link centre pivot**, not the painted boundary corner. Reject a
   pivot beyond the observed entry span or a badly misaligned approach.
3. Approach the pivot at at most `corner_entry_speed_mps: 0.03`. Store it in
   odometry coordinates and require each new same-side boundary to match the
   transformed observation. No fresh boundary/odometry means stop. Entry has
   a 20-second budget and never commands reverse motion.
4. Within `corner_pivot_tolerance_m: 0.025`, command zero translation and
   brake for `corner_brake_seconds: 0.3` before turning. At control frequency,
   recheck current odometry so a stale inference target cannot carry the robot
   through the pivot.
5. Turn in place toward the observed exit heading, bounded by
   `corner_max_angular_speed: 0.25`. Stop at the heading, on pivot drift, on
   image/result/odometry expiration, or at `corner_spin_timeout_s: 20.0`.
6. Resume entry-speed following only after three fresh near-path confirmations
   (forward target within 30 cm) and heading alignment within 12 degrees.
   Seeing a distant stripe does not release the turn. Reset/enable is required
   after an exhausted time budget; new frames do not renew it.

The stationary mode intentionally does **not** require a successful swept
Bezier path. It is **not an obstacle detector or a guarantee that rotating
corners of the physical body remain inside painted lines**. Lidar stopping is
still disabled in this project's existing configuration. Test first with a
supervisor, clearance around the robot, and an immediate stop available.
Braking is time-based, not measured wheel standstill. The odometry base origin
must match the physical drive rotation centre; deployment does not validate
that mounting assumption. No physical test was performed for this change.

Parameters are loaded when the node starts; restart to apply. Setting
`corner_staged_turn: false` restores the existing Bezier corner path policy.

## Offline startup follow-up and deployment status

The subsequent robot startup log contained `Lane=2` and
`normal_pairs=1, minimum=3`, using the previously deployed curve-planner code,
not this new staged-turn implementation. The original normal-pair loop stopped
at the first broken run, even if that run contained only one correspondence.
It now checks subsequent contiguous runs independently, keeping minimum three
correspondences, 2 cm support, width consistency and nonintersection checks.
It does not bridge gaps or accept one correspondence as a lane. Synthetic
regression tests reproduce the first-singleton rejection; the latest original
camera frame was not retrieved before robot disconnection, so this is not a
confirmed replay of the user's exact scene.

Some logged robot postprocessing durations exceeded 1.6 s while PC processing
was typically tens of milliseconds. Debug Image publishing now uses
BEST_EFFORT, depth 1, compatible with the existing `cam_view.py` sensor QoS.
Other viewers must select Best Effort. New `geometry` and `debug_publish`
timings distinguish numerical processing from visualization work. This
removes reliable-delivery backlog from the debug stream, but is not proof that
the measured delay has been eliminated. Image/inference/odometry stop limits
were not increased or disabled.

The staged-turn and startup-follow-up changes are locally tested only. Pinky2
became unavailable before deployment. The running robot, if any, still has the
previous deployed version. Do not interpret Git upload as robot deployment or
successful physical driving. Deploy after reconnection, build, then restart
explicitly; do not automatically enable motion during deployment.

Motor-free functional checks (from the ROS workspace):

```bash
source /opt/ros/jazzy/setup.bash
cd pinky_move
PYTHONPATH="$PWD:$PYTHONPATH" python3 -m pytest -q test \
  --ignore=test/test_flake8.py --ignore=test/test_pep257.py \
  --ignore=test/test_copyright.py
```

These are functional regressions, not a claim that legacy formatting/lint
checks or real-world vehicle safety have passed.
