# Camera-only normal, S-bend and sharp-corner policy

## Local lane map in rqt (diagnostic only)

Run the established robot lane launch and PC inference launch. No enable call
is required for preview, but fresh hardware odometry and camera-time projection
are required for accumulation. On the PC:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws/pinky_pro/install/setup.bash
ROS_DOMAIN_ID=22 ros2 run rqt_image_view rqt_image_view
```

Select `/lane_autonomy/local_map` from the topic dropdown. Select
`/lane_autonomy/debug_image` in a second image view for the camera overlay.
Use raw image transport and Best Effort QoS if the viewer exposes that choice.
The publisher is `sensor_msgs/Image`, bgr8, depth 1/Best Effort, with camera stamp
and base_link frame ID. `publish_debug_image` must remain enabled. Rendering
uses the existing latest-only debug worker, at most 2 Hz and only with a map
subscriber; it adds no planner or velocity publisher.

Map is 480x480 pixels at 2.5 mm/pixel, 10 cm grid, x forward upwards and y left
to the left. Robot origin is pixel (240,400); display covers approximately
20 cm behind to 1 m ahead and +/-60 cm laterally. LEFT is green, RIGHT orange,
current centre path cyan, target magenta, fixed corner pivot red. Old observed
curves fade over five seconds, bounded to 50 observations. They remain recorded
in odom and are transformed to base at current camera time. Reset/odom jumps
clear them. Missing fresh odometry hides geometry rather than showing it in a
wrong frame. No recent measurements shows an empty-grid message. History can
contain measured boundaries whose centre-path generation failed; it is not a
validated free-space map. It does not fuse geometry, run SLAM, correct drift,
or supply motion decisions. A static still-image view does not validate actual
centimetre accuracy; projection/odom errors may appear as overlapping ghosts.

## 2026-10-04: bounded measured history and entry diagnostics

Entry arrival update: pivot and entry yaw are fixed in odom at confirmation.
Forward remaining distance and lateral error are resolved on that fixed axis.
Within +/-25 mm along and +/-25 mm lateral (default tolerance), brake then
turn; crossing the target plane stops forward commands at control rate even
before another camera frame. Overshoot beyond tolerance, excessive lateral
deviation or arrival outside lateral tolerance latches an ENTRY_FAULT until
explicit reset. No reverse correction or goal relocation is performed. Entry
requires at least 5 mm progress per three seconds; a deadline also travels with
the control target so it stops without waiting for another inference. Existing
20-second entry and spin limits remain. Diagnostics expose fixed odom pivot,
robot position, signed remaining distance and lateral error. The exact ordinary
`no forward centre path` error is now separated for confirmed staged control,
like `inferred target discontinuity`; fresh measured same-side boundary and
all fragment/odometry checks are still mandatory. Physical validation pending.

Confirmed staged-control separation: an exact `inferred target discontinuity`
error from a single-mask ordinary tracker may now pass its fresh measured
boundary to an already-active staged turn. The rejected ordinary target itself
is never used. The staged controller keeps its odom pivot and checks same-side
fragment correspondence, fresh odometry, drift and deadlines as before. No
exception applies before staged confirmation, to multiple masks, identity errors
or missing observations. `ordinary_target_ignored` exposes this distinction in
debug state. The ordinary tracker threshold is unchanged. Regression tests
cover valid staged entry plus seven rejection paths. No new physical trial was
performed for this separation change.

Supervised 20-second follow-up reached staged ENTRY, then rejected a fragment
with 100% correspondence, ~5.2 mm median error and 39.32 mm overlap. Strong
fragments may now use 30 mm minimum support: >=90% contiguous correspondence,
90th-percentile distance <=15 mm and 10th-percentile absolute tangent cosine
>=0.95. Otherwise the 40 mm threshold remains. Less than 30 mm, endpoint-only,
and folded correspondence still reject. This does NOT bypass the separately
observed `inferred target discontinuity` gate. Debug labels now annotate each
segmentation as LEFT/RIGHT/UNKNOWN and USED/HINT/OBSERVED; NO TARGET, plus the
global corner STATE, without guessing roles from screen position.

Staged entry fragment matching now uses nearest-segment projection onto the
fixed odometry-transformed measured corner. No forward-x clipping is applied
while turning; reversed sample order is accepted only when arc correspondence
is consistently reversed. At least 70% contiguous samples must match within
35 mm and tangent cosine magnitude >=0.82, with >=4 cm of both measured and
reference support. Endpoint-only proximity, different parallel lines, crossings
and folded correspondence are rejected. Stored landmarks/pivots are not moved
to fit current detections. History-assisted entry retains the original measured
full corner so an exit-only view can match later. Fresh same-side observations,
odometry and all stopping deadlines remain mandatory. The old generic matcher
already supported overlap, but also clipped forward x and required matching
sample direction; this change removes those constraints only for staged turns.
Debug now reports fragment fraction, median error, supported length and rejection
reason. Original metric-tracker errors are also exposed instead of being hidden
behind a generic corner wait message. Static synthetic tests are not proof of
successful physical cornering; moving-robot validation remains pending.

Partial-view follow-up: a NORMAL/UNKNOWN observation that matches a real recent
corner can be tagged PARTIAL_CORNER rather than negative straight-road evidence.
Its real segment must be >=5 cm, shorter than 85% of the full recorded boundary,
TLS residual <=8 mm and tangent within 20 degrees of a recorded leg, in addition
to the existing position/width gates. One actual CORNER plus two fresh aligned
partial observations may now satisfy the three-observation/50% rule. No partial
observation refreshes the original anchor's two-second lifetime. A straight line
without a prior real corner cannot initiate a turn. Missing current boundary
still stops. This supersedes the strict full-CORNER-only voting described below.

Right-angle candidate voting now retains up to 40 observations from the last
two seconds. At least three same-side/same-direction observations at the same
odom landmark (5 cm, exit heading 20 degrees) and a 50% share of all recent
observations can confirm a candidate despite intervening NORMAL frames.
Only actual CORNER detections vote; recovered history never votes for itself.
Old geometry is transformed into the current base frame. Current real boundary
must match within 25 mm, 90% of its samples must be within 35 mm of the stored
curve, and width must agree within 20%. Missing observation/pose, a current S
bend, or mismatching geometry cannot authorize a turn. A single opposite vote
cannot inherit the older direction's votes. Reset clears history. Existing
entry support, odometry/camera freshness, braking, and 20-second spin limits
remain active. This is limited to staged 65–115-degree corners, before entry;
the existing exit state machine is not overridden. No auto-enable is added.

Follow-up live diagnostics resolved the discrepancy: the running tracker used
measured width 0.214 m (not configured 0.154 m), right boundary, ~85-degree LEFT.
Pivot was ~0.200 m ahead, but entry travel was 0.002–0.004 m within ~0.103 m
observed support. The old `travel >= 0.005` incorrectly rejected this valid
endpoint-adjacent pivot. Travel is relative to the visible entry sample, NOT
robot travel. Accept `[0, support]` without extrapolation; preserve alignment,
freshness, width, braking and spin guards. Six mirrored regression cases cover
1, 3 and 4.9 mm support offsets and actual negative-support rejection.

The PC/robot split is unchanged. `LaneHistory` retains at most five successful
measured-boundary observations for two seconds, in odometry coordinates. If
optical-flow association is missing, two distinct historical observations must
agree with the CURRENT measured curve (25 mm match gate, 20 mm ambiguity gap).
Fresh capture-time odometry is required. This supplies a role hint only; current
geometry, identity continuity, watchdogs and corner checks still apply. Empty
images never replay an old target. Odometry resets clear history.

History is consulted before exactly one tracker/corner update per frame, not
as a second processing pass that advances confirmation counts twice.

When staged entry is rejected but a valid ordinary centre path remains and the
corner is beyond lookahead plus the robot front, retain that low-speed ordinary
path. Failed/absent centre paths still stop; no unsupported pivot is permitted.
Corner diagnostics now include boundary side, width/source, entry travel/support
and base-frame pivot coordinates.

Observed live case: approximately 86-degree LEFT was repeatedly confirmed, but
entry was rejected as outside observed support. A fresh local replay of the
captured scene with configured 0.154 m width and right-boundary role placed its
pivot within support. Runtime history/width/role was not exposed in the old log,
so the exact source of this discrepancy is not yet proven. Camera/inference
timeouts were also observed and are not disabled by this change. No physical
driving validation or automatic enable is part of this deployment.

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

## 2026-10-01: 407-image offline dataset audit

Dataset: `/home/tory/Downloads/rosStudy_YOLO.v1i.yolov11`, 285 train / 81 valid /
41 test, all 640x480. Each image is an independent cold start, not a fabricated
sequence. The deployed segmentation model, confidence .55 and input size 320
were held fixed. Predictions were cached by model/settings/image SHA256 so
before/after differences came from geometry code, not changed detections.

Two changes were validated: try strict normal correspondence when monotone
boundaries lack a common forward-x interval (do not double-project its width),
and batch pixel projection/reuse projection within the same update. Existing
minimum support and ambiguity rejection limits were not weakened.

| Independent target generation | Before | After |
| --- | ---: | ---: |
| Train (285) | 239 | 248 |
| Valid (81) | 67 | 69 |
| Test (41) | 36 | 39 |
| All (407) | 342 | 356 |

14 newly available targets; zero previously available targets lost; maximum
coordinate difference on existing valid targets was 0 m. PC geometry p50
62.0 -> 43.6 ms, p95 98.4 -> 61.8 ms in these sequential runs (not a robot
latency guarantee). No unexpected exceptions; 51 images still have no valid
cold-start target. Pixel-union IoU for Lane is .7483 and unchanged; this is
not instance mAP. Training split results are not held-out generalization.
One valid label contains a bounding box rather than a segmentation polygon;
that image is excluded from ground-truth segmentation comparison. Original
images and labels were not edited.

Functional regressions: 208 passed; colcon build succeeded. Physical camera
pose/metric accuracy, temporal tracking, odometry and motion are NOT validated
by these still images. No robot deployment or real driving was performed.

Reproduce the audit with the existing inference virtual environment:

```bash
PYTHONPATH=/home/tory/ws/pinky_pro/pinky_move \
  /home/tory/venv/omx/bin/python test/evaluate_lane_dataset.py \
  --dataset /home/tory/Downloads/rosStudy_YOLO.v1i.yolov11 \
  --model /home/tory/Downloads/yolo_runs/segment/train/weights/best.pt \
  --output reports/dataset_20261001/recheck \
  --cache reports/dataset_20261001/prediction_cache
```

The saved baseline/improved CSVs, overlays and comparison page are in
`reports/dataset_20261001/`. Preserve baseline results rather than overwriting
them with the improved code. `test/compare_lane_dataset.py` generates the
before/after HTML and JSON from those two saved runs.
# 2026-10-04 실기 로그 대응: 제어 지연·회전 반경·추적

실기 로그에서 debug 처리/발행 1.26–1.45초, yaw -0.15 포화,
NORMAL 판정과 차선 재획득 실패를 확인했다. 주행 비활성화 후 수정했다.

- debug 영상은 독립 latest-only worker에서 그려 발행한다. 최대 5 Hz,
  대기 슬롯 1개이며 느린 송신 중 오래된 영상은 쌓지 않는다. 제어 상태는
  복사본만 사용한다. 로그는 debug_enqueue와 debug_worker를 구분한다.
- 일반 metric pursuit에도 v <= angular_limit / abs(curvature)를 적용한다.
  전진을 유지한 채 yaw만 clipping하여 회전 반경이 커지는 문제를 방지한다.
- lane_tracking_gap_seconds=2.0: 좌우 정체성/재획득 유예만 연장한다.
  result_timeout, image_timeout, odometry timeout 및 stale target 주행 중단은
  연장하지 않는다. 긴 곡선을 선택한 다음 프레임에 짧은 hook으로 되돌아가는
  문제는 이전 긴 곡선과 실제로 일치하는 후보가 있을 때만 복구한다.
- 직각 후보는 짧은 heading window에서 변화가 작아도 전체 heading 및
  두 관측 직선의 지지를 검사한다. 새 분기는 65–115도에 한정한다.
  확인된 직각 staged mode는 swept footprint Bezier planner를 호출하지
  않는 것을 좌/우 테스트로 검증했다. 다른 코너의 footprint 검사를
  무조건 삭제한 것은 아니며 실제 회전 공간 안전을 보장하지 않는다.
- 기존 추출이 실패한 단일 마스크에 한해 2px 이하 연결부를 제거한 뒤
  유일한 가까운 큰 성분을 재검사한다. 두 개의 가까운 성분 사이에서
  임의 선택하거나 끊어진 차선을 이어 붙이지 않는다.

로컬 기능 246 tests passed(스타일/저작권 lint 제외). 같은 모델 예측 407장:
384 target 유지, 기존 정상→실패 0, 목표점 좌표 변화 0 m,
corner kind 변경 0. 중간 실험에서 완만한 굽이 2장이 CORNER로 바뀌어
직각 전용 조건으로 제한했다. 최종 보고서는
`reports/control_20261004_comparison.html`이다.
PC geometry p50 66.2 ms/p95 127.5 ms이며 이전 64.4/127.9 ms와 비교 시
일괄적인 처리속도 향상을 주장하지 않는다. 새 worker의 실제 DDS 지연
개선과 해당 S자·직각 코스의 주행 성공은 현장 재검증이 필요하다.
추가 원본 촬영은 이미 다른 차선 형태였으므로 실패 순간의 원본 마스크를
정확히 재현했다고 주장하지 않는다. 합성 브리지/좌우 직각 테스트를 추가했다.

# 2026-10-01 남은 43장 유형별 보완

최종 보고서: `reports/dataset_20261001/remaining_comparison.html`.
원래 320 추론 결과와 이번 320 + 무검출 시 640 재시도 결과를 비교하므로
동일 추론 비교가 아니다. 모델 가중치/confidence 0.55/IoU 0.70은 유지했다.
기존 정상 364장의 목표점 좌표 변화 0 m, 정상→실패 0장.
일반 목표점 생성은 384/407장(train 267/285, valid 77/81, test 40/41).
기능 테스트 235개 통과(스타일/저작권 lint 제외), 로컬 빌드 성공.

수정 내용:

- Cold start에서 좌우 절대 부호만으로 한 쌍이 되지 않으면 두 마스크를
  각각 추출하고 유일한 양의 법선 대응으로 상대적인 좌우를 확인한다.
  기존 정상 쌍이나 이전 추적 역할은 교체하지 않는다. 11장 개선.
- S_BEND에서 전체 offset이 실패하면 관측된 가까운 prefix를 다시 fit한다.
  관측 이탈 최대 8 mm, 정상 offset, 전방 지지 길이 6 cm 이상을 요구한다.
  진짜 급코너에는 적용하지 않는다. S자 4장 개선.
- 4 cm 경계가 자기 자신과 interior 4 cm를 다시 비교하다 실패하는 문제를
  수정했다. 초기화한 동일 관측은 중복 판정하지 않는다. 이후 프레임은
  전체 4–6 cm 경로가 5 mm 이내/접선 cosine 0.95 이상으로 일치해야 한다.
  끝점 하나만 가깝거나 역순인 경우는 거부한다. 1장 개선.
- PC worker는 Lane 무검출 때만 같은 프레임을 640으로 1회 재추론한다.
  confidence는 낮추지 않고 기존 Crossline을 보존한다. token/capture TTL은
  연장하지 않는다. `--retry-imgsz 0`으로 끌 수 있다. 무검출 5장 중 4장 개선.
- 쌍을 만들 수 없을 때 유일한 가까운 선이 다른 관측보다 6 cm 이상
  가까우며 픽셀·metric 좌우가 일치하면 3회 연속 확인 후 단일 차선으로
  전환한다. 추출 불가 마스크가 가까이 있으면 전환하지 않는다.
  측정된 폭이 부적합한 경우는 이 복구를 허용하지 않는다.

남은 일반 목표점 없음 23장:

- 가까운 선 확인 대기 6 + 초기 좌우 확인 대기 4 = 10장.
  각각 저장 이미지의 합성 3회 재생에서 목표점 생성까지 확인했다.
  독립적인 실시간 프레임이나 실주행 성공으로 계산하지 않는다.
- 실제 급코너 6장은 offset 거부를 유지하고 기존 staged corner 테스트로
  별도 경로 전이를 확인한다. 일반 중앙 목표점 성공 수에 포함하지 않는다.
- 좌우 쌍/추출 모호 4, 중앙 겹침 2, 640에서도 무검출 1 = 7장 미해결.
  데이터/연속 관측 없이 방향을 만들어 내지 않는다.

입력 오류/예상 밖 예외 0, 기존 라벨 형식 문제 1장. PC 기하 처리 p50
64.4 ms/p95 127.9 ms/최대 230.8 ms. 복구 계산과 재추론에는 추가 비용이
있으므로 실제 장비의 freshness 예산은 별도 검증해야 한다.
이 변경은 로컬에만 적용했다. 핑키 배포·실주행·GitHub 업로드는 하지 않았다.

# 2026-10-01 S자 초기 경계 보완

기존 row/column 곡선 추출이 모두 실패한 단일 마스크에만 연결 중심선
추출을 적용한다. 1/2 크기 이진 영상에서 bounded Zhang-Suen thinning을
수행하고, 영상 하단의 유일한 가까운 끝에서 연결된 중심선을 따른다.
분리된 주요 성분, 폐곡선, 모호한 시작점은 거부하고 분기점에서는
관측된 접두 구간까지만 사용한다. 픽셀·바닥 좌표를 x순으로 재정렬하거나
끊어진 구간을 연결하지 않는다. 기존 fitting/offset/continuity 검사는 유지한다.

초기 metric 좌우 판정이 중앙과 겹칠 때만 가까운 픽셀의 좌우를 보조
근거로 사용한다. 같은 방향·기하 연속성·0 < 프레임 시간차 ≤ 0.8초가
3회 확인되어야 초기화한다. 같은 시간 재호출, 긴 공백, 무검출 또는
다중 마스크는 확인을 이어가지 못한다. 기존 추적 정체성 우선순위는 유지한다.
관측된 가까운 경로에 맞추는 기존 adaptive lookahead를 그대로 사용한다.

검증: 기능 테스트 222개 통과(스타일/저작권 lint 제외), 로컬 빌드 성공.
같은 YOLO 예측 407장을 독립 초기화하여 359 → 364장에 일반 목표점 생성,
기존 정상의 실패 전환 0장, 기존 목표점 좌표 변화 0 m.
기존 initial boundary=None 9장 중 5장은 목표점 생성, 4장은 곡선 추출 후
offset 접힘 검사에서 정지한다. 중앙 겹침 6장 중 4장은 초기 3회 확인
대기, 2장은 픽셀 보조 근거도 부족하여 계속 거부한다.
저장된 겹침 이미지 1장의 3회 합성 재생은 오른쪽 정체성 확정 및 목표점
생성을 확인했지만 실제 연속 영상·오도메트리를 대체하지 않는다.

보고서: `reports/dataset_20261001/s_curve_comparison.html`.
남은 일반 목표점 없음 43장, 입력 오류/예상 밖 예외 0, 기존 라벨 오류 1장.
PC 기하 처리 p50 63.2 ms, p95 87.5 ms, 최대 174.9 ms로 failure-only
중심선 처리는 추가 비용이 있다. 실제 로봇의 지연은 별도 측정해야 한다.
실주행·로봇 배포·GitHub 업로드는 수행하지 않았다.

# 2026-10-01 전체 이미지 재검사

407장(train 285 / valid 81 / test 41)의 기존 YOLO 예측을 동일하게 사용해
기하 처리만 비교했다. 최종 결과는 `reports/dataset_20261001/edge_verified/`,
전후 비교는 `reports/dataset_20261001/edge_comparison.html`에 저장했다.

- 일반 중앙 목표점 생성: 356 → 359장. 기존 성공 → 실패 0장,
  기존 성공 목표점 최대 좌표 변화 0 m. 최종 분할별 248 / 71 / 40장.
- 중복 차선 2장: 가까운 픽셀의 좌우가 같고 근거리 IoU ≥ 0.90,
  전체 마스크 IoU ≥ 0.85인 경우만 중복을 제거한다. 서로 다른 원거리
  가지는 보존하며 마스크를 합치지 않는다.
- 짧은 끝부분 때문에 접힌 1장: 기존 단일 차선 초기 경로가 실제로
  OffsetCurveError로 실패할 때만 긴 관측 후보를 재시도한다.
  같은 좌우 정체성과 정상 offset 검사를 통과해야 한다. 정상 경로 및
  진행 중인 추적을 임의로 교체하지 않는다.
- 무조건 긴 후보를 우선한 중간 수정본은 기존 정상 사례를 악화시켰다.
  `edge_improved`는 이 실패 실험의 기록이며 최종 배포 후보가 아니다.
- 실제 급코너 6장은 여전히 일반 offset 접힘을 거부한다. 저장 마스크와
  합성 영점 오도메트리로 3회 확인 후 staged corner 진입, 관측 유실 시
  정지를 테스트했다. 같은 정지 이미지를 반복한 단위 테스트이므로
  실제 프레임 연속성이나 회전 공간, 실주행 성공을 입증하지 않는다.
- 일반 목표점이 없는 48장: 좌우 쌍 모호 18, 초기 좌우 미확정 9,
  로봇 기준과 중첩 6, 실제 접힘 코너 6, 무검출 5, 공통 관측 부족 3,
  추적 정체성 모호 1. 미해결 사례를 무조건 허용하지 않았다.
- 입력 오류 및 예상 밖 코드 예외 0. 기존 bbox 형식 라벨 1장은
  segmentation 정답 비교에서 제외했다. 원본 데이터는 수정하지 않았다.
- 기능 테스트 218개 통과(스타일/저작권 lint 3종 제외), 로컬 colcon 빌드
  성공. setuptools의 pytest-repeat egg 경고는 있었지만 빌드는 성공했다.

정지 이미지의 목표점 생성률은 주행 성공률이 아니다. 학습 이미지가
포함되므로 일반화 성능도 아니다. 핑키2 배포, 실주행, GitHub 업로드는
이번 재검사에서 수행하지 않았다. 기존 보정이 데이터 촬영 당시와
물리적으로 일치하는지도 별도 현장 확인이 필요하다.

### 2026-10-04: 부분 대응 및 도착 후 무검출 처리

- 일반 대응점 허용 거리 3.5cm, 연속 대응률 70% 조건은 유지한다.
  40~70% 미만은 대응점 거리 90분위 1cm 이하, 방향 코사인 10분위
  0.95 이상, 관측/기준 연속 길이 모두 5cm 이상일 때만 허용한다.
  4cm 넘게 떨어진 평행 구간이 전체의 20%를 넘으면 잘못된 코너로
  거부한다. 역순/접힘/끝점 단독 대응 거부는 유지한다.
- 도착 허용값은 고정 진입축 기준 전방/좌우 각각 2.5cm이다.
  제어 주기에서 신선한 odom과 0.8초 이내 실제 대응 관측으로
  도착을 기록하며, 저장된 코너 목표를 다시 갱신하지 않는다.
- 실제 검출 개수가 0이어도 도착이 기록된 코너는 BRAKE/PIVOT_TURN을
  유지할 수 있다. 마지막 실제 대응 후 최대 2초, 전진 0m/s,
  회전 최대 0.15rad/s로 제한한다. 무검출 프레임은 제한 시간을
  연장하지 않는다. 시간이 끝나면 TURN_WAIT로 정지한다.
- 도착 전 무검출, 다른 좌우 정체성, 기하 거부, 오래된 odom/카메라,
  추론 실패는 이 예외를 사용하지 않는다. 회전 중 원래 위치 이탈
  제한과 전체 회전 제한도 유지한다. 이 기능은 지도 기반 맹목 주행이 아니다.
- 위 수치는 코드의 허용 조건이지 실제 거리 정확도의 보증이 아니다.
  보정/카메라 장착 오차와 odom 오차는 실측 비교가 필요하다.

### 차선 내부 회전 목표 후보 선택

기존 중앙 경로 우선순위는 유지한다. staged 코너를 처음 확정할 때만
중앙 오프셋 교점의 전후/좌우 각각 2cm 이내 후보를 비교한다. 차체
꼭짓점·변 중점·중심을 31개 회전 각도로 샘플링해 최소 차선 여유를
평가한다. 외곽 L자 차선은 두 진입/출구 통로의 합집합으로 취급하며,
반대편 차선이 관측되면 그 경계도 검사한다. 한쪽만 보이면 반대편은
추정 차선 폭에 의존하므로 물리적인 충돌 안전 보증은 아니다.

기존 목표 여유가 1cm 이상이면 유지한다. 새 후보는 설정 여유를
뺀 회전 여유가 4mm 이상일 때만 채택한다. 새 목표가 없으면 기존
지원 구간 조건을 적용하며, 임의로 구간 밖 진입을 허용하지 않는다.
짧은 코너 분석 창보다 앞에 같은 진입 차선이 실제 관측된 경우에는
직선에서 5mm 이내인 관측점을 이용해 지원 구간을 확인한다. 전체
차선 끝점 밖으로는 연장하지 않는다. 후보 비교는 회전 공간 평가이며
진입 궤적 전체의 충돌 검증이나 보정 오차 검증을 대신하지 않는다.

선택 후 pivot은 odom 좌표에 고정된다. 로그에는 기존 travel, 전후 이동,
좌우 이동, 후보 최소 회전 여유, 실제 관측 진입 시작점을 남긴다.

### 확정 코너의 근거리 카메라 유실: 제한된 odom 접근

실제 차선 검출 개수가 0일 때만 적용한다. 이미 staged 코너가 확정되고,
마지막 실제 차선 대응이 0.8초 이내이며, 남은 전방 거리가 도착 허용값
+ 3cm 이내(기본 5.5cm), 횡오차가 도착 허용값 이내, 진입 방향 오차가
15도 이내여야 시작한다. 다른 차선/기하 오류는 이 예외를 사용하지 않는다.

목표는 고정된 odom 좌표를 유지한다. 속도 최대 0.015m/s, 마지막 실제
대응 후 최대 2초, 마지막 대응 pose부터 누적 이동 최대 3cm이다.
제어 주기마다 누적 이동을 계산하므로 왕복해도 예산을 돌려받지 않는다.
카메라/추론/odom 신선도 검사는 기존대로 유지한다. 시간·거리·횡오차·
방향 제한을 넘으면 정지한다. 빈 프레임은 예산을 연장하지 않는다.

도착하면 전진을 멈추고 BRAKE/PIVOT_TURN으로 전환한다. 접근의 2초
제한은 회전에 넘기지 않는다. 회전 마감은 도착 확인 시각 + 제동 시간
+ `corner_spin_timeout_s`(기본 20초)로 한 번 고정한다. 무검출 프레임이나
차선 재획득으로 마감을 갱신하지 않는다. 무검출 회전 속도는 최대
0.15rad/s이고 전진은 0이다. odom 목표각 오차가 12도 이내면 즉시
회전을 멈추며, 실제 근거리 출구 경로가 확인돼야 정상 주행으로 돌아간다.
차체 중심이 pivot에서 허용 거리 이상 이탈하거나 카메라/추론/odom이
오래되면 기존 안전 정지를 유지한다. 모순되는 차선 관측도 무시하지 않는다.
이것은 실제 카메라 사각의 짧은 틈을 메우는 기능이지, 유실 상황에서
무조건 전진하거나 odom 오차를 보정하는 기능은 아니다.

### PC 기하 계산 / 핑키 경량 제어 분리

`lane_autonomy_pinky2.launch.py`는 `remote_geometry:=true`를 기본으로
사용한다. PC 워커가 YOLO, 픽셀 투영, 차선 정체성/관측 히스토리,
중앙 경로 및 코너 후보 계산을 기존 함수로 수행한다. PC에는 ROS 노드,
속도 발행자, enable 서비스가 생성되지 않는다. 핑키는 카메라 시각에
보간한 odom·보정·설정·현재 확정 코너 상태를 기존 SSH 요청에 포함한다.

응답에는 목표 좌표/코너 목표각과 진단 정보가 담긴다. 단일 사용 프레임
토큰, 활성화 세대, 촬영 시각, 유한 좌표를 검사한다. 유효기간은 핑키가
보관한 실제 촬영 시각 기준이며 PC 시계로 연장하지 않는다. odom 점프나
enable/disable은 이전 응답을 무효화한다. PC 대기 중 핑키가 확인한
도착, 누적 무검출 이동, fault를 오래된 응답으로 되돌리지 않는다.

일반 목표도 odom 좌표로 저장하고 제어 주기마다 최신 base 좌표로
변환해 Pure Pursuit를 계산한다. 코너는 기존 고정 pivot/목표각 제어를
사용한다. 로봇에서 속도 제한, 신선도/통신 끊김 정지, stop-line 판정,
접근 2초/3cm 및 회전 마감 제한을 유지한다. `/cmd_vel`은 핑키만 발행한다.
디버그 이미지와 지도 표시용 경량 관측 이력은 기존 핑키 토픽을 유지한다.

두 프로세스를 모두 새 코드로 재시작해야 한다. 모델/보정/속도/타임아웃은
임의로 바꾸지 않았다. 기하 조건으로 목표 생성이 실패하는 장면은 PC로
옮겨도 그대로 실패할 수 있으며, 이 변경은 그 조건을 무시하지 않는다.

관제 PC:

```bash
source /opt/ros/jazzy/setup.bash
source /home/tory/ws/pinky_pro/install/setup.bash
ros2 launch pinky_move lane_inference_pc.launch.py
```

핑키2 (기존 동일 런치를 중복 실행하지 말 것):

```bash
source /opt/ros/jazzy/setup.bash
source /home/pinky/pinky_pro/install/setup.bash
ROS_DOMAIN_ID=22 ros2 launch pinky_move lane_autonomy_pinky2.launch.py
```

처음에는 disabled 상태로 카메라/목표를 확인한다. 활성화는 사용자가
로봇 옆에서 별도로 수행한다. 이전 분리 방식으로 돌아가려면 핑키 런치에
`remote_geometry:=false`를 추가한다(PC는 segmentation만 응답).
PC 계산을 쓰는데 구버전 워커가 plan 없이 응답하면 정지하며 핑키에서
무거운 계산으로 자동 fallback하지 않는다.

검증 기록: 로컬 전체 기능 테스트 341개 통과, 핑키2 선택 테스트 109개
통과, 양쪽 colcon 빌드 완료 및 배포 코드 SHA-256 일치 확인.
저장 사진 `/tmp/pinky_lane_raw_20261004.jpg` 한 장을 실제 기존 모델로
5회 반복 처리한 PC 오프라인 결과: 첫 응답 0.277초(기하 0.258초),
이후 총 0.073~0.080초(기하 0.054~0.060초), 5회 모두 목표 생성.
이는 영점 odom을 가정한 정지 사진 반복 검증이다. 네트워크 왕복,
실제 연속 프레임, 모든 코너 사례 또는 실주행 성공률의 측정이 아니다.
배포 백업: `/home/pinky/deploy_backups/pc_geometry_20261004_OlDApS`.
배포 과정에서 로봇 주행 노드/모터는 활성화하지 않았다.

### 회전 완료와 출구 차선 재획득 분리

2026-10-04 정지 후 위치 차이 복구 보완: 마지막 정상 pose 대비 1cm/5도
제한은 정지하는 동안의 이동까지 영구 차단했다. 재획득 후보 범위만
5cm/20도로 변경하고, 실제 재출발은 현재 관측 8cm 이상이 저장 경로와
오차 1cm 이내·접선 cosine 0.95 이상으로 일치해야 한다. 3회 확인 동안
첫 확인 pose 대비 변위 5mm/각도 3도 이내로 안정돼 있어야 하며 확인
중 목표는 반환하지 않는다. 범위 초과나 기하 불일치는 계속 정지한다.

가림 판정은 저장된 실제 차선 경계를 현재 odom으로 변환해 동일 보정의
robot_floor_pixel로 역투영한다. 중앙 오프셋 경로 x가 14cm를 넘더라도
대응 실제 차선이 영상 밖/하단 10%라면 근거리 가림 후보로 재검증한다.
관측 누락 길이 6cm, 숨겨진 굽힘 20도 제한은 유지한다. 보정이 없거나
잘못됐으면 이 예외를 허용하지 않는다. 보정 모델 기반 판단이지 물리적
시야/안전을 검증한 결과는 아니다. 저장 경계와 중앙의 거리도 폭/2에서
4cm 이상 어긋나면 예외를 사용하지 않는다.

실패 원인은 crop too long / boundary not verified near/out of view /
hidden bend too sharp로 분리한다. s_route_crop_m, crop_max_x_m,
crop_heading_deg, optical_crop, match_age_s, pose_delta_m,
yaw_delta_deg를 로그에 남겨 다음 실패를 수치로 구분한다.

2026-10-04 근거리 부분 가림/정지 재획득: 저장 경로 첫 점부터 전부
현재 영상에 나타나야 한다는 조건을 완화한다. 현재 중앙 경로 시작점을
저장 경로에 대응시키되 누락 구간은 호 길이 최대 6cm, base 전방 14cm
이내, 구간 방향 변화 20도 이내로 제한한다. 이는 기하 기반 근거리
가림 후보 제한이며 광학적으로 가시성이 증명됐다는 뜻이 아니다.
보이는 구간은 최소 5cm 이상 기존 오차/순서/방향 검사를 통과해야 한다.
목표나 진행 위치를 보이는 시작점으로 점프시키지 않으며 저장된 가까운
경로를 유지한다. 숨겨진 코너/먼 S자 가지/불일치 경로는 거부한다.

마지막 성공 대응으로부터 2초가 지났어도 즉시 영구 reset 상태로 만들지
않는다. 마지막 확인 pose에서 변위 1cm 이내·각도 5도 이내일 때만 새
유효 경로를 3회 연속 재검증한다. 확인 중에는 목표를 반환하지 않아
정지하며, 관측 간 max_gap_s 초과·불일치·위치 변화는 횟수를 초기화한다.
3회 성공해야 시간 기준을 갱신한다. 큰 이동/odom 변화는 기존 reset
요구를 유지한다. s_route_near_crop_m, visible_overlap_m,
reacquire_count로 판단 과정을 확인한다. 차선 0개를 무제한 재생하지 않는다.

2026-10-04 관측 진입 구간 끝점 허용오차: 확정 전 pivot의 마지막 구간
검사에만 최대 0.010m(10mm, 경계 포함)를 허용한다(사용자 요청으로 5mm에서 상향). 시작/끝 구간 밖의
초과량을 entry_support_overrun_m으로 기록한다. 후보 탐색 범위·차체
검사·자세·odom·맹목 이동 예산·회전 시간 제한은 변경하지 않는다.
10mm는 관측/피팅 경계의 수치 허용오차이며 물리적 안전 보장은 아니다.

2026-10-04 한쪽 차선 S자 지속: 이미 진행 중인 S자 일부가 CORNER로
분류됐다는 이유만으로 즉시 정지시키지 않는다. 현재의 유효한 단일 차선
중앙 경로를 생성한 후 기존 odom 경로와 대응 검사를 통과하면 S자로 계속
따른다. 매번 실제 관측이 이어지면 처음의 S자 분류로부터 2초가 지났다는
이유로 중단하지 않는다. 2초 제한은 마지막 성공한 실제 경로 대응으로부터의
공백에 적용한다. 역할/폭 불일치, 경로 건너뜀, 오래된 odom/관측은 여전히
정지한다. 이미 만료된 경로는 자동으로 다시 유효해지지 않는다.

픽셀 좌우 힌트/영상 연속성/기하 역할 판정은 기존 MetricLaneTracker를
공통 사용한다. S자라고 화면 중앙 좌우로 기존 역할을 매 프레임 뒤집지 않는다.
역할이 확인된 단일 차선에서 inferred target discontinuity만 발생했다면
새 목표를 바로 채택하지 않고 저장 S자 경로로 재검증한다. 좌우 정체성이
모호한 오류나 다중 마스크 오류는 이 예외에 포함하지 않는다.

2026-10-04 굽힘 조기 조향 완화: 이미 유효한 중앙 경로에서 5cm chord
방향이 근거리 방향과 20도 이상 달라지는 진입점을 찾는다. 기본 목표가
그 이후에 있으면 같은 경로의 진입점으로만 당긴다. 추가 직진 거리나
가상의 바깥쪽 경로를 만들지 않는다. 진입점이 6cm 이내이거나 전방 5cm
이내면 제한을 해제해 정상 곡선 추종으로 이어간다. bend_entry_limited,
bend_entry_arc_m, bend_entry_distance_m으로 적용 여부를 확인한다.
75도 미만 곡선에서 유효한 관측 중앙 경로가 있으면 코너 반경 안에
들어왔다는 이유만으로 별도 Bezier/footprint 계획으로 교체하지 않는다.
유효 경로가 없으면 기존 거부 조건은 유지한다. 75~115도 직각 코너의
제자리 회전 진입 절차는 바꾸지 않는다.

저장 실영상 재현(20261004)은 왼쪽 경계 61도, 중앙 목표 약 x=19cm,
y=-11cm였고 실제 정지 직전 로그는 x=17.8cm,y=-12.9cm,w=-0.206였다.
코너의 최종 회전 방향과 진입 방향이 다를 수 있으므로 이 수치만으로
인코스 오차를 확정하지 않는다. 실제 영상/odom 연속 기록 없이 물리적
차선 여유나 궤적 개선을 보장할 수 없으며 테스트는 무동력 회귀 검증이다.

2026-10-04 S자 진행 경로 유지: PC 플래너의 CornerPolicy가 관측된 중앙
경로를 odom 좌표로 저장한다. 새 영상은 아직 지나지 않은 곡선을 바꾸지
않는다. 현재 로봇 위치를 경로의 국소 호 구간에 투영하되 실제 odom
변위의 1.5배 + 5mm 범위 안에서만 진행 위치를 찾는다. 목표는 전방 x가
아니라 경로 호 길이로 선택하고, 아직 경로 시작점에 도착하지 않았으면
그 지점까지의 거리도 lookahead 예산에 포함한다.

현재 관측 경로가 다음 5~12cm를 2cm 이내 오차로 순서·방향에 맞게
지지해야 한다. 먼 굽힘만 새로 보이는 경우 경로를 건너뛰지 않고 정지한다.
기존 끝 5cm 이상이 새 관측과 1cm 이내로 겹치고 방향 차가 20도 이내인
경우에만 새 꼬리를 붙인다. 이미 지난 부분만 제거하고 자기교차는 거부한다.
2초 초과 관측 간격, 큰 odom 변화, 역할/폭 불일치도 정지 사유다.
S자로 저장된 최초 출구에 5cm 이내로 접근하고 NORMAL 관측 및 경로 대응이
확인되면 일반 추종으로 복귀한다. 미완료 S 경로와 새 코너 판단이 충돌하면
갑자기 제자리 회전으로 전환하지 않는다. 상태 초기화는 기존 disable/reset을
사용한다. s_route_progress_m / remaining_m / target_arc_m을 진단에 표시한다.
실주행 성공을 보장하는 변경이 아니며 새 관측이 끊긴 경로를 무제한 재생하지 않는다.

2026-10-04 도착/인식 분리: 제어 루프는 활성 상태에서 최신 odom(기존
corner_odom_timeout 이내)을 확인한 뒤 추론 결과 소비 및 안전 정지 검사보다
먼저 도착 사실을 기록한다. metric_target이 없거나 추론이 일시 실패해도
도착 기록은 가능하지만 속도 출력은 여전히 모든 안전 검사를 통과해야 한다.
고정 pivot의 전방/횡오차가 각각 기존 허용 범위 이내이고 진입 자세 오차가
15도 이내이며 마지막 실제 차선 대응이 2초 미만일 때만 인정한다.
기존 fault, 차선 불일치로 취소된 blind_eligible, 3cm 이동 예산 소진,
20초 진입 제한은 우회하지 않는다. 부분/빈 관측은 시간 예산을 갱신하지 않는다.
로컬 brake_at은 과거 요청에 대한 PC 응답으로 덮어쓰지 않으며, 추론 복구 후
후속 계획에서 회전 단계로 진행한다. 기존 회전 마감 시간도 연장하지 않는다.
이는 추가 맹목 전진이나 이미 오래 정지한 ENTRY_WAIT의 자동 해제를 뜻하지 않는다.

2026-10-04 S자 분리 개선: 신규 제자리 회전 진입과 코너 히스토리
복구에 사용하는 각도 범위를 좌우 모두 절댓값 75~115도(경계 포함)로
제한한다. 일반 코너 검출 자체의 각도 조건과 이미 확정한 회전은 바꾸지 않는다.
S자 판정은 잡음의 양/음 각도 합이 아니라, 각각 5cm 이상의 관측 호에서
20도를 넘는 서로 반대 방향의 변화가 있는지 검사한다. S자에 최대 각도
제한을 두지는 않지만 단일 60도 코너를 각도만 보고 S자로 간주하지 않는다.

실제 S자 관측은 odom 좌표로 최대 2초 보관한다. 현재 차선의 역할·폭·
기하 대응이 맞는 경우 일부만 보여도 S자 판정을 유지한다. 부분 관측은
유효 기간을 연장하지 않으며, 관측 단절/불일치는 기억을 해제한다.
유효한 기존 중앙 목표를 우선 사용하고, 없을 때는 현재 곡선의 호 순서와
법선으로 중앙 경로를 생성한다. 양쪽 차선은 기존 대응 검사를 재사용한다.
기존 접힘/자기교차/전방 유효성 검사는 유지하고 유효 경로가 없으면 정지한다.
S자에서 기존 코너용 단일 Bezier 회전 경로나 제자리 회전을 요청하지 않는다.
이는 S자의 양방향 굽힘이 실제 관측됐거나 최근 대응 이력이 있을 때만
적용되며, 최초 화면에 한쪽 굽힘만 보이는 문제의 해결을 보장하지 않는다.

도착 전 ENTRY의 차선 대응 조건은 유지한다. 도착 확인 뒤 BRAKE /
PIVOT_TURN에서는 차선 검출 개수·좌우 재획득 오류가 회전 목표를
취소하지 않는다. 저장된 pivot/출구 yaw, 최신 odom, 고정 회전 마감,
위치 이탈 제한으로 제자리 회전을 관리한다. 전진 속도는 0이다.
카메라/추론/odom 신선도 및 통신 정지는 여전히 로봇에서 검사한다.

목표각에 도달하면 heading_reached를 유지하고 EXIT_REACQUIRE에서
속도 0으로 기다린다. 차선이 늦게 나타나도 다시 회전을 시작하지 않는다.
회전을 못 마치고 20초가 지나면 TURN_TIMEOUT이다. 회전을 이미 마쳤다면
그 이후는 회전 연장이 아니라 정지 상태의 출구 관측 대기다.

### 분리된 상단 검출은 참고용으로 구분

신선한 기존 좌우 경계와 유일하게 대응하는 차선이 lookahead 이내에서
시작할 때만 적용한다. 별도 마스크가 화면 65% 높이 위에만 있고 바닥
좌표에서도 lookahead+6cm보다 멀며 가까운 마스크와 픽셀 연결이 없으면
현재 목표 생성에서 제외한다. 디버그는 `CONTEXT [FAR; NOT STEERING]`으로
표시한다. 검출 순서가 바뀌어도 원본 마스크 인덱스를 유지한다.
가까운 경계와 같은 마스크로 이어진 먼 구간은 잘라내지 않아 기존
코너 방향 판단에 남긴다. 새로운 근거리 반대편 선은 3프레임 확인 규칙을
유지한다. 가까운 경계를 잃었거나 초기 좌우 잠금이 없으면 이 분기로
임의의 선을 선택하지 않는다. 상단 영상 전체를 검게 지우는 처리가 아니다.

### 가까운 연결 구간 추출 상세

긴 연결 곡선의 전체 피팅이 실패하거나 RMS 6mm를 초과하면 첫 구간부터
8, 12, 16, 20, 24, 28cm 길이로 접두 구간을 늘려 검증한다. 각 단계는
RMS 6mm/최대 오차 8mm 이내여야 하며 최초 실패에서 중단한다.
관측 점 사이에 3cm를 넘는 간격이 있으면 연결하지 않는다. 첫 구간조차
검증되지 않으면 이 복구는 실패한다. 반환하는 것은 실제 관측의 첫 연속
부분이며, 먼 구간을 이어 붙인 경로나 외삽한 직선이 아니다. 기존 법선
오프셋/접힘 검사는 이후에도 적용된다. 좌우 역할과 정지 한도는 바꾸지 않는다.

long_s_prefix_20261004.json 정지 장면에서 전체 관측 길이 88.1cm,
전체 피팅 RMS 11.4mm 대신 가까운 24.0cm의 RMS 2.1mm 구간이 선택됐다.
반복된 정지 영상에서 중앙 목표 생성은 확인했으나 실제 주행 검증은 별도다.

행/열 추출은 폭이 넓게 투영된 가까운 횡방향 선을 제외하면서도 먼
출구 구간만 유효 곡선으로 반환할 수 있다. 단일 마스크에서 축 추출
시작점이 max(20cm, 최소 관측거리+6cm)보다 멀고 선택 구간 길이가 8cm
이상이면 연결 중심선도 검사한다. 기존의 짧은 경계 부트스트랩은 유지한다.
연결 구간이 최소 관측거리+4cm 이내에서 시작하고 기존 후보보다 6cm
이상 가까우며, 관측 길이 8cm 이상/피팅 RMS 6mm 이내일 때만 우선한다.
분기점에서 끝난 연결 구간을 먼 구간에 이어 붙이지 않는다. 정상적인
가까운 축 추출은 교체하지 않고, 다른 경계 식별/제어 안전 조건도 유지한다.

near_turn_20261004.json은 실제 정지 장면의 검출 다각형이다. 이 장면은
축 추출 시작 전방 25.35cm → 연결 추출 13.89cm로 바뀌고, 가까운 관측
구간 27.43cm를 보존한다. 정지 영상 반복 입력에서 중앙 목표 생성은
확인했으나 실주행 복귀 성공을 의미하지 않는다.

### 짧은 저장 S 경로 재사용 정책

크롭/가림/관측 구간 부족으로 현재 중앙 경로를 연결하지 못하거나
검출이 완전히 비었을 때, 마지막 검증된 odom 경로를 현재 자세로
변환해 `S_HISTORY` 목표를 생성할 수 있다. 최대 2초, 기준 자세에서
변위 4cm, 방향 변화 20도 중 하나라도 초과하면 정지한다.
속도 상한은 0.02m/s이며 두 번째 1초 동안 0으로 감속한다.
목표 생성은 저장 경로의 시각·자세·진행률을 갱신하지 않는다.
따라서 실패 프레임이 반복돼도 사용 가능 시간이 연장되지 않는다.
기존 경로 끝 밖으로 외삽하거나 전방이 아닌 목표를 만들지 않는다.

현재 경계가 있으면 같은 좌우 역할/폭이며 저장된 실제 경계와 공통인
연속 관측 구간에서 점의 70% 이상이 2.5cm 이내, 접선 코사인 0.9 이상으로
대응해야 한다. 전체 관측을 균등 재표본화하며 저장 경계 끝의 접선 방향
밖으로 투영되는 선두/후미만 비교에서 제외한다. 내부의 거리 불일치를
새 구간으로 제거하지 않는다. 공통 구간의 최대 오차는 5cm를 넘을 수 없다.
대응 순서와 최소 2cm 범위도 검사한다. 명백한 다른 경계·폭 변화·
좌우 모호성 오류는 재사용 대상으로 삼지 않는다. 카메라·추론 실패와
오도메트리 신선도 제한은 기존 로봇 안전 감시가 그대로 적용한다.
이 기능은 기록 전체를 무관측으로 끝까지 주행시키는 기능이 아니다.

정상 중앙 경로 검증에 성공한 프레임에서 실제 경계도 갱신한다.
5cm 이상의 공통 구간과 강화된 1cm/접선 0.95 기준을 사용하며, 기존
끝점과 2cm 이내로 연결되고 자기교차가 없는 새 후미만 추가한다.
저장된 미주행 경계와 중앙 경로는 교체하지 않으며 표본 상한은 600개다.
갱신 실패 시 기존 경계는 보존한다. 히스토리 재사용 자체는 경계를
연장하거나 시간 예산을 갱신하지 않는다. 디버그에는 공통 길이, 대응률,
최대 거리 오차, 최소 접선 코사인, 역방향 이동량 및 거부 이유를 기록한다.

### S 중앙 경로의 작은 거리 오차 재획득

일반 허용 오차 2cm는 유지한다. 실패 항목이 거리뿐이고 최대 오차가
2.5cm 이내, 공통 구간 10cm 이상, 모든 대응 접선 코사인 0.98 이상인
경우에만 별도 확인을 허용한다. 확장 회전 가림 예외와 중첩하지 않는다.
마지막 검증 자세와 변위 5cm/회전 20도 이내여야 한다.

3개의 신선한 프레임이 필요하며 처음 확인 자세에서 변위 5mm/회전
3도 이내, 대응 오차 벡터 변화 4mm 이내여야 한다. 확인 중에는 목표를
반환하지 않으며 경로의 시각·자세·진행률을 갱신하지 않는다. 불일치,
누락 및 시간 간격 초과는 확인을 다시 시작한다. 허용 후에도 매 프레임
같은 강화 조건으로 검사하고 속도를 0.02m/s 이하로 제한한다.
저장된 미주행 경로나 목표를 현재 관측 경로로 덮어쓰지 않는다.
디버그 s_distance_reacquire_count와 s_match_min_cosine으로 확인 가능하다.

### 새 경계 유지 상세

단일 경계 추종 중 새 반대편 경계가 나타나면 기존 경계와 유일하게
대응되는 현재 관측을 우선한다. 새 쌍의 폭 검사를 먼저 통과해야 하며,
현재 단일 경계에서 만든 중앙 경로와 쌍의 중앙 경로가 대응되고 새 경계가
3프레임 연속 일치할 때 쌍으로 승격한다. 실패/단일 관측/시간 간격 초과는
확인을 초기화한다. 쌍 승격 후에도 코너 판정의 기준 경계를 무조건 왼쪽으로
바꾸지 않는다. 확인 중에는 오래된 목표가 아니라 현재의 동일 경계로
계산한 목표를 사용하며 기존 기하/연속성 거부 조건을 우회하지 않는다.

기존 bend-entry 목표 제한은 유지하고 접근 경로와 이후 경로를 별도
메타데이터로 보존한다. 제한 중 디버그 중앙선은 접근 구간만 표시한다.
전체 경로는 삭제하지 않아 S 곡선 검증 및 코너 출구 방향을 유지한다.

### S 경로의 회전 중 화면 이탈 예외

일반적인 가까운 경로 누락은 기존 6cm/20도 제한을 유지한다. 더 긴
누락은 저장된 실제 경계(중앙 오프셋 경로가 아님)를 현재 odom 자세로
변환하고 보정 정보로 재투영했을 때 화면 밖임이 확인되어야 한다.
화면 하단에 아직 보이는 픽셀은 이 예외의 근거가 아니다.
예외 상한은 경로 길이 25cm, 숨은 경로 방향 변화 60도이다.
현재 중앙 경로와 실제 경계 모두 위치 오차 1cm 이내, 접선 코사인
0.95 이상, 순서가 유지되는 8cm 이상 대응을 요구한다.
저장된 경로의 진행 위치를 새 검출 시작점으로 이동시키지 않는다.

부분 관측 성공으로 갱신되지 않는 에피소드 기준으로 2초, 이동 6cm,
회전 30도 중 하나라도 초과하면 정지한다. 일반 관측 검증에 다시
성공해야 예외 예산을 해제한다. 현재 관측이 전혀 없을 때의 새 무관측
주행 기능은 추가하지 않았으며, 카메라/추론/odom 감시는 그대로다.
이는 명목 URDF 보정에 기반한 모델 판정이며 실제 가시성이나 안전성의
물리적 검증을 대신하지 않는다. 실주행 성공 여부는 별도 확인이 필요하다.

회전 완료 후 이전 좌우 잠금·영상 대응 이력을 한 번 초기화하고 측정
차선 폭은 유지한다. 새로운 자세에서 유효한 근거리 목표(전방 5~30cm,
목표 방향 ±20도)가 3프레임 연속 확인되고 현재 yaw도 목표각 ±12도
이내일 때만 정상 주행으로 복귀한다. 프레임 간격이 설정 max_gap_s를
넘으면 출구 확인 횟수는 초기화된다. 단일 차선이 보였다는 사실만으로
회전을 취소하거나 전진을 재개하지 않는다.

### 10초 로컬 차선 맵 보조

`CornerPolicy.local_lane_map`은 촬영 시점 odom으로 실제 관측한 좌우
경계를 변환해 최근 10초(최대 600프레임, 각 경계 80점)를 보관한다.
가상 경계와 중앙 경로는 맵에 넣지 않는다. 로봇에서 1m 밖으로 나가는
관측은 처음 이탈 지점에서 자르고, odom 점프(20cm/45도)에는 맵을 비운다.

현재 유효한 중앙 경로가 있지만 저장된 S 경로와의 형상 비교에서만
거절될 때, 같은 쪽 실제 차선의 첫 10cm를 과거 두 프레임과 비교한다.
최대 오차 1.5cm, 접선 코사인 0.95, 대응 길이 5cm, 순서 보존,
폭 차이 3cm 이내를 요구한다. 최근 대응 하나는 2초 이내여야 한다.
검증되면 **현재 관측으로 계산된** 중앙 경로의 첫 10cm만 선택하고
속도 상한 0.02m/s를 적용한다. 전체 저장 S 경로나 출구는 덮어쓰지 않는다.
상태는 `LOCAL_MAP`, 목표에는 `local_map_assisted=True`가 표시된다.

10초는 보관 기간이지 무관측 주행 허용 시간이 아니다. 차선 미검출,
현재 중앙 경로 생성 실패, 좌우 불일치, 단계별 직각 회전, odom 누락에는
이 대체 동작을 사용하지 않는다. 기존의 제한된 히스토리 주행은 별도로
유지하며, 과거의 먼 차선을 현재 경로에 임의로 이어 붙이지 않는다.
`/lane_autonomy/local_map` 표시 이력도 10초로 늘렸다. 표시는 제어 맵과
별도 진단 이력이며 맵에서 검증되었다는 의미는 아니다.
단위/회귀 검증과 실주행 성능 검증은 구분한다.

### 2026-10-05: S자 추종 제한 완화 프로필

실행 기본값은 `corner_relaxed_tracking: true`이다. `false`로 설정하면
기존 엄격한 S 경로 확인 조건으로 돌아간다. 제자리 직각 회전 정책은
변경하지 않으며, 몸 쪽 차선 우선 선택 기능도 다시 넣지 않는다.

- 현재 유효한 차선 경로가 있으면 만료 후 5cm/20도 제한과
  정지 상태 3프레임 재확인, 추가 1cm/8cm 대응 요구를 적용하지 않는다.
- 일반 경로 대응 오차는 2cm에서 4cm, 최소 대응 길이는 5cm에서 3cm,
  대응 순서의 작은 역행 허용은 2mm에서 10mm로 완화한다.
- 가려진 시작 구간은 6cm에서 20cm, 그 구간의 굽힘은 20도에서 90도로
  완화한다. 실제 차선이 화면 밖에 있는지 확인하는 로직은 유지한다.
- 로컬맵 대체는 기존 형상 충돌뿐 아니라 만료·크롭·재획득 실패에도
  연결한다. 현재 같은 쪽 차선과 최근 기록 1개가 오차 3.5cm,
  접선 코사인 0.80, 대응 길이 3cm로 맞으면 현재 중앙 경로를 사용한다.
  이때 실패한 저장 S 경로를 비워 다음 프레임이 새 경로를 만들게 한다.
- 로컬맵 보관 10초, 최근 확인 2초, 속도 상한 0.02m/s는 유지한다.
  차선이 없거나 현재 중앙 경로가 무효하면 이 대체 동작을 사용하지 않는다.

카메라/통신/odom 누락 정지, 비정상 수치 거부, 속도 제한과 비활성화
서비스는 그대로 유지한다. 이 변경은 실주행 완주 검증을 대신하지 않는다.

### UNKNOWN 이어받기·디버그 격리·전후 1m 관측 보관

`corner_relaxed_tracking=true`에서는 코너 분류가 UNKNOWN이어도 현재 실제
차선에서 계산된 유효한 중앙 경로가 있으면 Pure Pursuit를 계속한다.
분류 실패 원인은 `classification_reason`, 이어받기 여부는
`unknown_following`으로 기록한다. 진행 중인 제자리 회전은 이 분기로
취소하지 않으며, 차선 미검출/무효 경로를 새 목표로 만들지는 않는다.

디버그 이미지 DDS 발행은 `lane_debug_images` 별도 프로세스가 담당한다.
기존 스레드만으로는 Python GIL과 DDS participant를 제어와 공유했기 때문에,
이미지 변환·발행 지연이 제어 콜백에 영향을 줄 수 있었다. IPC 대기열은
2개로 제한하고 혼잡 시 진단 프레임만 버린다. 구독자가 없으면 렌더링과
맵 복사를 생략한다. 디버그 기본 발행은 3Hz, 맵은 최대 2Hz이며 실제
구독 주소 `/lane_autonomy/debug_image`, `/lane_autonomy/local_map`은 유지한다.
타이밍 로그의 `debug_publish`와 `debug_dropped`로 발행 정체를 구분한다.
이 격리는 주기적 지연의 후보 경로를 차단한 것이며, 실제 무선 환경에서
주기적 지연이 사라졌는지는 실행 로그로 별도 검증해야 한다.

맵 표시는 로봇을 중앙에 놓고 전방 +1m/후방 -1m, 좌우 +/-1m를 보여준다.
`SpatialLaneArchive`는 실제 관측을 odom 기준 1cm 셀에 중복 제거하여 보관한다.
10초가 지나도 공간 범위 안이면 유지하며 범위를 벗어난 점과 odom 리셋 시
무효 좌표는 제거한다. 10초 최근 대응 이력은 그대로 따로 유지한다.
누적 관측은 어두운 점, 최근 차선은 밝은 선으로 표시하고, 다른 조각들을
임의로 이어 붙이지 않는다. 이 보관은 실행 중 메모리 보관이며 재시작 후
자동 복원하는 파일 지도나 SLAM이 아니다. 오래된 점만으로 주행하지 않는다.
