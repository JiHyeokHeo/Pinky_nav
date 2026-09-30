# Cube calibration status

The default Pinky2 lane launch selects cube_calibration.json. The old
calibration.json is preserved but is not selected by that launch.
Current status: missing_camera_intrinsics. Autonomy enable is rejected in
cube mode; there is NO fallback to pixel steering or the old homography.
Other explicitly configured launches are not changed.

Measurements: 3 cm cubes, 2.4 cm black tag edges, 5 cm clear inner gap,
27 cm ground-projection-to-front-plane distance. Tag centres are assumed
centred on each cube. Both tags have the same ID and are sorted spatially.

Before a usable floor calibration can be produced:

1. Calibrate camera intrinsics using multiple board views at different tilts
   and positions, matching production resolution, crop, focus and 180-degree
   image rotation. A capture-settings YAML is not intrinsic calibration.
2. Provide JSON containing image_size [640,480], rotation_degrees 180,
   camera_matrix (3x3), distortion_coefficients (OpenCV), validated true.
   Do not label guessed values validated.
3. Run from the package source directory:

   python3 -m pinky_move.cube_calibration --image /home/tory/Downloads/pinky_two_cubes_20260928.jpg --intrinsics /path/to/intrinsics.json --output /tmp/cube_candidate.json

The output is a diagnostic candidate, NOT a driving calibration. Output files
are created exclusively to avoid overwriting existing measurements. Verify
floor distances at independent locations, cube alignment and camera mounting.
Reprojection error alone does not prove accurate floor distances.

The current implementation fits reference-to-optical pose and intersects
undistorted pixel rays with the floor. It does not yet implement validated
reference-to-base_footprint conversion or metric lane steering. Mechanical
front_camera_link must not be treated as an optical frame without verifying
axis conventions. Cube-mode driving remains blocked even for a fitted
candidate until these remaining pieces are implemented and validated.

No robot movement or remote deployment was performed for this implementation.
