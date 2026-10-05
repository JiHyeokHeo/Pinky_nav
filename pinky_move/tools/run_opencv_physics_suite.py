"""순차 실행: 실제 로봇 없이 격리된 Gazebo OpenCV 주행을 반복 검증한다."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=100.)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 1 <= args.seconds <= 300:
        parser.error('repeats: 1..5, seconds: 1..300')
    if args.output.exists():
        parser.error('출력 폴더가 이미 존재함: 기록을 덮어쓰지 않습니다')
    args.output.mkdir(parents=True)
    results = []
    # Three stripes are exercised from BOTH lanes, not just one favourable view.
    for course, lanes, lane in [('left90', 1, 0), ('right90', 1, 0),
                               ('s_sharp', 2, 0), ('s_sharp', 2, 1)]:
        for repeat in range(1, args.repeats+1):
            name = f'{course}_opencv_lane{lane}_r{repeat}'
            output = args.output/name
            directory = tempfile.mkdtemp(prefix='pinky-opencv-suite-')
            print(f'START {name}', flush=True)
            with (args.output/(name+'_launch.log')).open('w') as log:
                launch = subprocess.Popen(['ros2', 'launch', 'pinky_move', 'lane_gazebo.launch.py',
                    f'course:={course}', f'lane_count:={lanes}', f'target_lane:={lane}',
                    'domain:=172', 'perception:=opencv', 'gui:=false', f'output:={directory}'],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    trial = subprocess.run([sys.executable, str(Path(__file__).with_name('gazebo_lane_trial.py')),
                        '--domain', '172', '--seconds', str(args.seconds), '--course-dir', directory,
                        '--output', str(output)], timeout=args.seconds+90, capture_output=True, text=True)
                    (args.output/(name+'_recorder.log')).write_text(trial.stdout+trial.stderr)
                    record = output/'trial.json'
                    result = (json.loads(record.read_text())['summary'] if record.exists()
                              else dict(passed=False, error=trial.stderr[-2000:]))
                    results.append(dict(run=name, **result))
                    print(f'END {name}: '+json.dumps({k: result.get(k) for k in
                        ('passed', 'travelled_m', 'max_cross_track_m', 'error')}, ensure_ascii=False), flush=True)
                finally:
                    # Only the process group created above belongs to us.
                    # Ctrl+C lets ROS destroy nodes; bounded escalation avoids
                    # orphaned Gazebo children contaminating the next run.
                    if launch.poll() is None:
                        os.killpg(launch.pid, signal.SIGINT)
                    try:
                        launch.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        os.killpg(launch.pid, signal.SIGTERM)
                        try:
                            launch.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(launch.pid, signal.SIGKILL)
                            launch.wait()
                (args.output/'suite.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
                time.sleep(2)
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
