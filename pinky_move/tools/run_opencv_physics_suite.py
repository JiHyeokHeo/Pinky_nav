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
    parser.add_argument('--perception', choices=('opencv', 'yolo', 'hybrid'), default='opencv')
    parser.add_argument('--connected-geometry', action='store_true')
    parser.add_argument('--semantic-path', action='store_true', help='hybrid 입력에 OpenCV 성공 시험과 동일한 연결 경로/제어 프로파일 적용')
    parser.add_argument('--resume', action='store_true', help='완료된 trial.json을 보존하고 미시작 시험만 재개')
    parser.add_argument('--cases', choices=('all', 's', 'left90', 'right90', 's_right', 's_left'), default='all')
    args = parser.parse_args()
    if args.semantic_path and args.perception != 'hybrid':
        parser.error('--semantic-path requires --perception hybrid')
    if not 1 <= args.repeats <= 5 or not 1 <= args.seconds <= 300:
        parser.error('repeats: 1..5, seconds: 1..300')
    if args.output.exists() and not args.resume:
        parser.error('출력 폴더가 이미 존재함: 기록을 덮어쓰지 않습니다')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    # 재개 시 다른 제어 프로파일의 결과를 섞어 합격으로 표시하지 않는다.
    profile = dict(perception=args.perception, connected_geometry=args.connected_geometry,
                   semantic_path=args.semantic_path, seconds=args.seconds, domain=172)
    profile_path = args.output/'profile.json'
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        parser.error('재개 폴더의 perception/control profile이 다릅니다')
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    results = []
    # Three stripes are exercised from BOTH lanes, not just one favourable view.
    for course, lanes, lane in [('left90', 1, 0), ('right90', 1, 0),
                               ('s_sharp', 2, 0), ('s_sharp', 2, 1)]:
        if args.cases == 's' and course != 's_sharp':
            continue
        if args.cases in ('left90','right90') and course != args.cases:
            continue
        if args.cases == 's_right' and (course != 's_sharp' or lane != 0):
            continue
        if args.cases == 's_left' and (course != 's_sharp' or lane != 1):
            continue
        for repeat in range(1, args.repeats+1):
            name = f'{course}_{args.perception}_lane{lane}_r{repeat}'
            output = args.output/name
            if args.resume and output.exists():
                record = output/'trial.json'
                if not record.exists():
                    parser.error(f'미완료 자료를 덮어쓸 수 없음: {output}. 새 출력 폴더를 사용하세요')
                result = json.loads(record.read_text())['summary']
                if result.get('course') != course or result.get('lane_count') != lanes:
                    parser.error('재개 자료의 코스/차로가 일치하지 않습니다')
                results.append(dict(run=name, **result))
                (args.output/'suite.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(f'PRESERVED {name}: passed={result["passed"]}', flush=True)
                continue
            directory = tempfile.mkdtemp(prefix='pinky-opencv-suite-')
            print(f'START {name}', flush=True)
            with (args.output/(name+'_launch.log')).open('w') as log:
                launch = subprocess.Popen(['ros2', 'launch', 'pinky_move', 'lane_gazebo.launch.py',
                    f'course:={course}', f'lane_count:={lanes}', f'target_lane:={lane}',
                    'domain:=172', f'perception:={args.perception}', 'gui:=false', f'output:={directory}',
                    f'semantic_path:={str(args.semantic_path).lower()}',
                    f'connected_geometry:={str(args.connected_geometry).lower()}'],
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
