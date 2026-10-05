"""Convert only named Gazebo trial recordings; keep originals and audit JSON."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pinky_move.video_export import make_browser_mp4


def main():
    directory = ROOT/'reports/gazebo_physics_20261005'
    results = []
    for name in ('raw_frames.mp4','debug_frames.mp4'):
        for path in sorted(directory.glob('*/'+name)):
            result = make_browser_mp4(path)
            results.append(result)
            print(path.parent.name,name,'H.264 OK')
    (directory/'video_validation.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
