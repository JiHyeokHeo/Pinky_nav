"""Real codec conversion and failure-safe archival tests, without ROS motion."""
import shutil
import subprocess

import pytest

from pinky_move import video_export


@pytest.fixture
def recording(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    path = tmp_path / 'raw_frames.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=blue:s=64x48:r=15:d=0.4', '-c:v', 'mpeg4',
                    str(path)], check=True, timeout=30)
    return path


def test_conversion_preserves_original_and_is_idempotent(recording):
    original = recording.read_bytes()
    result = video_export.make_browser_mp4(recording)
    assert result['changed'] and result['full_decode_ok']
    assert result['after']['codec_name'] == 'h264'
    assert result['after']['pix_fmt'] == 'yuv420p'
    archive = recording.parent / 'original_mp4v' / recording.name
    assert archive.read_bytes() == original
    converted = recording.read_bytes()
    assert not video_export.make_browser_mp4(recording)['changed']
    assert recording.read_bytes() == converted
    assert archive.read_bytes() == original


def test_encoder_failure_leaves_input_intact(recording, monkeypatch):
    original = recording.read_bytes()
    run = video_export.subprocess.run

    def fail_encode(args, **kwargs):
        if '-c:v' in args:
            raise subprocess.CalledProcessError(1, args)
        return run(args, **kwargs)

    monkeypatch.setattr(video_export.subprocess, 'run', fail_encode)
    with pytest.raises(subprocess.CalledProcessError):
        video_export.make_browser_mp4(recording)
    assert recording.read_bytes() == original
    assert not list(recording.parent.glob('*.h264.mp4'))


def test_missing_tools_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(video_export.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='ffmpeg/ffprobe required'):
        video_export.make_browser_mp4(tmp_path / 'missing.mp4')
