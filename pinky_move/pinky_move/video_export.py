"""Validated browser MP4 export; never touches robot control.

OpenCV's portable mp4v writer is fine for capture but is not widely supported
by browsers. Convert after recording has stopped; preserve the original file.
"""
import json
from pathlib import Path
import shutil
import subprocess
import uuid


def probe_video(path):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames:format=duration',
        '-of','json',str(path)],capture_output=True,text=True,check=True,timeout=30)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError('no video stream: '+str(path))
    return dict(data['streams'][0],duration=float(data['format']['duration']))


def make_browser_mp4(path):
    """Atomic replacement only after H.264 encoding and full decode succeed.

    Never overwrite an existing archival original. Codec/size/duration/frame
    count are checked before replacement; encoder failures leave input intact.
    """
    path = Path(path)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg/ffprobe required for browser video export')
    before = probe_video(path)
    if before['codec_name'] == 'h264' and before.get('pix_fmt') == 'yuv420p':
        return dict(path=str(path),changed=False,before=before,after=before)
    temporary = path.with_name(path.stem+'.'+uuid.uuid4().hex+'.h264.mp4')
    archive = path.parent/'original_mp4v'/path.name
    try:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
            '-i',str(path),'-map','0:v:0','-an','-c:v','libx264','-preset','fast',
            '-crf','20','-profile:v','baseline','-level:v','3.1','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(temporary)],check=True,timeout=180)
        after = probe_video(temporary)
        if (after['codec_name'] != 'h264' or after['pix_fmt'] != 'yuv420p' or
                any(before[key] != after[key] for key in ('width','height')) or
                abs(before['duration']-after['duration']) > .1 or
                ('nb_frames' in before and before['nb_frames'] != after.get('nb_frames'))):
            raise ValueError('browser export changed dimensions/duration/frame count')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                        '-i',str(temporary),'-f','null','-'],check=True,timeout=180)
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            shutil.copy2(path,archive)
        temporary.replace(path)
        return dict(path=str(path),changed=True,original=str(archive),before=before,after=after,
                    full_decode_ok=True)
    finally:
        if temporary.exists():
            temporary.unlink()  # Only the exact temporary file created above.
