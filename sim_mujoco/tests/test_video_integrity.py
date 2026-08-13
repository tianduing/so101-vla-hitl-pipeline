import json
import subprocess


def test_scene_smoke_video_integrity(root):
    video = root / "outputs/scene_smoke.mp4"
    result = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt,width,height", "-of", "json", video], text=True)
    stream = json.loads(result)["streams"][0]
    assert stream == {"codec_name": "h264", "width": 1280, "height": 720, "pix_fmt": "yuv420p"}
