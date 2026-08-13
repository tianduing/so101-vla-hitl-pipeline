from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


class H264Writer:
    def __init__(self, path: str | Path, width: int = 1280, height: int = 720, fps: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"bad video frame {rgb.shape} {rgb.dtype}")
        assert self.process.stdin is not None
        self.process.stdin.write(rgb.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int] = (14, 28), scale: float = 0.55) -> np.ndarray:
    output = image.copy()
    x, y = origin
    for index, line in enumerate(lines):
        color = (255, 230, 90) if index == 0 else (235, 235, 235)
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if index else 2, cv2.LINE_AA)
        y += int(27 * max(scale / 0.55, 0.8))
    return output


def compose_four_panel(
    third: np.ndarray,
    wrist: np.ndarray,
    front: np.ndarray,
    lines: list[str],
    labels: tuple[str, str, str] = ("THIRD PERSON", "SIM WRIST", "POLICY/SIM FRONT"),
) -> np.ndarray:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:360, :640] = _fit(third, 640, 360)
    canvas[:360, 640:] = _fit(wrist, 640, 360)
    canvas[360:, :640] = _fit(front, 640, 360)
    panel = np.full((360, 640, 3), 22, dtype=np.uint8)
    panel = draw_lines(panel, lines, origin=(16, 28), scale=0.48)
    canvas[360:, 640:] = panel
    for label, xy in ((labels[0], (14, 28)), (labels[1], (654, 28)), (labels[2], (14, 388))):
        cv2.putText(canvas, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 235, 70), 2, cv2.LINE_AA)
    return canvas


def provenance_card(title: str, lines: list[str]) -> np.ndarray:
    card = np.full((720, 1280, 3), 18, dtype=np.uint8)
    cv2.putText(card, title, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (80, 220, 255), 3, cv2.LINE_AA)
    return draw_lines(card, lines, origin=(72, 190), scale=0.68)
