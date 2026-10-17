"""Reading frames from a scene and writing the result back out with the original audio."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    info = VideoInfo(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def frames(path: str, start: float = 0.0, end: float | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) between `start` and `end` seconds, every `step` frames."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first = int(start * fps)
    last = int(end * fps) if end else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index = first
    try:
        while last is None or index < last:
            if step > 1 and (index - first) % step:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index, frame
            index += 1
    finally:
        cap.release()


def read_frame(path: str, t: float) -> np.ndarray | None:
    return next((f for _, f in frames(path, start=t)), None)


def render(src: str, dst: str, process: Callable[[np.ndarray], np.ndarray], start: float = 0.0,
           end: float | None = None, crf: int = 18, on_progress: Callable[[int, int], None] | None = None) -> str:
    """Run `process` over every frame of `src` in [start, end) and encode to `dst` with the source audio."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg was not found on PATH; install it to render videos.")
    info = probe(src)
    end = min(end or info.duration, info.duration)
    total = max(int((end - start) * info.fps), 1)
    trim = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}", "-i", "-",
           *trim, "-i", src,
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for done, (_, frame) in enumerate(frames(src, start, end), 1):
            proc.stdin.write(process(frame).tobytes())
            if on_progress:
                on_progress(done, total)
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the output video.")
    return dst
