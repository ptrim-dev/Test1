import subprocess

import numpy as np
import pytest

from faceswap_studio import video
from faceswap_studio.engine import feather_mask, paste_back


def test_feather_mask_is_soft_box():
    mask = feather_mask(128)
    assert mask.shape == (128, 128)
    assert mask[64, 64] == pytest.approx(1.0, abs=1e-3)
    assert mask[0, 0] < 0.05


def test_paste_back_only_touches_face_region():
    frame = np.zeros((200, 300, 3), np.uint8)
    crop = np.full((128, 128, 3), 255, np.uint8)
    matrix = np.array([[0.5, 0, -50], [0, 0.5, -20]], np.float32)  # crop covers frame x 100..356, y 40..296
    out = paste_back(frame.copy(), crop, matrix, feather_mask(128))
    assert out[:40].max() == 0 and out[:, :100].max() == 0
    assert out[40 + 128, 100 + 128].min() > 200


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_frames_respects_range_and_step(clip):
    assert [i for i, _ in video.frames(clip, start=0.5, end=1.5, step=3)] == [5, 8, 11, 14]


def test_render_roundtrip(clip, tmp_path):
    out = tmp_path / "out.mp4"
    seen = []
    video.render(clip, str(out), lambda f: 255 - f, start=0.5, end=1.5, on_progress=lambda d, t: seen.append((d, t)))
    assert video.probe(str(out)).frames == 10
    assert seen[-1] == (10, 10)
