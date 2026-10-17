# FaceSwap Studio

A local web app for putting yourself and your friends into movie scenes. Upload a clip, let it
find the actors, give it a few photos of who should play each role, preview a frame, and render.

Everything runs on your own machine: no uploads, credits, or watermarks.

## How it works

```
scene.mp4 ─► sample frames ─► detect faces (SCRFD) ─► group by identity (ArcFace) ─► "Actor 1..N"
                                                                                        │
friend photos ─► ArcFace identity (averaged over all photos) ─────────────────────────┐ │
                                                                                      ▼ ▼
every frame ─► detect faces ─► match each one to the nearest actor ─► inswapper_128 swap ─► GFPGAN restore ─► ffmpeg (+ original audio)
```

| Stage | Model | File |
|---|---|---|
| Face detection and landmarks | InsightFace `buffalo_l` / SCRFD | auto-downloaded by `insightface` |
| Identity embeddings | InsightFace `buffalo_l` / ArcFace w600k_r50 | auto-downloaded by `insightface` |
| Face swap | `inswapper_128_fp16.onnx` | `faceswap_studio/engine.py` › `Swapper` |
| Face restoration | `gfpgan_1.4.onnx` | `faceswap_studio/engine.py` › `Restorer` |

Models download on first run (about 900 MB in total) to `~/.faceswap_studio/models` (override with
`FACESWAP_MODELS`) and `~/.insightface/models`.

## Setup (Windows/Linux with an NVIDIA GPU)

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`, and an
NVIDIA driver with CUDA 12 + cuDNN 9 (what current `onnxruntime-gpu` wheels expect).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[gpu]"      # or ".[cpu]" on a machine without an NVIDIA GPU
faceswap-studio              # then open http://127.0.0.1:7860
```

If `insightface` fails to build on Windows, install the "Desktop development with C++" workload
from the Visual Studio Build Tools and retry.

To confirm the GPU is in use, run
`python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`. The output should
include `CUDAExecutionProvider`.

## Using it

1. **Movie scene**: upload a clip and trim it with the Start/End sliders (short clips render fastest).
2. **Find the actors**: samples the clip and shows one card per distinct face.
3. **Cast your friends**: under each actor, upload 1–5 photos of the person who should replace
   them. Clear, front-facing, well-lit photos work best, and several photos give a steadier likeness.
   Leave a card empty to keep that actor's face.
4. **Preview frame**, then **Render video**. The output keeps the original audio.

Settings:
- **Match threshold**: lower it if the swap flickers off on side profiles. Raise it if the wrong
  person is being swapped.
- **Face restoration**: GFPGAN sharpening. Turn it down if faces look plasticky, or set it to 0
  for faster drafts.

## Project layout

```
faceswap_studio/
  app.py     Gradio UI and event handlers
  engine.py  detection, identity, swapper, restorer, blending
  cast.py    scanning a scene and clustering faces into actors
  video.py   frame reading and ffmpeg encoding with audio
  models.py  model download and GPU provider selection
tests/       model-free tests for blending and video I/O (pytest)
```

## Ideas for next steps

- Temporal smoothing of landmarks to reduce jitter between frames
- Face-parsing occlusion masks so hands and hair in front of the face stay intact
- A higher-resolution swapper or CodeFormer as a restoration option
- Lip and expression transfer (e.g. LivePortrait) for "talking" shots

## Responsible use and licensing

Only swap in faces of people who have agreed to it, and don't present results as real footage.
The `inswapper_128` and InsightFace models are released for **non-commercial research use
only**. Movie clips remain the property of their copyright holders, so keep this to personal fun.
