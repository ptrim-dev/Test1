"""Gradio web UI: load a scene, find its actors, cast your friends, preview and render."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from . import video
from .cast import scan
from .engine import FaceEngine

MAX_ACTORS = 6
_engine: FaceEngine | None = None
_identity_cache: dict[tuple[str, ...], np.ndarray | None] = {}


def engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def friend_latent(files: list[str] | None) -> np.ndarray | None:
    """Swapper latent for the friend whose photos are `files` (cached across previews and renders)."""
    if not files:
        return None
    key = tuple(sorted(files))
    if key not in _identity_cache:
        images = [img for img in (cv2.imread(f) for f in key) if img is not None]
        identity = engine().identity(images)
        _identity_cache[key] = None if identity is None else engine().swapper.latent(identity)
    return _identity_cache[key]


def assignments(centroids: list[np.ndarray], photo_sets: tuple) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs, missing = [], []
    for i, (centroid, files) in enumerate(zip(centroids, photo_sets), 1):
        if not files:
            continue
        latent = friend_latent(files)
        if latent is None:
            missing.append(str(i))
        else:
            pairs.append((centroid, latent))
    if missing:
        gr.Warning(f"No face found in the photos for actor {', '.join(missing)}; they will be left unchanged.")
    if not pairs:
        raise gr.Error("Upload at least one photo of a friend for one of the actors.")
    return pairs


def on_video(path):
    if not path:
        return gr.update(), gr.update(), gr.update()
    info = video.probe(path)
    duration = round(info.duration, 2)
    return (gr.update(maximum=duration, value=0), gr.update(maximum=duration, value=duration),
            gr.update(maximum=duration, value=round(duration / 2, 2)))


def on_scan(path, start, end, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a movie scene first.")
    progress(0, desc="Loading models and scanning the scene...")
    actors = scan(engine(), path, start, end or None)[:MAX_ACTORS]
    if not actors:
        raise gr.Error("No faces were found in that part of the scene.")
    updates = []
    for i in range(MAX_ACTORS):
        actor = actors[i] if i < len(actors) else None
        updates.append(gr.update(visible=actor is not None))
        updates.append(gr.update(value=actor.thumbnail if actor else None,
                                 label=f"Actor {i + 1} · seen in {actor.appearances} samples" if actor else None))
        updates.append(gr.update(value=None))
    status = f"Found {len(actors)} actor(s). Upload 1–5 photos of whoever should play each role; leave a role empty to keep the original face."
    return [[a.centroid for a in actors], status, *updates]


def on_preview(path, centroids, t, threshold, restore, *photo_sets):
    if not path or not centroids:
        raise gr.Error("Scan the scene for faces first.")
    frame = video.read_frame(path, t)
    if frame is None:
        raise gr.Error("Could not read a frame at that time.")
    before = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    after = engine().process(frame.copy(), assignments(centroids, photo_sets), threshold, restore)
    return before, cv2.cvtColor(after, cv2.COLOR_BGR2RGB)


def on_render(path, centroids, start, end, threshold, restore, *photo_sets, progress=gr.Progress()):
    if not path or not centroids:
        raise gr.Error("Scan the scene for faces first.")
    pairs = assignments(centroids, photo_sets)
    out = Path(tempfile.gettempdir()) / f"faceswap_{int(time.time())}.mp4"
    began = time.time()

    def report(done, total):
        fps = done / max(time.time() - began, 1e-6)
        progress(done / total, desc=f"Frame {done}/{total} · {fps:.1f} fps")

    video.render(path, str(out), lambda f: engine().process(f, pairs, threshold, restore), start, end or None,
                 on_progress=report)
    return str(out)


def build() -> gr.Blocks:
    with gr.Blocks(title="FaceSwap Studio") as demo:
        gr.Markdown("# FaceSwap Studio\nPut yourself and your friends into movie scenes. "
                    "Only use photos of people who have agreed to it.")
        centroids = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                scene = gr.Video(label="1 · Movie scene", sources=["upload"])
                with gr.Row():
                    start = gr.Slider(0, 1, value=0, step=0.1, label="Start (s)")
                    end = gr.Slider(0, 1, value=0, step=0.1, label="End (s)")
                scan_btn = gr.Button("2 · Find the actors", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Accordion("Settings", open=False):
                    threshold = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Match threshold",
                                          info="How closely a face must resemble a scanned actor to be swapped. "
                                               "Lower catches more angles, higher avoids swapping the wrong person.")
                    restore = gr.Slider(0, 1, value=0.8, step=0.05, label="Face restoration",
                                        info="Blends in GFPGAN detail. 0 = raw swap (faster), 1 = fully restored.")

        gr.Markdown("### 3 · Cast your friends")
        actor_inputs, actor_outputs = [], []
        with gr.Row():
            for i in range(MAX_ACTORS):
                with gr.Column(visible=False, min_width=180) as col:
                    thumb = gr.Image(interactive=False, height=180, show_label=True)
                    photos = gr.File(label="Photos of their replacement", file_count="multiple",
                                     file_types=["image"], type="filepath")
                actor_outputs += [col, thumb, photos]
                actor_inputs.append(photos)

        gr.Markdown("### 4 · Preview and render")
        with gr.Row():
            preview_t = gr.Slider(0, 1, value=0, step=0.1, label="Preview at (s)")
            preview_btn = gr.Button("Preview frame")
            render_btn = gr.Button("Render video", variant="primary")
        with gr.Row():
            before = gr.Image(label="Original", interactive=False)
            after = gr.Image(label="Swapped", interactive=False)
        result = gr.Video(label="Result")

        scene.change(on_video, scene, [start, end, preview_t])
        scan_btn.click(on_scan, [scene, start, end], [centroids, status, *actor_outputs])
        preview_btn.click(on_preview, [scene, centroids, preview_t, threshold, restore, *actor_inputs], [before, after])
        render_btn.click(on_render, [scene, centroids, start, end, threshold, restore, *actor_inputs], result)
    return demo


def main():
    parser = argparse.ArgumentParser(description="FaceSwap Studio web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    args = parser.parse_args()
    build().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
