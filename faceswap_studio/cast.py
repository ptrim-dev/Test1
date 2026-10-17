"""Finding the distinct people ("actors") in a scene so each can be assigned a friend."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .engine import Face, FaceEngine
from .video import frames, probe


@dataclass
class Actor:
    centroid: np.ndarray
    appearances: int = 0
    thumbnail: np.ndarray | None = None  # RGB crop of their clearest detection
    _best_score: float = field(default=0.0, repr=False)
    _sum: np.ndarray | None = field(default=None, repr=False)

    def add(self, face: Face, frame: np.ndarray):
        self._sum = face.embedding.copy() if self._sum is None else self._sum + face.embedding
        self.centroid = self._sum / np.linalg.norm(self._sum)
        self.appearances += 1
        size = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        quality = face.score * np.sqrt(size)
        if quality > self._best_score:
            self._best_score = quality
            self.thumbnail = crop_face(frame, face)


def crop_face(frame: np.ndarray, face: Face, margin: float = 0.35) -> np.ndarray:
    x1, y1, x2, y2 = face.bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = frame.shape[:2]
    x1, y1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
    x2, y2 = int(min(x2 + mx, w)), int(min(y2 + my, h))
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def scan(engine: FaceEngine, path: str, start: float = 0.0, end: float | None = None, samples: int = 60,
         same_person: float = 0.45, min_appearances: int = 2) -> list[Actor]:
    """Sample ~`samples` frames and group the faces found by identity, most frequent first.

    Two faces whose embeddings have cosine similarity >= `same_person` are treated as the
    same actor. Faces seen fewer than `min_appearances` times (extras, false detections) are dropped.
    """
    info = probe(path)
    span = (min(end or info.duration, info.duration) - start) * info.fps
    step = max(int(span / samples), 1)
    actors: list[Actor] = []
    for _, frame in frames(path, start, end, step=step):
        for face in engine.detect(frame, min_score=0.6):
            sims = [a.centroid @ face.embedding for a in actors]
            if sims and max(sims) >= same_person:
                actors[int(np.argmax(sims))].add(face, frame)
            else:
                actor = Actor(centroid=face.embedding)
                actor.add(face, frame)
                actors.append(actor)
    actors = [a for a in actors if a.appearances >= min_appearances] or actors
    return sorted(actors, key=lambda a: a.appearances, reverse=True)
