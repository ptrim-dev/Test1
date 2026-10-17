"""Face detection, identity embedding, swapping and restoration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnx
import onnxruntime
from insightface.app import FaceAnalysis
from onnx import numpy_helper

from .models import model_path, providers

# Five-point landmark templates (eyes, nose tip, mouth corners), normalised to a unit square.
ARCFACE_TEMPLATE = np.array(
    [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
     [0.37097589, 0.82469196], [0.63151696, 0.82325089]], dtype=np.float32)
FFHQ_TEMPLATE = np.array(
    [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32)


@dataclass
class Face:
    bbox: np.ndarray       # x1, y1, x2, y2
    kps: np.ndarray        # 5x2 landmarks
    embedding: np.ndarray  # L2-normalised 512-d identity vector
    score: float


def align(img: np.ndarray, kps: np.ndarray, template: np.ndarray, size: int):
    """Warp a face to `template` at `size`x`size`. Returns the crop and its affine matrix."""
    matrix = cv2.estimateAffinePartial2D(kps, template * size, method=cv2.LMEDS)[0]
    crop = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def feather_mask(size: int, blur: float = 0.3, padding: float = 0.0) -> np.ndarray:
    """A soft-edged box mask in crop space, used to blend a crop back into the frame."""
    mask = np.ones((size, size), np.float32)
    edge = max(int(size * blur * 0.5), 1)
    pad = int(size * padding)
    mask[: edge + pad] = 0
    mask[-(edge + pad):] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0
    return cv2.GaussianBlur(mask, (0, 0), edge * 0.5)


def paste_back(frame: np.ndarray, crop: np.ndarray, matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inverse-warp `crop` into `frame`, blending through `mask`, touching only the face's bounding box."""
    h, w = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    size = crop.shape[0]
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float32) @ inverse.T
    x1, y1 = np.clip(np.floor(corners.min(0)).astype(int), 0, [w, h])
    x2, y2 = np.clip(np.ceil(corners.max(0)).astype(int), 0, [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame
    inverse[:, 2] -= (x1, y1)
    region = (x2 - x1, y2 - y1)
    warped = cv2.warpAffine(crop, inverse, region, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(mask, inverse, region)[..., None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * warped + (1 - alpha) * roi).clip(0, 255).astype(np.uint8)
    return frame


class Swapper:
    """inswapper_128: replaces the identity of an aligned 128px face with a source embedding."""

    def __init__(self):
        path = str(model_path("inswapper_128_fp16.onnx"))
        graph = onnx.load(path).graph
        # The last initializer maps ArcFace embeddings into the swapper's latent space.
        self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        self.session = onnxruntime.InferenceSession(path, providers=providers())
        self.mask = feather_mask(128, blur=0.3, padding=0.05)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        latent = embedding.reshape(1, -1) @ self.emap
        return (latent / np.linalg.norm(latent)).astype(np.float32)

    def swap(self, frame: np.ndarray, target: Face, latent: np.ndarray) -> np.ndarray:
        crop, matrix = align(frame, target.kps, ARCFACE_TEMPLATE, 128)
        blob = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {"target": blob, "source": latent})[0][0]
        swapped = (out.transpose(1, 2, 0)[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
        return paste_back(frame, swapped, matrix, self.mask)


class Restorer:
    """GFPGAN 1.4: sharpens the low-resolution swapped face back up to 512px detail."""

    def __init__(self):
        self.session = onnxruntime.InferenceSession(str(model_path("gfpgan_1.4.onnx")), providers=providers())
        self.mask = feather_mask(512, blur=0.3)

    def restore(self, frame: np.ndarray, target: Face, strength: float) -> np.ndarray:
        crop, matrix = align(frame, target.kps, FFHQ_TEMPLATE, 512)
        blob = (crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0 - 0.5) / 0.5
        out = self.session.run(None, {"input": blob})[0][0].clip(-1, 1)
        restored = ((out.transpose(1, 2, 0)[:, :, ::-1] + 1) * 127.5).astype(np.float32)
        blended = (strength * restored + (1 - strength) * crop).clip(0, 255).astype(np.uint8)
        return paste_back(frame, blended, matrix, self.mask)


class FaceEngine:
    """Loads all models once and swaps faces in frames according to a cast mapping."""

    def __init__(self, det_size: int = 640):
        self.analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                     providers=providers())
        self.analyzer.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.swapper = Swapper()
        self._restorer: Restorer | None = None

    @property
    def restorer(self) -> Restorer:
        if self._restorer is None:
            self._restorer = Restorer()
        return self._restorer

    def detect(self, img: np.ndarray, min_score: float = 0.5) -> list[Face]:
        faces = [Face(f.bbox, f.kps, f.normed_embedding, float(f.det_score)) for f in self.analyzer.get(img)]
        return [f for f in faces if f.score >= min_score]

    def identity(self, images: list[np.ndarray]) -> np.ndarray | None:
        """Average identity of the largest face in each photo. More photos = a more stable likeness."""
        embeddings = []
        for img in images:
            faces = self.detect(img)
            if not faces:
                # Tightly cropped selfies fill the frame and slip past the detector; give them a border.
                pad = max(img.shape[:2]) // 2
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
                faces = self.detect(img)
            if faces:
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(largest.embedding)
        if not embeddings:
            return None
        mean = np.mean(embeddings, axis=0)
        return mean / np.linalg.norm(mean)

    def process(self, frame: np.ndarray, assignments: list[tuple[np.ndarray, np.ndarray]],
                match_threshold: float = 0.3, restore_strength: float = 0.8) -> np.ndarray:
        """Swap every face in `frame` that matches an assigned actor.

        `assignments` pairs an actor's centroid embedding (from the scene scan) with the
        swapper latent of the friend replacing them. Each detected face goes to the closest
        actor, provided the cosine similarity clears `match_threshold`.
        """
        if not assignments:
            return frame
        centroids = np.stack([c for c, _ in assignments])
        for face in self.detect(frame):
            sims = centroids @ face.embedding
            best = int(np.argmax(sims))
            if sims[best] < match_threshold:
                continue
            frame = self.swapper.swap(frame, face, assignments[best][1])
            if restore_strength > 0:
                frame = self.restorer.restore(frame, face, restore_strength)
        return frame
