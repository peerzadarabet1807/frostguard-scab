"""ONNX Runtime inference wrapper for the YOLO11n apple scab detector.

The detector loads ``models/scab_detector.onnx`` (exported from the Google Colab notebook).
If that file is absent it transparently falls back to the colour-heuristic graph in
:mod:`src.vision.fallback_model`, generating it on the fly when needed, so the API and
dashboard never crash on a fresh clone.

Pipeline::

    image -> letterbox 640x640 (RGB, /255, NCHW) -> ONNX Runtime
          -> decode [cx, cy, w, h, scores] -> confidence filter -> class-aware NMS
          -> undo letterbox -> Detection(x1, y1, x2, y2, confidence, label)

Both output layouts produced by Ultralytics are supported:

* raw head   ``[1, 4 + nc, N]`` (YOLOv8/YOLO11 default export, NMS done here);
* end-to-end ``[1, N, 6]`` as ``[x1, y1, x2, y2, score, class]`` (``nms=True`` exports).
"""

from __future__ import annotations

import ast
import io
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image, ImageOps

from src.vision.fallback_model import FALLBACK_CLASS_NAMES, FALLBACK_MODEL_PATH, build_fallback_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "scab_detector.onnx"
LETTERBOX_COLOR = (114, 114, 114)

ImageInput = Image.Image | npt.NDArray[np.uint8] | bytes | str | Path
Backend = Literal["trained", "fallback"]


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map network coordinates back onto the source image."""

    scale: float
    pad_x: int
    pad_y: int
    orig_width: int
    orig_height: int


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("x1", "y1", "x2", "y2"):
            out[key] = round(out[key], 1)
        out["confidence"] = round(out["confidence"], 4)
        return out


@dataclass
class DetectionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    inference_ms: float
    backend: Backend
    model_path: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def lesion_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "lesion_count": self.lesion_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 2),
            "backend": self.backend,
            "model_path": self.model_path,
            **self.extras,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def xywh_to_xyxy(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert ``[cx, cy, w, h]`` rows to ``[x1, y1, x2, y2]``."""
    out = np.empty_like(boxes)
    half_w, half_h = boxes[:, 2] / 2, boxes[:, 3] / 2
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def box_iou(box: npt.NDArray[np.float32], boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """IoU of one ``xyxy`` box against many."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def non_max_suppression(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_threshold: float = 0.45,
    class_ids: npt.NDArray[np.int64] | None = None,
    max_detections: int = 300,
) -> npt.NDArray[np.int64]:
    """Greedy NMS returning kept indices sorted by descending score.

    With ``class_ids`` the suppression is class-aware (boxes of different classes
    never suppress each other), implemented with the usual per-class offset trick.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    work = boxes.astype(np.float32, copy=True)
    if class_ids is not None:
        offset = float(work.max()) + 1.0
        work += (class_ids.astype(np.float32) * offset)[:, None]
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_detections:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(work[i], work[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _parse_names(raw: str | None) -> dict[int, str] | None:
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return dict(enumerate(map(str, parsed)))
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class ScabVisionDetector:
    """Apple scab lesion detector backed by ONNX Runtime (CPU by default)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        class_names: dict[int, str] | None = None,
        fallback_path: str | Path = FALLBACK_MODEL_PATH,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        requested = Path(model_path or os.getenv("FROSTGUARD_MODEL_PATH") or DEFAULT_MODEL_PATH)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        threads = intra_op_threads or int(os.getenv("FROSTGUARD_ORT_THREADS", "0") or 0)
        if threads:
            # Pinned thread count (e.g. 1 on fractional-CPU hosts): also stop idle worker threads
            # from busy-waiting, which otherwise burns a small CPU quota without doing work.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = providers or ["CPUExecutionProvider"]

        self.backend: Backend
        if requested.is_file():
            self.session = ort.InferenceSession(str(requested), options, providers=providers)
            self.backend, self.model_path = "trained", str(requested)
        else:
            logger.warning("%s not found - using fallback colour-heuristic detector", requested)
            self.session, self.model_path = self._load_fallback(Path(fallback_path), options, providers)
            self.backend = "fallback"

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.input_size = self._resolve_input_size(model_input.shape, meta.get("imgsz"))
        self.class_names = class_names or _parse_names(meta.get("names")) or dict(FALLBACK_CLASS_NAMES)

    # -- construction helpers ----------------------------------------------
    @staticmethod
    def _load_fallback(
        path: Path, options: ort.SessionOptions, providers: list[str]
    ) -> tuple[ort.InferenceSession, str]:
        if path.is_file():
            return ort.InferenceSession(str(path), options, providers=providers), str(path)
        model_bytes = build_fallback_model().SerializeToString()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(model_bytes)
            logger.info("Generated fallback ONNX model at %s", path)
            location = str(path)
        except OSError as exc:  # read-only filesystem etc. -> keep it in memory
            logger.warning("Could not persist fallback model (%s); running from memory", exc)
            location = "<in-memory fallback>"
        return ort.InferenceSession(model_bytes, options, providers=providers), location

    @staticmethod
    def _resolve_input_size(shape: list[Any], imgsz_meta: str | None) -> int:
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])
        if imgsz_meta:
            try:
                parsed = ast.literal_eval(imgsz_meta)
                return int(parsed[0] if isinstance(parsed, (list, tuple)) else parsed)
            except (ValueError, SyntaxError, TypeError, IndexError):
                pass
        return 640

    @property
    def is_fallback(self) -> bool:
        return self.backend == "fallback"

    # -- image I/O ----------------------------------------------------------
    @staticmethod
    def load_image(image: ImageInput) -> Image.Image:
        """Accept a PIL image, HxWxC uint8 array, encoded bytes or a file path; return RGB."""
        if isinstance(image, Image.Image):
            pil = image
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        elif isinstance(image, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            pil = Image.open(image)
        else:
            raise TypeError(f"unsupported image type: {type(image).__name__}")
        pil = ImageOps.exif_transpose(pil)
        return pil.convert("RGB")

    # -- pipeline -----------------------------------------------------------
    def preprocess(self, image: ImageInput) -> tuple[npt.NDArray[np.float32], LetterboxInfo]:
        """Letterbox to ``input_size`` (aspect preserved, grey padding) and build an NCHW tensor."""
        pil = self.load_image(image)
        w, h = pil.size
        size = self.input_size
        scale = min(size / w, size / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas = Image.new("RGB", (size, size), LETTERBOX_COLOR)
        canvas.paste(resized, (pad_x, pad_y))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor), LetterboxInfo(scale, pad_x, pad_y, w, h)

    def infer(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run the raw network forward pass."""
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def postprocess(self, raw: npt.NDArray[np.float32], info: LetterboxInfo) -> list[Detection]:
        """Decode raw output into detections in original-image pixel coordinates."""
        pred = raw[0] if raw.ndim == 3 else raw
        nc = len(self.class_names)
        end_to_end = pred.shape[-1] == 6 and pred.shape[0] != 4 + nc
        if end_to_end:  # [N, 6] = x1, y1, x2, y2, score, class
            boxes, scores, class_ids = pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int64)
            mask = scores >= self.conf_threshold
            boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
            keep = np.argsort(-scores)[: self.max_detections]
        else:  # [4 + nc, N] = cx, cy, w, h, class scores
            pred = pred.T
            class_scores = pred[:, 4:]
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(len(pred)), class_ids]
            mask = scores >= self.conf_threshold
            boxes = xywh_to_xyxy(pred[mask, :4])
            scores, class_ids = scores[mask], class_ids[mask]
            keep = non_max_suppression(boxes, scores, self.iou_threshold, class_ids, self.max_detections)

        boxes = boxes[keep].astype(np.float64)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_height)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores[keep], class_ids[keep], strict=True):
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue  # box fell entirely inside the letterbox padding
            detections.append(
                Detection(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                    int(cls),
                    self.class_names.get(int(cls), f"class_{int(cls)}"),
                )
            )
        return detections

    def detect(self, image: ImageInput) -> DetectionResult:
        """End-to-end detection on a single image."""
        tensor, info = self.preprocess(image)
        t0 = time.perf_counter()
        raw = self.infer(tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        detections = self.postprocess(raw, info)
        return DetectionResult(
            detections=detections,
            image_width=info.orig_width,
            image_height=info.orig_height,
            inference_ms=inference_ms,
            backend=self.backend,
            model_path=self.model_path,
        )

    def warmup(self, runs: int = 1) -> None:
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(runs):
            self.infer(dummy)
