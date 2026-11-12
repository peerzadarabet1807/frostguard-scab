"""Input check: does this photo look like an apple leaf?

It runs before the scab detector and answers only that question; it never changes the detector's output. The
image encoder of TinyCLIP ViT-8M/16 (Microsoft, MIT licence; ONNX, float16 weights, models/gate/) embeds the photo
squashed to 224 x 224, and the photo passes when its cosine similarity with the nearest class prototype (mean
embeddings of apple-leaf photos, models/input_check.json) is at least the threshold:

    score  = max over prototypes of cosine(embedding, prototype)
    passed = score >= threshold

A safeguard, not a guarantee: the measured rates are in results/input_check/evaluation.json.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_PATH = PROJECT_ROOT / "models" / "gate" / "image_encoder.fp16.onnx"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "input_check.json"
GATE_SIZE = 224


def prepare(image: Image.Image | bytes | str | Path) -> npt.NDArray[np.float32]:
    """Decode like the detector (EXIF orientation applied, RGB) and squash to 224 x 224: HWC float32, 0..255."""
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    else:
        pil = Image.open(image)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    resized = pil.resize((GATE_SIZE, GATE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


@dataclass(frozen=True)
class InputCheckResult:
    """What the API reports in ``input_check``."""

    score: float
    threshold: float
    passed: bool
    #: the caller asked not to be blocked (``?skip_input_check=true``); the score is still computed and recorded
    skipped: bool
    model: str
    version: str
    inference_ms: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["score"] = round(out["score"], 6)
        out["inference_ms"] = round(out["inference_ms"], 1)
        return out


class InputChecker:
    """Loads the ONNX image encoder and the rule once; ``check`` scores one photo."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("FROSTGUARD_GATE_PATH") or DEFAULT_GATE_PATH)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"input check model not found: {self.model_path}")
        config = json.loads(Path(config_path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        self.threshold: float = float(config["threshold"])
        self.version: str = config["version"]
        self.model_name: str = config["model"]["name"]
        self.prototypes = np.asarray(config["prototypes"]["vectors"], dtype=np.float64)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        threads = intra_op_threads or int(os.getenv("FROSTGUARD_ORT_THREADS", "0") or 0)
        if threads:  # same settings as the detector on fractional-CPU hosts
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])

    def embed(self, images: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """[N, 224, 224, 3] float32 0..255 -> [N, 512] unit vectors."""
        out = self.session.run(None, {"image": images.astype(np.float32)})[0]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def score(self, embedding: npt.NDArray[np.floating]) -> float:
        e = np.asarray(embedding, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(e))
        if e.shape != (self.prototypes.shape[1],) or not np.isfinite(norm) or norm == 0:
            raise ValueError("malformed embedding")
        return float((self.prototypes @ (e / norm)).max())

    def check(self, image: Image.Image | bytes | str | Path, skip: bool = False) -> InputCheckResult:
        t0 = time.perf_counter()
        s = self.score(self.embed(prepare(image)[None])[0])
        return InputCheckResult(
            score=s,
            threshold=self.threshold,
            passed=s >= self.threshold,
            skipped=skip,
            model=self.model_name,
            version=self.version,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def warmup(self) -> None:
        self.embed(np.zeros((1, GATE_SIZE, GATE_SIZE, 3), dtype=np.float32))
