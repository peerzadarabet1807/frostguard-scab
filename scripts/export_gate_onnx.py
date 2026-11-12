"""Export the input check's image encoder to ONNX for the API.  [PyTorch venv, see requirements-gate.txt]

The image encoder of TinyCLIP ViT-8M/16 (wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M, Microsoft, MIT licence), pinned
to one Hugging Face revision, wrapped so that it takes the photo as the API prepares it and returns a unit vector:

    input   image      float32 [N, 224, 224, 3]  RGB 0..255 (CLIP normalisation happens inside the graph)
    output  embedding  float32 [N, 512]          unit length

Exported once in float32 (opset 17), then converted to float16 weights with float32 inputs and outputs
(onnxruntime.transformers.float16, keep_io_types). The float16 file ships in models/gate/; both are checked against
PyTorch on the five sample leaves (cosine >= 0.999 required) and the result is written to
results/input_check/onnx_export.json. The same gate family and weights run in the poultry and saffron web apps.

Usage: python scripts/export_gate_onnx.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.input_check import GATE_SIZE, prepare  # noqa: E402

REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
WORK = ROOT / ".work" / "gate"
OUT = ROOT / "models" / "gate" / "image_encoder.fp16.onnx"
REPORT = ROOT / "results" / "input_check" / "onnx_export.json"
MIN_COS = 0.999


def encoder():
    """torch module: float32 [N, 224, 224, 3] in 0..255 -> unit-length embedding (the gate's reference)."""
    import torch
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(REPO, revision=REVISION).eval()

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
            self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1) * 255.0)
            self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1) * 255.0)

        def forward(self, x):
            x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
            f = self.inner.get_image_features(x)
            return f / f.norm(dim=-1, keepdim=True)

    return ImageEncoder().eval()


def embed(enc, images: np.ndarray, batch: int = 32) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out.append(enc(torch.from_numpy(images[i : i + batch].astype(np.float32))).numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16

    enc = encoder()
    WORK.mkdir(parents=True, exist_ok=True)
    fp32 = WORK / "image_encoder.fp32.onnx"
    torch.onnx.export(
        enc,
        torch.zeros(1, GATE_SIZE, GATE_SIZE, 3),
        str(fp32),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(OUT))

    samples = sorted((ROOT / "data" / "samples").glob("*.jpg"))
    images = np.stack([prepare(p.read_bytes()) for p in samples])
    ref = embed(enc, images)
    report = {
        "description": "ONNX export of the input check's image encoder vs PyTorch on data/samples/ "
        "(scripts/export_gate_onnx.py)",
        "repo": REPO,
        "revision": REVISION,
        "images": [p.name for p in samples],
        "required_min_cosine": MIN_COS,
    }
    for label, path in (("fp32", fp32), ("fp16", OUT)):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"image": images.astype(np.float32)})[0]
        got /= np.linalg.norm(got, axis=1, keepdims=True)
        cos = (got * ref).sum(axis=1)
        report[label] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cosine_vs_pytorch_min": float(cos.min()),
            "shipped": path == OUT,
        }
        print(f"{label}: {path.stat().st_size / 1e6:.1f} MB, cosine vs PyTorch min {cos.min():.7f}")
        assert cos.min() >= MIN_COS, label
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
