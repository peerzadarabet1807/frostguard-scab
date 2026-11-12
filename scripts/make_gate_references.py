"""PyTorch reference embeddings for the input check's parity test (tests/test_input_check.py).  [PyTorch venv]

Fixtures (all committed): the five sample leaves (data/samples/), the two procedural test leaves (data/), and the
first ood-test photo of every OOD category (docs/ood/split.json). Each photo goes through the API's preparation
(src/vision/input_check.prepare) and the original PyTorch model (scripts/export_gate_onnx.encoder); the rule in
models/input_check.json gives the reference decision. Writes tests/fixtures/input_check_reference.json.

Usage: python scripts/make_gate_references.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_gate_onnx import embed, encoder  # noqa: E402

from src.vision.input_check import prepare  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "input_check_reference.json"


def fixtures() -> list[str]:
    files = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data" / "samples").glob("*.jpg"))]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("sample_leaf_*.jpg"))]
    credits = json.loads((ROOT / "docs" / "ood" / "credits.json").read_text("utf-8"))["photos"]
    test = set(json.loads((ROOT / "docs" / "ood" / "split.json").read_text("utf-8"))["test"])
    first: dict[str, str] = {}
    for p in sorted(credits, key=lambda c: c["file"]):
        if p["file"] in test:
            first.setdefault(p["category"], p["file"])
    files += [f"docs/ood/{f}" for f in first.values()]
    return files


def main() -> int:
    config = json.loads((ROOT / "models" / "input_check.json").read_text("utf-8"))
    files = fixtures()
    emb = embed(encoder(), np.stack([prepare(ROOT / f) for f in files])).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    protos = np.asarray(config["prototypes"]["vectors"], dtype=np.float64)
    scores = (emb @ protos.T).max(axis=1)
    t = config["threshold"]
    images = []
    for f, e, s in zip(files, emb, scores, strict=True):
        images.append({"file": f, "score": float(s), "passed": bool(s >= t), "embedding": np.round(e, 7).tolist()})
        print(f"{f:<90} {s:.6f} {'pass' if s >= t else 'reject'}")
    doc = {
        "description": "PyTorch (Hugging Face) reference embeddings of the input check's image encoder on committed "
        "photos, prepared as the API prepares them. scripts/make_gate_references.py",
        "version": config["version"],
        "threshold": t,
        "images": images,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"min |score - threshold| = {min(abs(i['score'] - t) for i in images):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
