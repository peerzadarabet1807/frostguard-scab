"""Build ``models/model_card.json`` from a Colab training run.

Reads ``runs/<run>/args.yaml`` + ``results.csv`` and the final ``best.val()`` line printed
by the executed notebook, and records the exported ONNX file's size and SHA-256 so
``scripts/download_model.py`` can verify release downloads.

Usage::

    python scripts/build_model_card.py            # defaults: runs/scab_yolo11n
    python scripts/build_model_card.py --run runs/my_run --notebook notebooks/x.ipynb
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

RELEASE_TAG = "model-v1"
RELEASE_BASE = f"https://github.com/peerzadarabet1807/frostguard-scab/releases/download/{RELEASE_TAG}"
FINAL_METRICS = re.compile(
    r"mAP50:\s*(?P<map50>[\d.]+)\s*\|\s*mAP50-95:\s*(?P<map>[\d.]+)\s*\|\s*"
    r"precision:\s*(?P<p>[\d.]+)\s*\|\s*recall:\s*(?P<r>[\d.]+)"
)
SPLITS = re.compile(r"->\s*train:\s*(?P<train>\d+)\s*\|\s*val\s*\((?P<vsplit>\w+)\):\s*(?P<val>\d+)")
SUMMARY = re.compile(r"summary \(fused\):\s*(?P<layers>\d+) layers,\s*(?P<params>[\d,]+) parameters.*?(?P<gflops>[\d.]+) GFLOPs")


def notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if "text" in out:
                chunks.append("".join(out["text"]))
            elif "text/plain" in out.get("data", {}):
                chunks.append("".join(out["data"]["text/plain"]))
    return "\n".join(chunks)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(run: Path, notebook: Path, model: Path) -> dict[str, Any]:
    args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    with (run / "results.csv").open(encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    # Ultralytics' checkpoint fitness: 0.1 * mAP50 + 0.9 * mAP50-95.
    best = max(rows, key=lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]))

    text = notebook_text(notebook)
    final = FINAL_METRICS.search(text)
    splits = SPLITS.search(text)
    summary = SUMMARY.search(text)
    if final is None:
        raise SystemExit(f"could not find the final 'mAP50: ... | recall: ...' line in {notebook}")

    return {
        "name": "FrostGuard apple scab detector",
        "architecture": "YOLO11n (Ultralytics), fine-tuned from COCO-pretrained yolo11n.pt",
        "task": "object detection",
        "classes": ["Apple - Scab"],
        "input_size": int(args["imgsz"]),
        "parameters": int(summary["params"].replace(",", "")) if summary else None,
        "gflops": float(summary["gflops"]) if summary else None,
        "dataset": {
            "name": "PDD – Apple – Scab (Roboflow Universe, thesis-okplj/pdd-apple-scab v1)",
            "url": "https://universe.roboflow.com/thesis-okplj/pdd-apple-scab",
            "license": "CC BY 4.0",
            "origin": "PlantVillage leaf photographs (lab conditions, single leaf on plain background)",
            "train_images": int(splits["train"]) if splits else None,
            "val_images": int(splits["val"]) if splits else None,
            "val_split": "20 % random hold-out of train (seed 42); the dataset version ships no validation split",
        },
        "training": {
            "epochs": int(args["epochs"]),
            "epochs_completed": len(rows),
            "best_epoch": int(float(best["epoch"])),
            "batch": int(args["batch"]),
            "optimizer": args["optimizer"],
            "lr0": float(args["lr0"]),
            "cos_lr": bool(args["cos_lr"]),
            "seed": int(args["seed"]),
            "hardware": "Google Colab, NVIDIA Tesla T4",
        },
        "metrics": {
            "split": "val",
            "mAP50": float(final["map50"]),
            "mAP50_95": float(final["map"]),
            "precision": float(final["p"]),
            "recall": float(final["r"]),
        },
        "export": {
            "format": "ONNX (opset 17, simplified, static 1x3x640x640)",
            "file": model.name,
            "size_bytes": model.stat().st_size,
            "sha256": sha256(model),
            "release_tag": RELEASE_TAG,
            "url": f"{RELEASE_BASE}/{model.name}",
        },
        "intended_use": "Decision support: confirming visible scab lesions on single apple-leaf photos.",
        "limitations": [
            "Trained on 160 lab photos of single leaves; expect false positives on fruit, field scenes, "
            "microscopy or cluttered backgrounds.",
            "Validation set is small (40 images), so metrics carry wide uncertainty.",
            "Weights derive from Ultralytics YOLO11 and are distributed under AGPL-3.0.",
        ],
        "license": "AGPL-3.0 (model weights)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=PROJECT_ROOT / "runs" / "scab_yolo11n")
    parser.add_argument("--notebook", type=Path, default=PROJECT_ROOT / "notebooks" / "train_yolo_colab_executed.ipynb")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "scab_detector.onnx")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "model_card.json")
    args = parser.parse_args()
    card = build(args.run, args.notebook, args.model)
    args.out.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    m = card["metrics"]
    print(f"wrote {args.out.relative_to(PROJECT_ROOT)}: mAP50={m['mAP50']} mAP50-95={m['mAP50_95']} sha256={card['export']['sha256'][:12]}…")


if __name__ == "__main__":
    main()
