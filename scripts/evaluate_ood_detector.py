"""What does the scab detector (without any input check) do on photos that are not apple leaves?

Runs the trained YOLO11n detector (models/scab_detector.onnx, scripts/download_model.py) on every photo in
docs/ood/ and records, per photo, whether it reports scab and how many lesions. In the web app before the input
check, a photo with lesions showed "Scab lesions detected" and a photo without any showed "No lesions detected ...
The leaf looks clean" (web/src/lib/risk.ts at that time), whatever the photo was. Writes
docs/ood/detector_predictions.json. Measured before any input check code ran on these photos.

Usage: python scripts/evaluate_ood_detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.detector import ScabVisionDetector  # noqa: E402

OOD = ROOT / "docs" / "ood"


def main() -> int:
    detector = ScabVisionDetector()
    if detector.is_fallback:
        print("models/scab_detector.onnx is missing: run scripts/download_model.py first", file=sys.stderr)
        return 1
    photos = json.loads((OOD / "credits.json").read_text("utf-8"))["photos"]
    rows = []
    for p in photos:
        r = detector.detect(OOD / p["file"])
        conf = [d.confidence for d in r.detections]
        rows.append(
            {
                "file": p["file"],
                "category": p["category"],
                "proximity": p["proximity"],
                "scab_detected": bool(r.detections),
                "lesion_count": len(r.detections),
                "max_confidence": round(max(conf), 4) if conf else None,
                "app_showed": "scab lesions detected" if r.detections else "no lesions detected, leaf looks clean",
            }
        )
    cats = {}
    for c in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == c]
        cats[c] = {"proximity": sub[0]["proximity"], "n": len(sub), "scab_detected": sum(r["scab_detected"] for r in sub)}
    doc = {
        "description": (
            "The trained scab detector, with no input check, on every OOD photo (scripts/evaluate_ood_detector.py). "
            "Every photo got an answer about scab; none was told it is not an apple leaf."
        ),
        "model": "models/scab_detector.onnx (model-v1 release)",
        "conf_threshold": detector.conf_threshold,
        "summary": {
            "n": len(rows),
            "scab_detected": sum(r["scab_detected"] for r in rows),
            "told_leaf_looks_clean": sum(not r["scab_detected"] for r in rows),
            "by_category": cats,
        },
        "photos": rows,
    }
    (OOD / "detector_predictions.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    s = doc["summary"]
    print(f"{s['n']} OOD photos: scab reported on {s['scab_detected']}, 'leaf looks clean' on {s['told_leaf_looks_clean']}")
    for c, v in cats.items():
        print(f"  {c:<20} {v['proximity']:<5} scab on {v['scab_detected']} of {v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
