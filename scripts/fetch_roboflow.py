"""Download the detector's training data, "PDD - Apple - Scab" v1 (Roboflow Universe, CC BY 4.0), to .work/.

The input check reports its false-rejection rate on these 200 scab photos too (they are what the detector was
trained on). Needs ROBOFLOW_API_KEY in the environment; the key is only read from there, never printed or stored.
The download (YOLO format zip, unpacked) lands in .work/roboflow/ (gitignored).

Usage: ROBOFLOW_API_KEY=... python scripts/fetch_roboflow.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".work" / "roboflow"
WORKSPACE, PROJECT, VERSION = "thesis-okplj", "pdd-apple-scab", 1  # notebooks/train_yolo_colab_executed.ipynb


def main() -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("ROBOFLOW_API_KEY is not set", file=sys.stderr)
        return 1
    meta_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/yolov11?api_key={key}"
    try:
        with urllib.request.urlopen(meta_url, timeout=120) as r:  # noqa: S310
            link = json.loads(r.read())["export"]["link"]
        with urllib.request.urlopen(link, timeout=600) as r:  # noqa: S310
            data = r.read()
    except Exception as exc:  # noqa: BLE001 (report the type only: messages could contain the URL and its key)
        print(f"download failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(OUT)
    images = sorted(p for p in OUT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"{WORKSPACE}/{PROJECT} v{VERSION}: {len(images)} images in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
