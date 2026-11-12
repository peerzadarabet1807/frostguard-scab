"""Seeded sample of PlantVillage apple leaves (lab photos) for the input check.

PlantVillage (Hughes & Salathe, 2015, arXiv:1511.08060), via spMohanty/PlantVillage-Dataset, CC BY-SA 3.0: the
same source as data/samples/. Four apple classes (healthy, scab, black rot, cedar apple rust) cover healthy and
diseased leaves, so the check is not tuned on scab alone.

* ``--select`` lists each class folder (GitHub contents API; it returns at most the first 1,000 files of a folder in
  name order, which matters only for Apple___healthy), drops the five photos already shipped in data/samples/,
  draws a seeded sample of 90 per class and assigns 30 each to "prototype", "tune" and "test". Writes
  data/plantvillage_selection.json (committed).
* Without ``--select`` it downloads exactly those files to .work/plantvillage/ (gitignored), unmodified.

Usage:
    python scripts/fetch_plantvillage.py --select     # once
    python scripts/fetch_plantvillage.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data" / "plantvillage_selection.json"
OUT = ROOT / ".work" / "plantvillage"
REPO = "spMohanty/PlantVillage-Dataset"
FOLDER = "raw/color"
CLASSES = ["Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust"]
PER_CLASS = 90
PARTS = ("prototype", "tune", "test")
SEED = 20261112
UA = {"User-Agent": "frostguard-scab/1.0 (https://github.com/peerzadarabet1807/frostguard-scab)"}


def get(url: str, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def shipped_samples() -> set[str]:
    text = (ROOT / "data" / "samples" / "ATTRIBUTION.md").read_text("utf-8")
    return {line.split("`")[3].split("/", 1)[1] for line in text.splitlines() if line.startswith("| `") and "___" in line}


def select() -> int:
    skip = shipped_samples()
    rng = random.Random(SEED)
    rows = []
    for cls in CLASSES:
        url = f"https://api.github.com/repos/{REPO}/contents/{urllib.parse.quote(FOLDER + '/' + cls)}"
        names = sorted(e["name"] for e in json.loads(get(url)) if e["type"] == "file" and e["name"] not in skip)
        picks = rng.sample(names, PER_CLASS)
        for i, name in enumerate(picks):
            rows.append({"class": cls, "file": name, "part": PARTS[i * len(PARTS) // PER_CLASS]})
        print(f"{cls:<26} {len(names)} listed, {PER_CLASS} sampled")
        time.sleep(1)
    doc = {
        "description": (
            "Seeded sample of PlantVillage apple leaves (raw/color) for the input check: prototype photos build the "
            "class prototypes, tune photos set the threshold, test photos measure false rejection. "
            "scripts/fetch_plantvillage.py"
        ),
        "source": f"https://github.com/{REPO}/tree/master/{FOLDER}",
        "license": "CC BY-SA 3.0 (Hughes & Salathe, 2015, arXiv:1511.08060)",
        "seed": SEED,
        "per_class": PER_CLASS,
        "excluded": "the five photos in data/samples/",
        "photos": rows,
    }
    SELECTION.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} photos -> {SELECTION}")
    return 0


def fetch() -> int:
    doc = json.loads(SELECTION.read_text("utf-8"))
    for r in doc["photos"]:
        path = OUT / r["class"] / r["file"]
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/master/{urllib.parse.quote(FOLDER + '/' + r['class'] + '/' + r['file'])}"
        path.write_bytes(get(url))
    print(f"{len(doc['photos'])} photos in {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    return select() if args.select else fetch()


if __name__ == "__main__":
    sys.exit(main())
