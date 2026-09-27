"""Download the trained detector from the GitHub release and verify its SHA-256.

The ONNX model is git-ignored (it is a ~10 MB binary under AGPL-3.0); the canonical copy
lives on the ``model-v1`` GitHub release, and ``models/model_card.json`` pins its hash.

Usage::

    python scripts/download_model.py            # -> models/scab_detector.onnx
    python scripts/download_model.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARD = PROJECT_ROOT / "models" / "model_card.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download even if a verified copy exists")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "scab_detector.onnx")
    args = parser.parse_args()

    export = json.loads(CARD.read_text(encoding="utf-8"))["export"]
    expected = export["sha256"]
    if args.out.exists() and not args.force and sha256(args.out) == expected:
        print(f"{args.out.name} already present and verified")
        return 0

    tmp = args.out.with_suffix(".part")
    print(f"downloading {export['url']}")
    with urllib.request.urlopen(export["url"], timeout=120) as resp, tmp.open("wb") as f:  # noqa: S310
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        print(f"SHA-256 mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 1
    tmp.replace(args.out)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
