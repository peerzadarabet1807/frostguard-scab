"""Split the Commons photo sets once, before any tuning, with a fixed seed (stratified by category).

* docs/ood/: ood-tune (about 40%) and ood-test (about 60%).
* docs/orchard/: prototype (about 30%, builds the orchard prototypes), tune (about 30%, joins the threshold choice)
  and test (the rest, reported only).

Writes docs/<set>/split.json and refuses to overwrite it (the split is fixed from then on).

Usage: python scripts/split_sets.py ood|orchard [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261112
SHARES = {"ood": {"tune": 0.4}, "orchard": {"prototype": 0.3, "tune": 0.3}}
REST = {"ood": "test", "orchard": "test"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=sorted(SHARES))
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (only if the set changed)")
    args = ap.parse_args()
    root = ROOT / "docs" / args.which
    out = root / "split.json"
    if out.exists() and not args.force:
        print(f"{out} exists; the split is fixed.")
        return 1
    photos = json.loads((root / "credits.json").read_text("utf-8"))["photos"]
    rng = random.Random(SEED)
    parts: dict[str, list[str]] = {p: [] for p in [*SHARES[args.which], REST[args.which]]}
    per_category = {}
    for cat in sorted({p["category"] for p in photos}):
        members = sorted(p["file"] for p in photos if p["category"] == cat)
        rng.shuffle(members)
        start, counts = 0, {}
        for part, share in SHARES[args.which].items():
            n = round(share * len(members))
            parts[part] += members[start : start + n]
            counts[part] = n
            start += n
        parts[REST[args.which]] += members[start:]
        counts[REST[args.which]] = len(members) - start
        per_category[cat] = counts
    doc = {
        "description": f"Fixed split of docs/{args.which}/, written once by scripts/split_sets.py.",
        "seed": SEED,
        "shares": SHARES[args.which],
        "per_category": per_category,
        **{k: sorted(v) for k, v in parts.items()},
    }
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
