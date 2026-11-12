"""Write docs/<set>/CREDITS.md, a readable table of docs/<set>/credits.json (photo, author, licence, source).

Usage: python scripts/write_credits.py ood|orchard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"ood": "out-of-distribution photos", "orchard": "orchard photos of apple leaves"}


def main(which: str) -> int:
    root = ROOT / "docs" / which
    doc = json.loads((root / "credits.json").read_text("utf-8"))
    lines = [
        f"# Credits: {TITLES[which]}",
        "",
        f"All {len(doc['photos'])} photos in this folder come from Wikimedia Commons. {doc['note']}",
        "",
        "| File | Category | Author | License | Source | Changes |",
        "|---|---|---|---|---|---|",
    ]
    for c in doc["photos"]:
        lic = f"[{c['license']}]({c['license_url']})" if c.get("license_url") else c["license"]
        lines.append(
            f"| `{c['file']}` | {c['category'].replace('_', ' ')} | {c['author'].replace('|', '/')} | {lic} | "
            f"[Wikimedia Commons]({c['source']}) | {c.get('changes', '')} |"
        )
    (root / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(doc['photos'])} photos -> {root / 'CREDITS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
