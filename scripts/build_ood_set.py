"""Build the photo sets for the input check from Wikimedia Commons.

Two sets, chosen the same way:

* ``--set ood`` (default): photos that are not apple leaves (docs/ood/), "near" (leaves of other plants, apple
  fruit, blossom, bark and branches, grass) and "far" (hands, sky, people, documents, screenshots, scenes,
  everyday objects, other animals). The check must reject them.
* ``--set orchard``: real photos of apple leaves, healthy or diseased, taken outdoors (docs/orchard/). The check
  must pass them; they are reported separately from the PlantVillage-style lab photos.

Two steps per set:

1. ``--discover`` runs the Commons searches listed in the set's categories, keeps files whose licence is CC0,
   public domain, CC BY or CC BY-SA, and writes the candidates plus a numbered contact sheet per category to
   ``.work/<set>-discover/`` (gitignored). A person looks at the sheets and writes the chosen Commons titles
   into ``docs/<set>/selection.json``, with a reason for every candidate that was skipped.
2. Without ``--discover`` the script downloads exactly the titles in the selection, downscales each to at most
   640 px on the long side (JPEG quality 85), and writes ``docs/<set>/set/<category>/`` and the credits in
   ``docs/<set>/credits.json`` (title, author, licence, source URL and the change made).

The committed selection file, not the live search ranking, decides what is in a set. Licences are re-checked
on download.

Usage:
    python scripts/build_ood_set.py --discover                 # OOD candidates and contact sheets
    python scripts/build_ood_set.py                            # download the committed OOD selection
    python scripts/build_ood_set.py --set orchard --discover
    python scripts/build_ood_set.py --set orchard
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "frostguard-scab/1.0 (https://github.com/peerzadarabet1807/frostguard-scab)"}
API = "https://commons.wikimedia.org/w/api.php"
OPEN_LICENSES = ("CC0", "CC BY", "Public domain", "PD")
LONG_SIDE = 640

#: set -> slug -> (proximity, description, Commons search queries). In the OOD set "near" categories are
#: look-alikes or things a user might photograph instead of an apple leaf, "far" categories are unrelated (the far
#: titles are the same as the poultry repository's far categories, plus hands and sky). The orchard set is
#: in-domain: apple leaves photographed outdoors.
CATEGORIES: dict[str, dict[str, tuple[str, str, list[str]]]] = {
    "ood": {
        "pear_leaves": ("near", "Pear leaves", ["pear leaves", "Pyrus communis leaf", "pear tree leaf", "pear leaf"]),
        "cherry_leaves": ("near", "Cherry leaves", ["cherry leaves", "Prunus avium leaf", "cherry tree leaf"]),
        "grape_leaves": ("near", "Grape leaves", ["grape leaves", "Vitis vinifera leaf", "vine leaf"]),
        "maple_leaves": ("near", "Maple leaves", ["maple leaf", "Acer leaf", "maple leaves autumn"]),
        "other_leaves": (
            "near",
            "Leaves of other plants",
            ["plum leaf", "quince leaf", "walnut leaf", "oak leaf", "rose leaf", "apricot leaf", "peach leaf"],
        ),
        "apple_fruit": ("near", "Apples (the fruit)", ["apple fruit", "apples on tree", "red apple", "green apple"]),
        "apple_blossom": ("near", "Apple blossom", ["apple blossom", "Malus domestica flowers", "apple flowers"]),
        "bark_branches": ("near", "Bark and branches", ["tree bark", "apple tree bark", "tree branch", "twig"]),
        "grass": ("near", "Grass and lawn", ["grass", "lawn close up", "meadow grass"]),
        "hands": ("far", "Hands", ["hand", "hands close up", "open hand", "palm of hand"]),
        "sky": ("far", "Sky", ["blue sky", "clouds sky", "sky"]),
        "documents": ("far", "Documents and printed pages", ["document page", "printed page", "newspaper page"]),
        "screenshots": ("far", "Screenshots of software and websites", ["screenshot", "software screenshot"]),
        "people": ("far", "People", ["portrait photograph", "people walking", "farmer portrait"]),
        "everyday_objects": ("far", "Everyday objects", ["mug", "keyboard", "chair", "bicycle", "shoes", "car"]),
        "scenes": ("far", "Landscapes, buildings and streets", ["landscape", "city street", "building facade"]),
        "other_animals": ("far", "Other animals", ["cat", "dog portrait", "cow field", "horse", "sheep"]),
    },
    "orchard": {
        "healthy": (
            "in",
            "Healthy apple leaves on the tree",
            ["apple leaves", "Malus domestica leaves", "apple tree leaves", "apple leaf"],
        ),
        "scab": ("in", "Apple scab on leaves", ["apple scab", "Venturia inaequalis", "apple scab leaf"]),
        "other_diseases": (
            "in",
            "Other diseases and damage on apple leaves",
            [
                "cedar apple rust",
                "Gymnosporangium juniperi-virginianae apple",
                "apple powdery mildew",
                "Podosphaera leucotricha",
                "apple leaf spot",
                "apple leaf disease",
                "apple rust leaf",
            ],
        ),
    },
}
CATEGORIES_ALL = {slug: v for cats in CATEGORIES.values() for slug, v in cats.items()}


def paths(which: str) -> tuple[Path, Path, Path, Path]:
    root = ROOT / "docs" / which
    return root, root / "set", root / "selection.json", ROOT / ".work" / f"{which}-discover"


def load_credits(which: str) -> list[dict]:
    path = paths(which)[0] / "credits.json"
    return json.loads(path.read_text("utf-8"))["photos"] if path.exists() else []


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def http_get(url: str, *, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
        except (urllib.error.URLError, ConnectionError, TimeoutError):  # dropped connections: retry too
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def api(params: dict) -> dict:
    data = http_get(f"{API}?{urllib.parse.urlencode({'format': 'json', **params})}")
    time.sleep(1.0)
    return json.loads(data)


def license_ok(name: str) -> bool:
    return name.startswith(OPEN_LICENSES) and "NC" not in name and "ND" not in name


def describe(info: dict, title: str) -> dict:
    meta = info.get("extmetadata", {})
    return {
        "title": title,
        "author": plain(meta.get("Artist", {}).get("value", "unknown")),
        "license": meta.get("LicenseShortName", {}).get("value", ""),
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "source": info["descriptionurl"],
        "width": info.get("width"),
        "height": info.get("height"),
    }


def discover(which: str, limit: int, only: list[str] | None = None) -> int:
    from PIL import Image, ImageDraw

    _, _, _, discover_dir = paths(which)
    discover_dir.mkdir(parents=True, exist_ok=True)
    taken = {p["title"] for w in CATEGORIES for p in load_credits(w)}
    for slug, (_, _, queries) in CATEGORIES[which].items():
        if only and slug not in only:
            continue
        seen: dict[str, dict] = {}
        per_query: list[list[str]] = []
        for query in queries:
            per_query.append([])
            data = api(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{query} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 330,
                }
            )
            pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
            for page in pages:
                title = page["title"].removeprefix("File:")
                info = (page.get("imageinfo") or [{}])[0]
                if title in seen or title in taken or info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 300:
                    continue
                d = describe(info, title)
                if license_ok(d["license"]):
                    seen[title] = {**d, "query": query, "thumb": info["thumburl"]}
                    per_query[-1].append(title)
        # round-robin over the queries, so the first query cannot fill the whole list
        order = [t for group in zip(*[q + [None] * (50 - len(q)) for q in per_query], strict=True) for t in group if t]
        cands = [seen[t] for t in dict.fromkeys(order)][:limit]
        (discover_dir / f"{slug}.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False), "utf-8")
        tiles = []
        for i, c in enumerate(cands):
            try:
                im = Image.open(io.BytesIO(http_get(c["thumb"]))).convert("RGB")
            except Exception:  # noqa: BLE001 (a broken thumbnail just leaves a blank tile)
                im = Image.new("RGB", (200, 200), "grey")
            im.thumbnail((200, 200))
            tile = Image.new("RGB", (200, 220), "white")
            tile.paste(im, ((200 - im.width) // 2, 20))
            ImageDraw.Draw(tile).text((4, 2), str(i), fill="black")
            tiles.append(tile)
            time.sleep(0.3)
        cols = 8
        sheet = Image.new("RGB", (cols * 200, max(1, -(-len(tiles) // cols)) * 220), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 200, (i // cols) * 220))
        sheet.save(discover_dir / f"{slug}.jpg", quality=80)
        print(f"{slug:<24} {len(cands)} candidates", flush=True)
    return 0


def fetch(which: str) -> int:
    from PIL import Image

    root, set_dir, selection_path, _ = paths(which)
    selection = json.loads(selection_path.read_text("utf-8"))
    photos = []
    for slug, entry in selection["categories"].items():
        proximity, description, _ = CATEGORIES[which][slug]
        out = set_dir / slug
        out.mkdir(parents=True, exist_ok=True)
        titles = entry["titles"]
        for i in range(0, len(titles), 25):
            batch = titles[i : i + 25]
            data = api(
                {
                    "action": "query",
                    "titles": "|".join(f"File:{t}" for t in batch),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 960,
                }
            )
            norm = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
            by_title = {}
            for page in data["query"]["pages"].values():
                if "imageinfo" not in page:
                    raise LookupError(f"{page['title']}: missing on Commons")
                by_title[norm.get(page["title"], page["title"]).removeprefix("File:")] = page["imageinfo"][0]
            for title in batch:
                info = by_title[title]
                d = describe(info, title)
                if not license_ok(d["license"]):
                    raise ValueError(f"{title}: licence {d['license']} is not open enough")
                im = Image.open(io.BytesIO(http_get(info.get("thumburl") or info["url"]))).convert("RGB")
                scale = LONG_SIDE / max(im.size)
                if scale < 1:
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                stem = re.sub(r"[^a-z0-9]+", "_", Path(title).stem.lower()).strip("_")[:60]
                if len(stem) < 4:  # titles in other scripts: name the file after a hash of the title
                    stem = f"{slug}_{hashlib.sha1(title.encode()).hexdigest()[:10]}"
                path = out / f"{stem}.jpg"
                n = 2
                while path.relative_to(root).as_posix() in {q["file"] for q in photos}:
                    path, n = out / f"{stem}_{n}.jpg", n + 1
                im.save(path, quality=85, optimize=True)
                photos.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "category": slug,
                        "proximity": proximity,
                        "description": description,
                        **{k: d[k] for k in ("title", "author", "license", "license_url", "source")},
                        "changes": f"Downscaled to {max(im.size)} px on the long side, saved as JPEG quality 85.",
                    }
                )
                print(f"{photos[-1]['file']:<70} {d['license']}", flush=True)
                time.sleep(0.5)
    doc = {
        "note": (
            "Downscaled as stated in 'changes'. Each photo keeps its own licence; the repository's licence does not "
            "apply to them."
        ),
        "photos": photos,
    }
    (root / "credits.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    print(f"{len(photos)} photos in {set_dir.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", choices=sorted(CATEGORIES), default="ood")
    ap.add_argument("--discover", action="store_true", help="write candidates and contact sheets for review")
    ap.add_argument("--limit", type=int, default=40, help="candidates per category with --discover")
    ap.add_argument("--only", action="append", choices=sorted(CATEGORIES_ALL), help="with --discover: these only")
    args = ap.parse_args()
    return discover(args.which, args.limit, args.only) if args.discover else fetch(args.which)


if __name__ == "__main__":
    sys.exit(main())
