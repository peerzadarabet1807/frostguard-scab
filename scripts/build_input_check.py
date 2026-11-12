"""Fix the input check's prototypes and threshold, and evaluate it once on the test sets.

The gate family chosen in the poultry repository's candidate study is reused with FrostGuard's own prototypes and
threshold. Embeddings come from the shipped ONNX image encoder (models/gate/, src/vision/input_check.py), so what
is measured is what the API runs.

In-domain photos (apple leaves, healthy or diseased; the check must pass them):
* PlantVillage lab photos, four classes (data/plantvillage_selection.json, scripts/fetch_plantvillage.py):
  30 per class build the prototypes, 30 per class tune the threshold, 30 per class are test photos;
* orchard photos from Wikimedia Commons (docs/orchard/, split by scripts/split_sets.py into prototype, tune and
  test), reported separately because lab photos alone prove little;
* the detector's own training data, "PDD - Apple - Scab" v1 (200 photos, scripts/fetch_roboflow.py), test only;
* the shipped sample leaves (data/samples/ and data/sample_leaf_*.jpg).

Rule: score = cosine with the nearest of seven prototypes (four PlantVillage classes, three orchard categories);
threshold = the highest value with at most 1% false rejection on the pooled tune photos (PlantVillage and orchard).
Every other rate is on test photos and ood-test only, with 95% Wilson intervals.

Writes models/input_check.json, results/input_check/{evaluation.json, evaluation.md, summary.json,
figures/input_check_roc.png} and docs/ood/input_check.json.

Usage (a venv with numpy, pillow, onnxruntime and matplotlib):
    python scripts/fetch_plantvillage.py && python scripts/fetch_roboflow.py   # .work/, once
    python scripts/build_input_check.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.input_check import DEFAULT_GATE_PATH, prepare  # noqa: E402

WORK = ROOT / ".work"
OUT = ROOT / "results" / "input_check"
CONFIG = ROOT / "models" / "input_check.json"
NAME = "tinyclip-vit-8m-16-yfcc15m"
REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
VERSION = "frostguard-tinyclip-8m16-proto-1"
MAX_FRR = 0.01
DECIMALS = 8
PV_LABEL = {
    "Apple___healthy": "healthy",
    "Apple___Apple_scab": "scab",
    "Apple___Black_rot": "black_rot",
    "Apple___Cedar_apple_rust": "cedar_apple_rust",
}


# ----------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * lo:.1f} to {100 * hi:.1f})"


def threshold_for(scores: np.ndarray, max_frr: float = MAX_FRR) -> float:
    s = np.sort(scores)
    return float(s[int(math.floor(max_frr * len(s)))])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------------- photo sets


def photo_sets() -> list[dict]:
    """Every photo: set, part, label, path."""
    rows = []
    pv = json.loads((ROOT / "data" / "plantvillage_selection.json").read_text("utf-8"))
    for r in pv["photos"]:
        rows.append(
            {
                "set": "plantvillage",
                "part": r["part"],
                "label": PV_LABEL[r["class"]],
                "id": f"{r['class']}/{r['file']}",
                "path": WORK / "plantvillage" / r["class"] / r["file"],
            }
        )
    for name in ("orchard", "ood"):
        root = ROOT / "docs" / name
        credits = json.loads((root / "credits.json").read_text("utf-8"))["photos"]
        split = json.loads((root / "split.json").read_text("utf-8"))
        part = {f: k for k in ("prototype", "tune", "test") for f in split.get(k, [])}
        for p in credits:
            rows.append(
                {
                    "set": name,
                    "part": part[p["file"]],
                    "label": p["category"],
                    "proximity": p["proximity"],
                    "id": p["file"],
                    "path": root / p["file"],
                }
            )
    robo = sorted(p for p in (WORK / "roboflow").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows += [
        {"set": "roboflow", "part": "test", "label": "scab", "id": p.name, "path": p} for p in robo
    ]
    samples = sorted((ROOT / "data" / "samples").glob("*.jpg")) + sorted((ROOT / "data").glob("sample_leaf_*.jpg"))
    rows += [
        {"set": "samples", "part": "test", "label": "sample", "id": p.relative_to(ROOT).as_posix(), "path": p}
        for p in samples
    ]
    missing = [str(r["path"]) for r in rows if not Path(r["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} photos missing, e.g. {missing[0]} (run the fetch scripts)")
    return rows


def embed_all(rows: list[dict]) -> np.ndarray:
    """Embed every photo with the shipped ONNX encoder (cached in .work/gate/embeddings.npz)."""
    import onnxruntime as ort
    cache = WORK / "gate" / "embeddings.npz"
    ids = [f"{r['set']}:{r['id']}" for r in rows]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids:
            return z["emb"].astype(np.float64)
    session = ort.InferenceSession(str(DEFAULT_GATE_PATH), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(rows), 16):
        batch = np.stack([prepare(r["path"]) for r in rows[i : i + 16]])
        e = session.run(None, {"image": batch})[0]
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        print(f"embedded {min(i + 16, len(rows))}/{len(rows)}", end="\r", flush=True)
    emb = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), emb=emb)
    print()
    return emb.astype(np.float64)


# -------------------------------------------------------------------------------------------------- evaluate


def frr_block(scores: np.ndarray, labels: list[str], t: float) -> dict:
    cats = sorted(set(labels))
    return {
        "overall": rate(int((scores < t).sum()), len(scores)),
        "by_label": {c: rate(int(sum(s < t for s, lab in zip(scores, labels) if lab == c)), labels.count(c)) for c in cats},
    }


def far_block(scores: np.ndarray, rows: list[dict], t: float) -> dict:
    prox = {r["label"]: r["proximity"] for r in rows}
    cats = sorted(prox, key=lambda c: (prox[c] != "near", c))

    def acc(sel):
        idx = [i for i, r in enumerate(rows) if sel(r)]
        return rate(int(sum(scores[i] >= t for i in idx)), len(idx))

    return {
        "overall": acc(lambda r: True),
        "near": acc(lambda r: r["proximity"] == "near"),
        "far": acc(lambda r: r["proximity"] == "far"),
        "by_category": {c: {"proximity": prox[c], **acc(lambda r, c=c: r["label"] == c)} for c in cats},
        "accepted": sorted(r["id"] for r, s in zip(rows, scores) if s >= t),
    }


def figure(roc: dict, at_t: dict, t: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(roc["far"], roc["frr_lab"], label="lab test photos vs all ood-test", color="#1f6f8b")
    ax.plot(roc["far"], roc["frr_orchard"], label="orchard test photos vs all ood-test", color="#2e7d32")
    ax.plot(roc["far_near"], roc["frr_orchard"], label="orchard test vs near ood-test", color="#b5651d", ls="--")
    ax.scatter([at_t["far"]], [at_t["frr_lab"]], color="#1f6f8b", zorder=3)
    ax.scatter([at_t["far"]], [at_t["frr_orchard"]], color="#2e7d32", zorder=3)
    ax.scatter([at_t["far_near"]], [at_t["frr_orchard"]], color="#b5651d", zorder=3, label=f"threshold {t:.4f}")
    ax.set_xlabel("false acceptance (OOD photos that pass)")
    ax.set_ylabel("false rejection (apple-leaf test photos that fail)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Input check on the test sets (TinyCLIP 8M/16, prototype score)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    rows = photo_sets()
    emb = embed_all(rows)

    def sel(**kw):
        return [i for i, r in enumerate(rows) if all(r.get(k) == v for k, v in kw.items())]

    names, protos = [], []
    for lab in PV_LABEL.values():
        names.append(f"plantvillage:{lab}")
        protos.append(emb[sel(set="plantvillage", part="prototype", label=lab)].mean(axis=0))
    for lab in sorted({r["label"] for r in rows if r["set"] == "orchard"}):
        idx = sel(set="orchard", part="prototype", label=lab)
        if idx:
            names.append(f"orchard:{lab}")
            protos.append(emb[idx].mean(axis=0))
    protos = np.stack(protos)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    protos = np.round(protos, DECIMALS)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores = (unit @ protos.T).max(axis=1)

    tune_idx = sel(set="plantvillage", part="tune") + sel(set="orchard", part="tune")
    t = threshold_for(scores[tune_idx])
    tuning = rate(int((scores[tune_idx] < t).sum()), len(tune_idx))
    counts = {n: len(sel(set=n.split(":")[0], part="prototype", label=n.split(":")[1])) for n in names}
    config = {
        "description": (
            "Input check: does this photo look like an apple leaf? Pass when the cosine similarity between the unit "
            "image embedding and the nearest prototype is at least the threshold. It says nothing about scab. A "
            "safeguard, not a guarantee; measured rates are in results/input_check/evaluation.json."
        ),
        "version": VERSION,
        "model": {
            "name": NAME,
            "repo": REPO,
            "revision": REVISION,
            "license": "MIT, Copyright (c) Microsoft Corporation (model card; microsoft/Cream TinyCLIP/LICENSE)",
            "input": "the photo with EXIF orientation applied, RGB, squashed to 224 x 224 (PIL bilinear), float32 0..255",
            "output": "unit-length image embedding",
            "embedding_dim": int(protos.shape[1]),
            "weights": "models/gate/image_encoder.fp16.onnx (ONNX, float16 weights)",
        },
        "score": "max over prototypes of cosine(image embedding, prototype)",
        "threshold": t,
        "rule": f"pass if score >= threshold; threshold = highest value with false rejection <= {MAX_FRR:.0%} on the "
        "tune photos (PlantVillage and orchard)",
        "tuning": {"photos": len(tune_idx), "false_rejection": tuning},
        "prototypes": {
            "source": "normalised mean embedding of the prototype photos of each PlantVillage class and orchard "
            f"category, rounded to {DECIMALS} decimals",
            "names": names,
            "photos": counts,
            "vectors": protos.tolist(),
        },
        "source": "scripts/build_input_check.py",
    }
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8", newline="\n")

    def labels(idx):
        return [rows[i]["label"] for i in idx]

    lab_test, orch_test = sel(set="plantvillage", part="test"), sel(set="orchard", part="test")
    robo, samp = sel(set="roboflow"), sel(set="samples")
    ood_test, ood_tune = sel(set="ood", part="test"), sel(set="ood", part="tune")
    near_test = [i for i in ood_test if rows[i]["proximity"] == "near"]

    det = json.loads((ROOT / "docs" / "ood" / "detector_predictions.json").read_text("utf-8"))
    det_by = {p["file"]: p for p in det["photos"]}
    scab_alone = [det_by[rows[i]["id"]]["scab_detected"] for i in ood_test]
    told_clean = [not x for x in scab_alone]
    passed = [scores[i] >= t for i in ood_test]

    all_test = np.concatenate([scores[lab_test], scores[orch_test]])
    grid = np.unique(np.quantile(np.concatenate([all_test, scores[ood_test]]), np.linspace(0, 1, 401)))
    roc = {
        "threshold": grid.tolist(),
        "frr_lab": [float((scores[lab_test] < g).mean()) for g in grid],
        "frr_orchard": [float((scores[orch_test] < g).mean()) for g in grid],
        "far": [float((scores[ood_test] >= g).mean()) for g in grid],
        "far_near": [float((scores[near_test] >= g).mean()) for g in grid],
    }
    at_t = {
        "frr_lab": float((scores[lab_test] < t).mean()),
        "frr_orchard": float((scores[orch_test] < t).mean()),
        "far": float((scores[ood_test] >= t).mean()),
        "far_near": float((scores[near_test] >= t).mean()),
    }
    ev = {
        "description": (
            "Evaluation of the shipped input check. The threshold comes from the tune photos only; every rate below "
            "except 'tuning' is on test photos and ood-test. 95% Wilson intervals."
        ),
        "version": VERSION,
        "model": NAME,
        "threshold": t,
        "tuning": {
            "false_rejection": frr_block(scores[tune_idx], [f"{rows[i]['set']}:{rows[i]['label']}" for i in tune_idx], t),
            "false_acceptance_ood_tune": far_block(scores[ood_tune], [rows[i] for i in ood_tune], t),
        },
        "false_rejection": {
            "plantvillage_test": frr_block(scores[lab_test], labels(lab_test), t),
            "orchard_test": frr_block(scores[orch_test], labels(orch_test), t),
            "detector_training_data": rate(int((scores[robo] < t).sum()), len(robo)),
        },
        "false_acceptance": far_block(scores[ood_test], [rows[i] for i in ood_test], t),
        "auroc_lab_test_vs_ood_test": auroc(scores[lab_test], scores[ood_test]),
        "auroc_orchard_test_vs_ood_test": auroc(scores[orch_test], scores[ood_test]),
        "samples": [{"id": rows[i]["id"], "score": round(float(scores[i]), 6), "passed": bool(scores[i] >= t)} for i in samp],
        "posthoc": {
            "note": "Post-hoc observations, not used for any choice.",
            "ood_test_detector_alone": {
                "scab_reported": rate(sum(scab_alone), len(ood_test)),
                "told_leaf_looks_clean": rate(sum(told_clean), len(ood_test)),
            },
            "ood_test_after_input_check": {
                "scab_reported": rate(sum(a and p for a, p in zip(scab_alone, passed)), len(ood_test)),
                "no_scab_message": rate(sum(c and p for c, p in zip(told_clean, passed)), len(ood_test)),
            },
        },
        "roc_test": roc,
        "source": "scripts/build_input_check.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8", newline="\n")
    ood_rows = [
        {
            "file": rows[i]["id"],
            "category": rows[i]["label"],
            "proximity": rows[i]["proximity"],
            "split": rows[i]["part"],
            "score": round(float(scores[i]), 6),
            "passed": bool(scores[i] >= t),
            "detector_scab_detected": det_by[rows[i]["id"]]["scab_detected"],
        }
        for i in sel(set="ood")
    ]
    (ROOT / "docs" / "ood" / "input_check.json").write_text(
        json.dumps({"description": "Input check score and decision for every OOD photo, with the detector's answer "
                    "(scripts/build_input_check.py).", "version": VERSION, "threshold": t, "photos": ood_rows},
                   indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure(roc, at_t, t, OUT / "figures" / "input_check_roc.png")
    fr, fa = ev["false_rejection"], ev["false_acceptance"]
    summary = {
        "description": "The numbers the API and web app show about the input check (from evaluation.json).",
        "version": VERSION,
        "model": {"name": NAME, "repo": REPO, "license": "MIT"},
        "threshold": t,
        "false_rejection": {
            "tune": tuning,
            "plantvillage_test": fr["plantvillage_test"]["overall"],
            "orchard_test": fr["orchard_test"]["overall"],
            "detector_training_data": fr["detector_training_data"],
        },
        "false_acceptance": {"near": fa["near"], "far": fa["far"], "overall": fa["overall"]},
        "samples": {"passed": sum(s["passed"] for s in ev["samples"]), "n": len(ev["samples"])},
        "source": "scripts/build_input_check.py",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_markdown(ev, config)
    print(f"threshold {t:.6f}  tuning FRR {pct(tuning)}")
    print(f"FRR lab {pct(fr['plantvillage_test']['overall'])}  orchard {pct(fr['orchard_test']['overall'])}  "
          f"detector training data {pct(fr['detector_training_data'])}")
    print(f"FAR near {pct(fa['near'])}  far {pct(fa['far'])}")
    print(f"samples failing: {[s['id'] for s in ev['samples'] if not s['passed']]}")
    return 0


def write_markdown(ev: dict, config: dict) -> None:
    fr, fa, ph = ev["false_rejection"], ev["false_acceptance"], ev["posthoc"]
    tn = ev["tuning"]
    lines = [
        "# Input check: evaluation of the shipped gate",
        "",
        f"Written by [`scripts/build_input_check.py`](../../scripts/build_input_check.py). Gate `{ev['version']}`: "
        "TinyCLIP ViT-8M/16 image embedding (the shipped ONNX file), cosine to the nearest of "
        f"{len(config['prototypes']['names'])} prototypes ({', '.join(config['prototypes']['names'])}), threshold "
        f"{ev['threshold']:.6f}: the highest value with at most 1% false rejection on the {config['tuning']['photos']} "
        "tune photos. Test photos and ood-test played no part in any choice. Rates: percent (count; 95% Wilson "
        "interval).",
        "",
        "## False rejection (apple-leaf photos the check turns away)",
        "",
        "| Set | Tune | Test |",
        "|---|---|---|",
    ]
    for lab, r in fr["plantvillage_test"]["by_label"].items():
        t_r = tn["false_rejection"]["by_label"].get(f"plantvillage:{lab}")
        lines.append(f"| PlantVillage (lab): {lab.replace('_', ' ')} | {pct(t_r) if t_r else ''} | {pct(r)} |")
    lines.append(f"| **PlantVillage, all** | | {pct(fr['plantvillage_test']['overall'])} |")
    for lab, r in fr["orchard_test"]["by_label"].items():
        t_r = tn["false_rejection"]["by_label"].get(f"orchard:{lab}")
        lines.append(f"| Orchard (Commons): {lab.replace('_', ' ')} | {pct(t_r) if t_r else ''} | {pct(r)} |")
    lines += [
        f"| **Orchard, all** | | {pct(fr['orchard_test']['overall'])} |",
        f"| Detector training data (PDD - Apple - Scab v1, all scab) | | {pct(fr['detector_training_data'])} |",
        f"| **Tune photos, all** | {pct(tn['false_rejection']['overall'])} | |",
        "",
        "## False acceptance (ood-test photos that pass)",
        "",
        "| Category | Near or far | Passed |",
        "|---|---|---|",
    ]
    for c, r in fa["by_category"].items():
        lines.append(f"| {c.replace('_', ' ')} | {r['proximity']} | {pct(r)} |")
    lines += [
        f"| **All near** | near | {pct(fa['near'])} |",
        f"| **All far** | far | {pct(fa['far'])} |",
        f"| **All** | | {pct(fa['overall'])} |",
        "",
        f"On ood-tune: near {pct(tn['false_acceptance_ood_tune']['near'])}, far "
        f"{pct(tn['false_acceptance_ood_tune']['far'])}.",
        "",
        f"Passing ood-test photos: {', '.join(f'`{f}`' for f in fa['accepted']) or 'none'}.",
        "",
        f"AUROC against ood-test: lab test photos {ev['auroc_lab_test_vs_ood_test']:.4f}, orchard test photos "
        f"{ev['auroc_orchard_test_vs_ood_test']:.4f}.",
        "",
        "![False rejection against false acceptance on the test sets](figures/input_check_roc.png)",
        "",
        "## Shipped sample leaves",
        "",
        f"{sum(s['passed'] for s in ev['samples'])} of {len(ev['samples'])} pass.",
        "",
        "## Post-hoc observations",
        "",
        "Not used for any choice. On ood-test photos, without the input check the detector reported scab on "
        f"{pct(ph['ood_test_detector_alone']['scab_reported'])} and the app called the leaf clean on "
        f"{pct(ph['ood_test_detector_alone']['told_leaf_looks_clean'])}. With the check in front: scab reported on "
        f"{pct(ph['ood_test_after_input_check']['scab_reported'])}, \"no scab lesions found\" on "
        f"{pct(ph['ood_test_after_input_check']['no_scab_message'])}; every other photo gets the notice instead.",
        "",
    ]
    (OUT / "evaluation.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
