"""Reproducible CPU benchmarks for BENCHMARKS.md.

Measures
  1. risk-engine throughput (hourly Mills state machine, hours/second);
  2. per-request risk assessment latency (72 h window);
  3. ONNX Runtime latency (inference only, and end-to-end detect) + model footprint.

Usage::

    python scripts/benchmark.py                                  # engine + fallback detector
    python scripts/benchmark.py --onnx models/scab_detector.onnx # + a trained YOLO11n export
    python scripts/benchmark.py --onnx a.onnx b.onnx --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from src.engine.mills_physics import FrostScabPhysicsEngine  # noqa: E402
from src.engine.weather_fetcher import generate_mock_weather, load_weather_csv  # noqa: E402
from src.vision.detector import ScabVisionDetector  # noqa: E402
from src.vision.fallback_model import FALLBACK_MODEL_PATH  # noqa: E402

SAMPLE_LEAF = PROJECT_ROOT / "data" / "sample_leaf_scab.jpg"


def timeit(fn: Callable[[], Any], runs: int, warmup: int) -> list[float]:
    """Wall-clock milliseconds for ``runs`` calls after ``warmup`` discarded calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def pct(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q))


def cpu_name() -> str:
    if platform.system() == "Windows":
        import subprocess

        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def bench_engine(runs: int) -> dict[str, Any]:
    engine = FrostScabPhysicsEngine()
    five_years = generate_mock_weather(start="2020-01-01", hours=24 * 365 * 5, scenario="spring_mixed", seed=1)
    real = load_weather_csv(PROJECT_ROOT / "data" / "shopian_spring_2024_hourly.csv")
    window = generate_mock_weather(start="2024-04-12", hours=72, scenario="scab_outbreak")

    long_ms = timeit(lambda: engine.compute_timeline(five_years), runs=runs, warmup=1)
    real_ms = timeit(lambda: engine.compute_timeline(real), runs=runs * 4, warmup=2)
    assess_ms = timeit(lambda: engine.assess(window, horizon_hours=48), runs=200, warmup=10)
    return {
        "five_year_hours": len(five_years),
        "five_year_median_ms": statistics.median(long_ms),
        "five_year_hours_per_s": len(five_years) / (statistics.median(long_ms) / 1000.0),
        "shopian_2024_hours": len(real),
        "shopian_2024_hours_per_s": len(real) / (statistics.median(real_ms) / 1000.0),
        "assess_72h_median_ms": statistics.median(assess_ms),
        "assess_72h_p95_ms": pct(assess_ms, 95),
    }


def bench_model(path: Path, runs: int, threads: int | None) -> dict[str, Any]:
    detector = ScabVisionDetector(model_path=path, fallback_path=FALLBACK_MODEL_PATH, intra_op_threads=threads)
    tensor, _ = detector.preprocess(SAMPLE_LEAF)
    infer_ms = timeit(lambda: detector.infer(tensor), runs=runs, warmup=10)
    e2e_ms = timeit(lambda: detector.detect(SAMPLE_LEAF), runs=max(20, runs // 2), warmup=3)
    return {
        "model": path.name,
        "size_mb": path.stat().st_size / 1e6,
        "input": detector.input_size,
        "threads": threads or "auto",
        "infer_median_ms": statistics.median(infer_ms),
        "infer_p95_ms": pct(infer_ms, 95),
        "e2e_median_ms": statistics.median(e2e_ms),
        "throughput_img_s": 1000.0 / statistics.median(infer_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", nargs="*", type=Path, default=[], help="extra ONNX detectors to benchmark")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, nargs="*", default=[0], help="intra-op threads (0 = ORT default)")
    parser.add_argument("--json", type=Path, help="also write raw results here")
    args = parser.parse_args()

    env = {
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }
    print("## Environment\n")
    for key, value in env.items():
        print(f"- **{key}**: {value}")

    engine = bench_engine(runs=5)
    print("\n## Risk engine\n")
    print("| Workload | Result |\n|---|---|")
    print(f"| 5-year hourly series ({engine['five_year_hours']:,} h) | {engine['five_year_hours_per_s']:,.0f} hours/s ({engine['five_year_median_ms']:.0f} ms) |")
    print(f"| Shopian spring 2024, ERA5 ({engine['shopian_2024_hours']:,} h) | {engine['shopian_2024_hours_per_s']:,.0f} hours/s |")
    print(f"| `assess()` on a 72 h request window | {engine['assess_72h_median_ms']:.2f} ms median, {engine['assess_72h_p95_ms']:.2f} ms p95 |")

    models = [FALLBACK_MODEL_PATH, *args.onnx]
    vision = []
    for model in models:
        for threads in args.threads:
            vision.append(bench_model(model, args.runs, threads or None))
    print("\n## Vision (ONNX Runtime, CPUExecutionProvider)\n")
    print("| Model | Size (MB) | Input | Threads | Inference median (ms) | p95 (ms) | End-to-end detect (ms) | img/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in vision:
        print(
            f"| `{r['model']}` | {r['size_mb']:.2f} | {r['input']} | {r['threads']} | {r['infer_median_ms']:.1f} | "
            f"{r['infer_p95_ms']:.1f} | {r['e2e_median_ms']:.1f} | {r['throughput_img_s']:.0f} |"
        )

    if args.json:
        args.json.write_text(json.dumps({"env": env, "engine": engine, "vision": vision}, indent=2, default=str))
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
