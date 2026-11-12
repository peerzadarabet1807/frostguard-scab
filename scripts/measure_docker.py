"""Peak memory and latency of the API image on Render's free-plan limits, with and without the input check.

Builds the API image, then runs it twice with ``--memory=512m --cpus=0.1`` and the Render settings
(WEB_CONCURRENCY=1, FROSTGUARD_ORT_THREADS=1): once as shipped (detector + input check) and once with the input
check disabled (FROSTGUARD_GATE_PATH pointing nowhere), for comparison. Each run waits for /health, posts the five
sample leaves, three OOD photos and one OOD photo with skip_input_check=true, and records each request's latency.
Peak memory is the container's cgroup ``memory.peak`` (falls back to the highest ``docker stats`` reading).
Writes results/input_check/docker_memory.json.

Usage: python scripts/measure_docker.py   (needs Docker; about 5 minutes on a laptop)
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "frostguard-scab-api:input-check"
PORT = 18000
LIMITS = {"memory": "512m", "cpus": "0.1"}
OOD = ROOT / "docs" / "ood" / "set"
PHOTOS = [
    *(("sample", p, "") for p in sorted((ROOT / "data" / "samples").glob("*.jpg"))),
    ("ood", OOD / "documents" / "1920_re_u_berliet_consignes_16.jpg", ""),
    ("ood", OOD / "sky" / "blue_and_orange_clouds_over_the_mekong_with_a_pirogue_runnin.jpg", ""),
    ("ood", OOD / "pear_leaves" / "eriophyes_pyri_129.jpg", ""),
    ("ood, skip_input_check", OOD / "hands" / "hand_comparison.jpg", "?skip_input_check=true"),
]


def sh(*args: str, check: bool = True) -> str:
    return subprocess.run(args, check=check, capture_output=True, text=True).stdout.strip()


def to_mib(text: str) -> float:
    value = text.strip().split("/")[0].strip()
    units = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1024, "MB": 1, "GB": 1024, "B": 1 / 2**20}
    for unit, factor in units.items():
        if value.endswith(unit):
            return float(value[: -len(unit)]) * factor
    return float(value) / 2**20


def post(path: Path, query: str) -> tuple[int, float, dict]:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/api/v1/detect-lesion{query}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 (localhost)
        data = json.loads(r.read())
        return r.status, (time.perf_counter() - t0) * 1000, data


def run(label: str, extra_env: list[str]) -> dict:
    name = f"frostguard-measure-{uuid.uuid4().hex[:8]}"
    sh(
        "docker", "run", "-d", "--rm", "--name", name, f"--memory={LIMITS['memory']}", f"--cpus={LIMITS['cpus']}",
        "-p", f"{PORT}:8000", "-e", "WEB_CONCURRENCY=1", "-e", "FROSTGUARD_ORT_THREADS=1", "-e", "FROSTGUARD_OFFLINE=1",
        *extra_env, IMAGE,
    )
    samples: list[float] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            out = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name, check=False)
            if out:
                samples.append(to_mib(out))
            time.sleep(0.5)

    threading.Thread(target=poll, daemon=True).start()
    try:
        t0 = time.perf_counter()
        while True:
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:  # noqa: S310
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.perf_counter() - t0 > 300:
                raise TimeoutError("API did not become healthy")
            time.sleep(1)
        startup_s = time.perf_counter() - t0
        requests = []
        for kind, path, query in PHOTOS:
            status, ms, data = post(path, query)
            check = data.get("input_check") or {}
            requests.append(
                {
                    "photo": path.relative_to(ROOT).as_posix(),
                    "kind": kind,
                    "status": status,
                    "latency_ms": round(ms, 1),
                    "analysed": data.get("analysed"),
                    "input_check_passed": check.get("passed"),
                    "input_check_ms": check.get("inference_ms"),
                }
            )
            print(f"  {label:<14} {kind:<22} {ms:8.0f} ms  passed={check.get('passed')}  analysed={data.get('analysed')}")
        peak_file = sh("docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak", check=False)
        peak = int(peak_file) / 2**20 if peak_file.isdigit() else max(samples)
        return {
            "startup_s": round(startup_s, 1),
            "peak_memory_mib": round(peak, 1),
            "peak_source": "cgroup memory.peak" if peak_file.isdigit() else "docker stats (polled every 0.5 s)",
            "max_polled_mib": round(max(samples), 1) if samples else None,
            "requests": requests,
        }
    finally:
        stop.set()
        sh("docker", "rm", "-f", name, check=False)


def main() -> int:
    print("building", IMAGE)
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, "."], check=True, cwd=ROOT)
    with_check = run("with check", [])
    without = run("without check", ["-e", "FROSTGUARD_GATE_PATH=/nonexistent.onnx"])
    doc = {
        "description": (
            "API container on Render's free-plan limits (scripts/measure_docker.py): detector + input check as "
            "shipped, and with the input check disabled for comparison. Latency includes the HTTP round trip."
        ),
        "limits": LIMITS,
        "env": {"WEB_CONCURRENCY": "1", "FROSTGUARD_ORT_THREADS": "1"},
        "machine": "local Docker Desktop (WSL2); Render's CPUs differ, so latencies are indicative",
        **with_check,
        "without_input_check": without,
    }
    out = ROOT / "results" / "input_check" / "docker_memory.json"
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"peak memory: {with_check['peak_memory_mib']} MiB with the check, {without['peak_memory_mib']} MiB without")
    return 0


if __name__ == "__main__":
    sys.exit(main())
