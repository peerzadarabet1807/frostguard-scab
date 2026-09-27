"""Deploy the FrostGuard API (FastAPI + trained model) to a Hugging Face Docker Space.

The Space is built from the repository's own ``Dockerfile``; this script stages only what
that image needs, writes the Space card, sets the CORS allow-list and uploads everything.

Usage::

    hf auth login                                   # once, with a *write* token
    python scripts/deploy_hf_space.py               # -> <your-hf-user>/frostguard-scab
    python scripts/deploy_hf_space.py --space someone/other-name --no-wait

Environment: ``HF_TOKEN`` overrides the cached login and ``HF_SPACE`` the target Space
(both used by the GitHub Action).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "https://github.com/peerzadarabet1807/frostguard-scab"
PAGES_ORIGIN = "https://peerzadarabet1807.github.io"
CORS_ORIGINS = ",".join(
    [PAGES_ORIGIN, "http://localhost:5173", "http://localhost:4173", "http://localhost:8080", "http://127.0.0.1:5173"]
)

# Files and folders the API image needs (see Dockerfile).
STAGE = [
    "Dockerfile",
    ".dockerignore",
    "requirements-runtime.txt",
    "src",
    "data",
    "models/fallback",
    "models/model_card.json",
    "scripts/download_model.py",
    "LICENSE",
]

SPACE_CARD = """---
title: FrostGuard-Scab API
emoji: 🍎
colorFrom: green
colorTo: red
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: Apple scab & frost early warning API for Kashmir orchards
---

# FrostGuard-Scab API

The backend for [FrostGuard-Scab]({github}): Revised Mills apple-scab infection-period modelling on
Open-Meteo weather, plus a YOLO11n lesion detector (ONNX Runtime) trained on Google Colab.

* **Web app:** {pages}/frostguard-scab/
* **Interactive API docs:** `/docs` on this Space
* **Source & model card:** {github}

This Space is deployed from the GitHub repository by `scripts/deploy_hf_space.py`; edit the code there.
The trained weights (release `model-v1`) are AGPL-3.0; the code is MIT.
"""


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage(target: Path) -> None:
    for rel in STAGE:
        src = PROJECT_ROOT / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    (target / "README.md").write_text(SPACE_CARD.format(github=GITHUB_REPO, pages=PAGES_ORIGIN), encoding="utf-8")


def wait_until_running(api: HfApi, repo_id: str, timeout_s: int = 1200) -> str:
    deadline, last = time.time() + timeout_s, None
    while time.time() < deadline:
        stage_name = api.get_space_runtime(repo_id).stage
        if stage_name != last:
            print(f"  space stage: {stage_name}", flush=True)
            last = stage_name
        if stage_name in {"RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            return stage_name
        time.sleep(15)
    return f"TIMEOUT ({last})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--space", default=os.getenv("HF_SPACE") or None, help="owner/name (default: $HF_SPACE or <your user>/frostguard-scab)"
    )
    parser.add_argument("--no-wait", action="store_true", help="don't wait for the Space build to finish")
    args = parser.parse_args()

    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    user = api.whoami()["name"]
    repo_id = args.space or f"{user}/frostguard-scab"
    owner, name = repo_id.split("/")
    app_url = f"https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.add_space_variable(repo_id, "FROSTGUARD_CORS_ORIGINS", CORS_ORIGINS)
    api.add_space_variable(repo_id, "WEB_CONCURRENCY", "2")

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Deploy {GITHUB_REPO.rsplit('/', 1)[-1]}@{git_sha()}",
            delete_patterns=["src/**", "data/**", "models/**", "scripts/**"],
        )
    print(f"uploaded to https://huggingface.co/spaces/{repo_id}")
    print(f"API base URL: {app_url}")

    if args.no_wait:
        return 0
    final = wait_until_running(api, repo_id)
    print(f"final stage: {final}")
    return 0 if final == "RUNNING" else 1


if __name__ == "__main__":
    sys.exit(main())
