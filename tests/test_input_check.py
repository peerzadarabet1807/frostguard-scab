"""The input check: its rule, ONNX-vs-PyTorch parity, and how /api/v1/detect-lesion uses it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from collections.abc import Iterator

from fastapi.testclient import TestClient

from src.api.app import app
from src.vision.input_check import DEFAULT_CONFIG_PATH, InputChecker, prepare

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests" / "fixtures" / "input_check_reference.json").read_text("utf-8"))
SUMMARY = json.loads((ROOT / "results" / "input_check" / "summary.json").read_text("utf-8"))
CONFIG = json.loads(DEFAULT_CONFIG_PATH.read_text("utf-8"))
SAMPLE = ROOT / "data" / "samples" / "scab_leaf_4.jpg"
NOT_A_LEAF = ROOT / "docs" / "ood" / "set" / "documents" / "1920_re_u_berliet_consignes_16.jpg"


@pytest.fixture(scope="module")
def checker() -> InputChecker:
    return InputChecker()


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    mp = pytest.MonkeyPatch()
    mp.setenv("FROSTGUARD_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    with TestClient(app) as test_client:
        yield test_client
    mp.undo()


# --- the rule ---------------------------------------------------------------------------------------------
def test_config_is_consistent() -> None:
    protos = np.asarray(CONFIG["prototypes"]["vectors"])
    assert protos.shape == (len(CONFIG["prototypes"]["names"]), CONFIG["model"]["embedding_dim"])
    np.testing.assert_allclose(np.linalg.norm(protos, axis=1), 1.0, atol=1e-6)
    assert 0 < CONFIG["threshold"] < 1
    assert CONFIG["threshold"] == SUMMARY["threshold"] == REFERENCE["threshold"]
    assert CONFIG["version"] == SUMMARY["version"] == REFERENCE["version"]
    assert CONFIG["tuning"]["false_rejection"]["rate"] <= 0.01


def test_score_is_cosine_with_nearest_prototype(checker: InputChecker) -> None:
    for p in checker.prototypes:
        assert checker.score(p * 2.5) == pytest.approx(1.0, abs=1e-6)
    q, _ = np.linalg.qr(checker.prototypes.T)
    v = np.random.default_rng(0).normal(size=checker.prototypes.shape[1])
    v -= q @ (q.T @ v)
    assert checker.score(v) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [np.zeros(512), np.full(512, np.nan), np.ones(7)])
def test_rejects_malformed_embeddings(checker: InputChecker, bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        checker.score(bad)


def test_prepare_is_224_rgb() -> None:
    x = prepare(SAMPLE)
    assert x.shape == (224, 224, 3) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 255


# --- parity: the shipped ONNX file against the original PyTorch model -------------------------------------
def test_onnx_matches_pytorch_reference(checker: InputChecker) -> None:
    rows = REFERENCE["images"]
    emb = checker.embed(np.stack([prepare(ROOT / r["file"]) for r in rows]))
    ref = np.asarray([r["embedding"] for r in rows])
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (emb * ref).sum(axis=1)
    assert cos.min() >= 0.999, dict(zip([r["file"] for r in rows], cos.round(5), strict=True))
    decisions = [checker.score(e) >= checker.threshold for e in emb]
    assert decisions == [r["passed"] for r in rows]


def test_shipped_sample_photos_pass(checker: InputChecker) -> None:
    for path in sorted((ROOT / "data" / "samples").glob("*.jpg")):
        assert checker.check(path).passed, path.name


# --- the API ----------------------------------------------------------------------------------------------
def _post(client: TestClient, path: Path, query: str = "") -> dict:
    res = client.post(f"/api/v1/detect-lesion{query}", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert res.status_code == 200, res.text
    return res.json()


def test_api_checks_a_leaf_then_detects(client: TestClient) -> None:
    body = _post(client, SAMPLE)
    check = body["input_check"]
    assert check["passed"] is True and check["skipped"] is False
    assert check["threshold"] == CONFIG["threshold"] and check["version"] == CONFIG["version"]
    assert check["score"] >= check["threshold"]
    assert body["analysed"] is True and isinstance(body["scab_detected"], bool)


def test_api_turns_away_a_photo_that_is_not_a_leaf(client: TestClient) -> None:
    body = _post(client, NOT_A_LEAF)
    assert body["input_check"]["passed"] is False
    assert body["analysed"] is False
    assert body["scab_detected"] is None  # never "no scab" for a photo that was not analysed
    assert body["detections"] == [] and body["lesion_count"] == 0
    assert any("doesn't look like an apple leaf" in w for w in body["warnings"])


def test_api_skip_input_check_runs_the_detector_and_says_so(client: TestClient) -> None:
    body = _post(client, NOT_A_LEAF, "?skip_input_check=true")
    assert body["analysed"] is True and isinstance(body["scab_detected"], bool)
    assert body["input_check"]["skipped"] is True and body["input_check"]["passed"] is False
    assert any("Unchecked input" in w for w in body["warnings"])


def test_model_info_describes_the_input_check(client: TestClient) -> None:
    info = client.get("/api/v1/model").json()["input_check"]
    assert info["version"] == CONFIG["version"] and info["threshold"] == CONFIG["threshold"]
    assert "false_rejection" in info["measured"] and "false_acceptance" in info["measured"]


def test_api_still_serves_without_the_input_check_model(tmp_path: Path) -> None:
    mp = pytest.MonkeyPatch()
    mp.setenv("FROSTGUARD_GATE_PATH", str(tmp_path / "absent.onnx"))
    mp.setenv("FROSTGUARD_CACHE_DIR", str(tmp_path / "cache"))
    try:
        with TestClient(app) as c:
            body = _post(c, SAMPLE)
            assert body["analysed"] is True and body["input_check"] is None
            assert any("Input check unavailable" in w for w in body["warnings"])
            assert c.get("/api/v1/model").json()["input_check"] is None
    finally:
        mp.undo()


# --- memory and latency on Render's free plan (512 MB, 0.1 CPU), in Docker; opt-in ---------------------------
@pytest.mark.skipif(os.getenv("FROSTGUARD_DOCKER_TESTS") != "1", reason="set FROSTGUARD_DOCKER_TESTS=1 (needs Docker)")
def test_api_fits_in_512_mb_with_the_input_check() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "measure_docker.py")], check=True, cwd=ROOT)
    result = json.loads((ROOT / "results" / "input_check" / "docker_memory.json").read_text("utf-8"))
    assert result["limits"] == {"memory": "512m", "cpus": "0.1"}
    assert result["peak_memory_mib"] < 448  # leave at least 64 MiB of the 512 MiB limit
    assert all(r["status"] == 200 for r in result["requests"])
