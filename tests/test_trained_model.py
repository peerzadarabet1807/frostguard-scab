"""Checks against the trained YOLO11n release model (skipped when it is not installed).

Install it with ``python scripts/download_model.py`` (CI does this automatically).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, load_model_card
from src.vision.detector import ScabVisionDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "scab_detector.onnx"
CARD = ROOT / "models" / "model_card.json"
SAMPLES = ROOT / "data" / "samples"

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason="trained model not installed (scripts/download_model.py)")


@pytest.fixture(scope="module")
def detector() -> ScabVisionDetector:
    return ScabVisionDetector(model_path=MODEL)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    mp = pytest.MonkeyPatch()
    mp.setenv("FROSTGUARD_MODEL_PATH", str(MODEL))
    with TestClient(app) as test_client:
        yield test_client
    mp.undo()


def test_loads_as_trained_with_metadata(detector: ScabVisionDetector) -> None:
    assert detector.backend == "trained"
    assert detector.class_names == {0: "Apple - Scab"}
    assert detector.input_size == 640


@pytest.mark.parametrize("name", ["scab_leaf_1.jpg", "scab_leaf_2.jpg", "scab_leaf_3.jpg", "scab_leaf_4.jpg"])
def test_detects_lesions_on_held_out_leaves(detector: ScabVisionDetector, name: str) -> None:
    result = detector.detect(SAMPLES / name)
    assert result.lesion_count >= 1
    assert all(d.label == "Apple - Scab" and 0.25 <= d.confidence <= 1 for d in result.detections)


def test_healthy_leaf_is_clean(detector: ScabVisionDetector) -> None:
    assert detector.detect(SAMPLES / "healthy_leaf_1.jpg").lesion_count == 0


def test_model_card_matches_released_model(detector: ScabVisionDetector) -> None:
    card = load_model_card(detector)
    assert card is not None, "model_card.json sha256 does not match models/scab_detector.onnx"
    assert card["metrics"]["mAP50"] == pytest.approx(0.531)
    assert card["classes"] == ["Apple - Scab"]


def test_mismatched_card_is_not_served(detector: ScabVisionDetector, tmp_path: Path) -> None:
    card = json.loads(CARD.read_text(encoding="utf-8"))
    card["export"]["sha256"] = "0" * 64
    stale = tmp_path / "model_card.json"
    stale.write_text(json.dumps(card), encoding="utf-8")
    assert load_model_card(detector, stale) is None


def test_api_serves_model_card(client: TestClient) -> None:
    body = client.get("/api/v1/model").json()
    assert body["backend"] == "trained"
    assert body["classes"] == ["Apple - Scab"]
    assert body["exported_by"].startswith("Ultralytics")
    assert body["card"]["export"]["release_tag"] == "model-v1"


def test_api_detects_on_sample(client: TestClient) -> None:
    sample = SAMPLES / "scab_leaf_4.jpg"
    body = client.post("/api/v1/detect-lesion", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")}).json()
    assert body["backend"] == "trained" and body["scab_detected"] is True
    assert body["warnings"] == []
