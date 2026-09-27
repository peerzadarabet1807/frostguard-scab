"""Tests for the ONNX Runtime scab detector, its fallback graph and post-processing."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import pytest
from PIL import Image

from src.vision.detector import (
    LetterboxInfo,
    ScabVisionDetector,
    box_iou,
    non_max_suppression,
    xywh_to_xyxy,
)
from src.vision.fallback_model import FALLBACK_MODEL_PATH, build_fallback_model, ensure_fallback_model
from src.vision.synthetic import render_healthy_leaf, render_synthetic_leaf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MISSING_MODEL = PROJECT_ROOT / "models" / "__does_not_exist__.onnx"


@pytest.fixture(scope="module")
def detector() -> ScabVisionDetector:
    return ScabVisionDetector(model_path=MISSING_MODEL)


@pytest.fixture(scope="module")
def scab_leaf() -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    return render_synthetic_leaf(seed=7)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class TestPreprocess:
    def test_tensor_shape_dtype_and_range(self, detector: ScabVisionDetector) -> None:
        tensor, _ = detector.preprocess(Image.new("RGB", (800, 600), (255, 0, 0)))
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.flags["C_CONTIGUOUS"]
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0

    def test_letterbox_geometry(self, detector: ScabVisionDetector) -> None:
        tensor, info = detector.preprocess(Image.new("RGB", (800, 600), (255, 0, 0)))
        assert info == LetterboxInfo(scale=0.8, pad_x=0, pad_y=80, orig_width=800, orig_height=600)
        grey = 114 / 255
        assert tensor[0, :, 0, 320] == pytest.approx([grey] * 3)  # top padding
        assert tensor[0, :, 320, 320] == pytest.approx([1.0, 0.0, 0.0])  # image content (RGB order)
        assert tensor[0, :, 639, 320] == pytest.approx([grey] * 3)  # bottom padding

    def test_portrait_image_pads_horizontally(self, detector: ScabVisionDetector) -> None:
        _, info = detector.preprocess(Image.new("RGB", (300, 600)))
        assert info.scale == pytest.approx(640 / 600)
        assert info.pad_y == 0 and info.pad_x == (640 - 320) // 2

    @pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
    def test_colour_modes_are_normalised_to_rgb(self, detector: ScabVisionDetector, mode: str) -> None:
        tensor, _ = detector.preprocess(Image.new(mode, (64, 48)))
        assert tensor.shape == (1, 3, 640, 640)

    def test_accepts_array_bytes_and_path(self, detector: ScabVisionDetector, tmp_path: Path) -> None:
        image = Image.new("RGB", (120, 90), (10, 200, 30))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        path = tmp_path / "leaf.png"
        image.save(path)
        reference, _ = detector.preprocess(image)
        for source in (np.asarray(image), buf.getvalue(), path, str(path)):
            tensor, _ = detector.preprocess(source)
            np.testing.assert_allclose(tensor, reference)

    def test_rejects_unknown_types(self, detector: ScabVisionDetector) -> None:
        with pytest.raises(TypeError):
            detector.preprocess(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Inference & fallback
# ---------------------------------------------------------------------------
class TestInference:
    def test_fallback_backend_is_selected(self, detector: ScabVisionDetector) -> None:
        assert detector.backend == "fallback"
        assert detector.is_fallback
        assert detector.class_names == {0: "apple_scab"}
        assert detector.input_size == 640

    def test_raw_output_follows_yolo_contract(self, detector: ScabVisionDetector, scab_leaf) -> None:  # noqa: ANN001
        tensor, _ = detector.preprocess(scab_leaf[0])
        raw = detector.infer(tensor)
        nc = len(detector.class_names)
        assert raw.shape == (1, 4 + nc, 6400)
        assert raw.dtype == np.float32
        scores = raw[0, 4:]
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_detects_synthetic_lesions(self, detector: ScabVisionDetector, scab_leaf) -> None:  # noqa: ANN001
        image, truth = scab_leaf
        result = detector.detect(image)
        assert result.lesion_count == len(truth)
        pred = np.array([d.box for d in result.detections], dtype=np.float32)
        for gt in truth:
            assert box_iou(np.array(gt, dtype=np.float32), pred).max() >= 0.4
        for det in result.detections:
            assert 0 <= det.x1 < det.x2 <= image.width
            assert 0 <= det.y1 < det.y2 <= image.height
            assert 0.25 <= det.confidence <= 1.0
            assert det.label == "apple_scab"

    def test_healthy_leaf_has_no_detections(self, detector: ScabVisionDetector) -> None:
        assert detector.detect(render_healthy_leaf()).lesion_count == 0

    def test_result_serialises(self, detector: ScabVisionDetector, scab_leaf) -> None:  # noqa: ANN001
        payload = detector.detect(scab_leaf[0]).to_dict()
        assert payload["backend"] == "fallback"
        assert payload["image_width"] == 800 and payload["image_height"] == 600
        assert payload["lesion_count"] == len(payload["detections"])
        assert {"x1", "y1", "x2", "y2", "confidence", "label", "class_id"} <= payload["detections"][0].keys()

    def test_fallback_is_generated_when_missing(self, tmp_path: Path) -> None:
        fallback = tmp_path / "nested" / "fallback.onnx"
        det = ScabVisionDetector(model_path=tmp_path / "none.onnx", fallback_path=fallback)
        assert fallback.is_file()
        assert det.backend == "fallback"
        assert det.model_path == str(fallback)

    def test_trained_model_takes_priority(self, tmp_path: Path) -> None:
        trained = tmp_path / "scab_detector.onnx"
        shutil.copy(ensure_fallback_model(), trained)
        det = ScabVisionDetector(model_path=trained)
        assert det.backend == "trained"
        assert det.class_names == {0: "apple_scab"}  # read from ONNX metadata

    def test_thread_pinning_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Fractional-CPU hosts (e.g. Render free) run one non-spinning ORT thread.
        monkeypatch.setenv("FROSTGUARD_ORT_THREADS", "1")
        det = ScabVisionDetector(model_path=MISSING_MODEL)
        opts = det.session.get_session_options()
        assert opts.intra_op_num_threads == 1
        assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert det.detect(render_healthy_leaf()).lesion_count == 0

    def test_env_var_selects_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        trained = tmp_path / "custom.onnx"
        shutil.copy(ensure_fallback_model(), trained)
        monkeypatch.setenv("FROSTGUARD_MODEL_PATH", str(trained))
        assert ScabVisionDetector().model_path == str(trained)

    def test_committed_fallback_matches_builder(self) -> None:
        assert FALLBACK_MODEL_PATH.is_file(), "run `python -m src.vision.fallback_model`"
        assert onnx.load(str(FALLBACK_MODEL_PATH)).graph == build_fallback_model().graph

    def test_shipped_sample_image(self, detector: ScabVisionDetector) -> None:
        truth = json.loads((PROJECT_ROOT / "data" / "sample_leaf_scab_boxes.json").read_text())["boxes_xyxy"]
        result = detector.detect(PROJECT_ROOT / "data" / "sample_leaf_scab.jpg")
        assert result.lesion_count == len(truth)
        assert detector.detect(PROJECT_ROOT / "data" / "sample_leaf_healthy.jpg").lesion_count == 0


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
class TestPostprocess:
    def test_xywh_to_xyxy(self) -> None:
        out = xywh_to_xyxy(np.array([[50, 40, 20, 10]], dtype=np.float32))
        np.testing.assert_allclose(out, [[40, 35, 60, 45]])

    def test_iou(self) -> None:
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        others = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
        np.testing.assert_allclose(box_iou(box, others), [1.0, 50 / 150, 0.0], rtol=1e-6)

    def test_nms_suppresses_overlaps_keeps_disjoint(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        assert non_max_suppression(boxes, scores, 0.45).tolist() == [0, 2]

    def test_nms_is_class_aware(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 1])
        assert non_max_suppression(boxes, scores, 0.45, classes).tolist() == [0, 1]

    def test_nms_handles_empty_and_max_detections(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        assert non_max_suppression(empty, np.zeros(0, dtype=np.float32)).size == 0
        boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(10)], dtype=np.float32)
        assert len(non_max_suppression(boxes, np.linspace(1, 0.1, 10).astype(np.float32), max_detections=3)) == 3

    def test_decodes_yolo_head_and_undoes_letterbox(self, detector: ScabVisionDetector) -> None:
        # Two overlapping candidates + one below threshold, in 640x640 network space.
        raw = np.zeros((1, 5, 8400), dtype=np.float32)
        raw[0, :, 0] = [320, 320, 64, 32, 0.9]
        raw[0, :, 1] = [322, 321, 64, 32, 0.6]  # duplicate -> NMS
        raw[0, :, 2] = [100, 200, 20, 20, 0.1]  # below conf threshold
        info = LetterboxInfo(scale=0.8, pad_x=0, pad_y=80, orig_width=800, orig_height=600)
        dets = detector.postprocess(raw, info)
        assert len(dets) == 1
        d = dets[0]
        assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((360.0, 280.0, 440.0, 320.0))
        assert d.confidence == pytest.approx(0.9)

    def test_decodes_end_to_end_layout(self, detector: ScabVisionDetector) -> None:
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [100, 100, 200, 180, 0.8, 0]
        raw[0, 1] = [300, 300, 340, 340, 0.2, 0]
        info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, orig_width=640, orig_height=640)
        dets = detector.postprocess(raw, info)
        assert [d.box for d in dets] == [(100.0, 100.0, 200.0, 180.0)]

    def test_boxes_are_clipped_to_image(self, detector: ScabVisionDetector) -> None:
        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, :, 0] = [5, 5, 40, 40, 0.95]
        info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, orig_width=640, orig_height=640)
        (det,) = detector.postprocess(raw, info)
        assert det.x1 == 0.0 and det.y1 == 0.0
