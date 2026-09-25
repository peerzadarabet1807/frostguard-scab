"""Offline fallback detector compiled directly to an ONNX graph.

Until the YOLO11n model trained on Google Colab is copied to ``models/scab_detector.onnx``,
the API and dashboard use this ~80 KB graph instead. It speaks the exact same I/O contract
as an Ultralytics YOLO export — ``images: float32[1, 3, 640, 640]`` in, ``output0:
float32[1, 4 + nc, N]`` (``cx, cy, w, h, class scores``) out — so the whole
pre/post-processing and NMS pipeline is exercised end-to-end.

What it computes is a classical colour heuristic, not a learned model:

1. **Lesion mask** — scab lesions are dark olive/brown, i.e. red >= green and low
   brightness, while healthy tissue is green-dominant and backgrounds are bright::

       mask = clip(60 * relu(R - G + 0.08) * relu(0.42 - mean(R, G, B)), 0, 1)

2. **Multi-scale density** — average-pool the mask with 16/32/64 px windows on an
   8 px grid (80 x 80 cells).
3. **Peak picking** — the sum of the three densities forms a pyramid-shaped kernel
   that peaks at blob centres; a 5 x 5 max-pool keeps local maxima only.
4. **Box regression** — box side ~ equivalent-circle diameter of the lesion area
   measured in the 64 px window; confidence = ``clip(2 * density16)``.

Regenerate with ``python -m src.vision.fallback_model``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "fallback" / "scab_detector_fallback.onnx"
FALLBACK_CLASS_NAMES: dict[int, str] = {0: "apple_scab"}
INPUT_SIZE = 640
GRID_STRIDE = 8
PREPOOL = 4  # mask is average-pooled 4x before the multi-scale windows (exact, ~20x cheaper)
OPSET = 17


def build_fallback_model(input_size: int = INPUT_SIZE) -> "onnx.ModelProto":  # type: ignore[name-defined]  # noqa: F821
    """Construct the colour-heuristic detector as an ONNX ``ModelProto``."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if input_size % GRID_STRIDE:
        raise ValueError(f"input_size must be a multiple of {GRID_STRIDE}")
    grid = input_size // GRID_STRIDE
    n_cells = grid * grid

    def const(name: str, value: np.ndarray | float) -> onnx.TensorProto:
        return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)

    # Cell centres of a 16 px window with pad 4 on an 8 px stride: 8*i + 4.
    centres = (np.arange(grid, dtype=np.float32) * GRID_STRIDE + GRID_STRIDE / 2).astype(np.float32)
    cx = np.broadcast_to(centres[None, :], (grid, grid)).reshape(1, 1, grid, grid)
    cy = np.broadcast_to(centres[:, None], (grid, grid)).reshape(1, 1, grid, grid)
    # Tiny deterministic ramp breaks exact ties on flat plateaus before peak picking.
    tie_break = (np.arange(n_cells, dtype=np.float32) / n_cells * 1e-4).reshape(1, 1, grid, grid)

    initializers = [
        const("feat_w", [[[[1.0]], [[-1.0]], [[0.0]]], [[[-1 / 3]], [[-1 / 3]], [[-1 / 3]]]]),
        const("feat_b", [0.08, 0.42]),
        const("mask_gain", 60.0),
        const("zero", 0.0),
        const("one", 1.0),
        const("two", 2.0),
        const("area_64", 64.0 * 64.0),
        const("diam_gain", 1.128 * 1.25),  # equivalent-circle diameter + 25 % margin
        const("side_min", 12.0),
        const("side_max", 96.0),
        const("cx", cx),
        const("cy", cy),
        const("tie_break", tie_break),
        numpy_helper.from_array(np.array([1, 5, n_cells], dtype=np.int64), name="out_shape"),
    ]

    def avg_pool(src: str, dst: str, kernel: int, stride: int, pad: int) -> onnx.NodeProto:
        return helper.make_node(
            "AveragePool",
            [src],
            [dst],
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
            count_include_pad=1,
        )

    def density(dst: str, window_px: int) -> onnx.NodeProto:
        # A window_px window centred on each 8 px cell, evaluated on the 4x pre-pooled mask.
        # Window edges fall on multiples of 4 px, so this equals pooling the full-res mask.
        pad_px = (window_px - GRID_STRIDE) // 2
        return avg_pool("mask4", dst, window_px // PREPOOL, GRID_STRIDE // PREPOOL, pad_px // PREPOOL)

    nodes = [
        # 1. Lesion mask
        helper.make_node("Conv", ["images", "feat_w", "feat_b"], ["feat"], kernel_shape=[1, 1]),
        helper.make_node("Relu", ["feat"], ["feat_relu"]),
        helper.make_node("ReduceProd", ["feat_relu"], ["feat_prod"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["feat_prod", "mask_gain"], ["mask_raw"]),
        helper.make_node("Clip", ["mask_raw", "zero", "one"], ["mask"]),
        # 2. Multi-scale lesion density on the 8 px grid
        avg_pool("mask", "mask4", PREPOOL, PREPOOL, 0),
        density("dens16", 16),
        density("dens32", 32),
        density("dens64", 64),
        # 3. Peak picking
        helper.make_node("Sum", ["dens16", "dens32", "dens64", "tie_break"], ["pyramid"]),
        helper.make_node("MaxPool", ["pyramid"], ["local_max"], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]),
        helper.make_node("Equal", ["pyramid", "local_max"], ["is_peak_bool"]),
        helper.make_node("Cast", ["is_peak_bool"], ["is_peak"], to=TensorProto.FLOAT),
        # 4. Confidence and box size
        helper.make_node("Mul", ["dens16", "two"], ["conf_raw"]),
        helper.make_node("Clip", ["conf_raw", "zero", "one"], ["conf_clip"]),
        helper.make_node("Mul", ["conf_clip", "is_peak"], ["conf"]),
        helper.make_node("Mul", ["dens64", "area_64"], ["area"]),
        helper.make_node("Sqrt", ["area"], ["area_sqrt"]),
        helper.make_node("Mul", ["area_sqrt", "diam_gain"], ["side_raw"]),
        helper.make_node("Clip", ["side_raw", "side_min", "side_max"], ["side"]),
        # 5. YOLO-style output: [cx, cy, w, h, score] x N
        helper.make_node("Concat", ["cx", "cy", "side", "side", "conf"], ["stacked"], axis=1),
        helper.make_node("Reshape", ["stacked", "out_shape"], ["output0"]),
    ]

    graph = helper.make_graph(
        nodes,
        "frostguard_fallback_scab_detector",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, input_size, input_size])],
        outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, n_cells])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="frostguard-scab",
        doc_string="Colour-heuristic fallback apple scab detector (YOLO I/O contract).",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = 8  # readable by onnxruntime >= 1.14
    helper.set_model_props(
        model,
        {
            "names": str(FALLBACK_CLASS_NAMES),
            "imgsz": str([input_size, input_size]),
            "stride": str(GRID_STRIDE),
            "task": "detect",
            "fallback": "true",
            "description": "FrostGuard colour-heuristic fallback detector",
        },
    )
    onnx.checker.check_model(model)
    return model


def ensure_fallback_model(path: str | Path = FALLBACK_MODEL_PATH, overwrite: bool = False) -> Path:
    """Write the fallback graph to ``path`` if it is missing (or ``overwrite``)."""
    import onnx

    target = Path(path)
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_fallback_model(), str(target))
    logger.info("Generated fallback ONNX detector at %s", target)
    return target


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    out = ensure_fallback_model(overwrite=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
