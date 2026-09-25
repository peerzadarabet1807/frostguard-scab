# Benchmarks

All numbers below were **measured**, not estimated, with [`scripts/benchmark.py`](scripts/benchmark.py) on a
modest laptop CPU. Re-run the script on your hardware; it prints these same tables.

## Targets vs. measured

| Target | Goal | Measured | Verdict |
|---|---|---|---|
| Risk-engine throughput | > 10,000 hours/s | **484,287 hours/s** (5-year hourly series) | ✅ 48× headroom |
| Detector footprint | < 12 MB | **10.62–10.74 MB** YOLO11n FP32 ONNX · 0.08 MB fallback graph | ✅ |
| Detector latency (CPU) | < 15 ms | **13.5 ms** @ 320 px · 20.0 ms @ 416 px · 48.9 ms @ 640 px | ✅ at 320 px · ❌ at 640 px on this CPU |

**Latency, plainly:** on this 15 W laptop chip, the < 15 ms goal is met at 320 px input but not at the
notebook's default 640 px. See [Reaching < 15 ms at 640 px](#reaching--15-ms-at-640-px) for the options.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-10710U @ 1.10 GHz (6 cores / 12 threads, 15 W ultrabook part, 2019) |
| RAM | 16 GB |
| OS | Windows 11 Pro |
| Runtime | Python 3.11.9 · ONNX Runtime 1.30.0 (CPUExecutionProvider, `ORT_ENABLE_ALL`) · NumPy 2.3.5 |

## 1. Risk engine (Revised Mills state machine)

| Workload | Result |
|---|---|
| 5-year hourly series (43,800 h, synthetic) | **484,287 hours/s** (90 ms per pass) |
| Shopian spring 2024, real ERA5 (1,080 h) | 90,870 hours/s |
| `assess()` for one API request (72 h window → 48 h horizon) | 11.2 ms median · 12.9 ms p95 |

The hour loop is a tight state machine over NumPy arrays: leaf wetness and the Mills rate `1/H(T)` are
vectorised, and only the wet/dry/reset bookkeeping runs per hour. On short windows the fixed cost of input
validation and hourly regularisation in pandas dominates. That is why the 1,080 h series shows lower
hours/s than the 5-year one.

## 2. Vision (ONNX Runtime, CPU)

Median over 100 inference runs after 10 warm-up runs. "End-to-end" is `ScabVisionDetector.detect()` on
[`data/sample_leaf_scab.jpg`](data/sample_leaf_scab.jpg) (800×600 JPEG): decode + letterbox + inference +
decode + NMS.

| Model | Size (MB) | Input | Threads | Inference median (ms) | p95 (ms) | End-to-end (ms) | img/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scab_detector_fallback.onnx` | 0.08 | 640 | auto | 3.8 | 4.2 | 26.3 | 262 |
| `scab_detector_fallback.onnx` | 0.08 | 640 | 1 | 6.3 | 6.7 | 20.9 | 159 |
| YOLO11n | 10.74 | 640 | auto | 48.9 | 55.9 | 72.7 | 20 |
| YOLO11n | 10.74 | 640 | 1 | 108.5 | 121.7 | 126.0 | 9 |
| YOLO11n | 10.64 | 416 | auto | 20.0 | 26.9 | 34.6 | 50 |
| YOLO11n | 10.64 | 416 | 1 | 46.2 | 51.4 | 56.6 | 22 |
| YOLO11n | 10.62 | 320 | auto | **13.5** | 16.1 | 24.8 | 74 |
| YOLO11n | 10.62 | 320 | 1 | 27.6 | 28.8 | 36.5 | 36 |

**About the YOLO11n rows.** The laptop has no GPU, so no scab model was trained locally. These rows use the
COCO-pretrained `yolo11n.pt`, exported with the exact call the Colab notebook makes
(`format="onnx", opset=17, simplify=True, dynamic=False`). The scab detector has the same backbone, neck and
head. Only the classification output channels differ (1 class instead of 80), which is well under 1 % of the
network's 6.5 GFLOPs. Latency therefore carries over, and the fine-tuned file comes out slightly *smaller*.

**Correctness check.** On the COCO export, FrostGuard's ONNX Runtime pipeline (letterbox → decode → NMS)
reproduced Ultralytics' own PyTorch predictions on its bundled `bus.jpg` and `zidane.jpg`: same detections and
classes, box IoU 0.95–0.995.

**Where end-to-end time goes (fallback, ~22 ms):** JPEG decode ≈ 4 ms, resize + letterbox + float conversion
≈ 12 ms, inference ≈ 4 ms, NMS ≈ 0.5 ms. Pre-processing, not the network, dominates for the fallback.

## Reaching < 15 ms at 640 px

In rough order of effort:

1. **Serve at 416 or 320 px.** Export with `imgsz=320` and re-check mAP in the notebook. Scab lesions are
   small, so accuracy has to be weighed against speed on your own validation split.
2. **OpenVINO Execution Provider** (Intel CPUs): `pip install onnxruntime-openvino`, then pass
   `providers=["OpenVINOExecutionProvider"]` to `ScabVisionDetector`. No model change needed.
3. **INT8 static quantisation** (QDQ via `onnxruntime.quantization.quantize_static`, calibrated on about
   100 validation images). Typically 2–3× faster on AVX-512 VNNI / AMX CPUs; re-validate mAP afterwards.
4. **Desktop or server CPU.** This benchmark ran on a 1.1 GHz-base, 15 W mobile part.

## Reproduce

```bash
# Engine + fallback detector
python scripts/benchmark.py

# Add the trained model (and/or other exports), single- and multi-threaded, raw JSON out
python scripts/benchmark.py --onnx models/scab_detector.onnx --threads 0 1 --json bench.json
```
