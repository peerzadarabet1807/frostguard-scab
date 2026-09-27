# Benchmarks

Every number here was **measured**, not estimated, with [`scripts/benchmark.py`](scripts/benchmark.py) on a
modest 2019 laptop CPU. Re-run the script on your hardware; it prints these same tables.

## Targets vs. measured

| Target | Goal | Measured | Verdict |
|---|---|---|---|
| Risk-engine throughput | > 10,000 hours/s | **316,000–484,000 hours/s** (5-year hourly series) | ✅ 30–48× headroom |
| Detector footprint | < 12 MB | **10.60 MB** trained YOLO11n ONNX · 0.08 MB fallback graph | ✅ |
| Detector latency (CPU) | < 15 ms | **59.6 ms** trained model @ 640 px · 13.5 ms @ 320 px (same architecture) | ❌ at 640 px · ✅ at 320 px on this CPU |

**Latency, plainly:** on this 15 W laptop chip, the < 15 ms goal is met only at 320 px input, not at the
640 px the model was trained and exported at. See [Reaching < 15 ms](#reaching--15-ms) for the options.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-10710U @ 1.10 GHz (6 cores / 12 threads, 15 W ultrabook part, 2019) |
| RAM | 16 GB |
| OS | Windows 11 Pro |
| Runtime | Python 3.11.9 · ONNX Runtime 1.30.0 (CPUExecutionProvider, `ORT_ENABLE_ALL`) · NumPy 2.3.5 |

Laptop timings move ±20 % with power and thermal state. The two runs below are from different days on the
same machine and are reported as measured.

## 1. Risk engine (Revised Mills state machine)

| Workload | Run 1 (25 Sep) | Run 2 (27 Sep) |
|---|---|---|
| 5-year hourly series (43,800 h, synthetic) | 484,287 hours/s | 316,016 hours/s |
| Shopian spring 2024, real ERA5 (1,080 h) | 90,870 hours/s | 75,039 hours/s |
| `assess()` for one API request (72 h window → 48 h horizon) | 11.2 ms median | 13.4 ms median · 17.2 ms p95 |

The hour loop is a tight state machine over NumPy arrays: leaf wetness and the Mills rate `1/H(T)` are
vectorised, and only the wet/dry/reset bookkeeping runs per hour. On short windows the fixed cost of input
validation and hourly regularisation in pandas dominates. That's why the 1,080 h series shows lower hours/s
than the 5-year one.

## 2. Vision (ONNX Runtime, CPU)

Median over 100 inference runs after 10 warm-up runs. "End-to-end" is `ScabVisionDetector.detect()` on a
sample leaf: decode + letterbox + inference + box decode + NMS.

### Trained model (release `model-v1`), run 2

| Model | Size (MB) | Input | Threads | Inference median (ms) | p95 (ms) | End-to-end (ms) | img/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scab_detector.onnx` (YOLO11n, 1 class) | 10.60 | 640 | auto | **59.6** | 65.7 | 92.1 | 17 |
| `scab_detector.onnx` (YOLO11n, 1 class) | 10.60 | 640 | 1 | 121.1 | 147.0 | 141.3 | 8 |
| `scab_detector_fallback.onnx` | 0.08 | 640 | auto | 4.6 | 5.0 | 31.2 | 218 |
| `scab_detector_fallback.onnx` | 0.08 | 640 | 1 | 6.7 | 8.5 | 25.1 | 149 |

### Input-size sweep (COCO-pretrained YOLO11n, same architecture), run 1

Before training, the input-size trade-off was measured on `yolo11n.pt`, exported with the notebook's exact call
(`format="onnx", opset=17, simplify=True`). It has the same backbone, neck and head as the scab model; only
the classification channels differ (80 vs 1), which is under 1 % of the 6.4 GFLOPs.

| Input | Size (MB) | Threads | Inference median (ms) | p95 (ms) | img/s |
|---:|---:|---:|---:|---:|---:|
| 640 | 10.74 | auto | 48.9 | 55.9 | 20 |
| 416 | 10.64 | auto | 20.0 | 26.9 | 50 |
| 320 | 10.62 | auto | **13.5** | 16.1 | 74 |
| 320 | 10.62 | 1 | 27.6 | 28.8 | 36 |

**Correctness check.** On that export, FrostGuard's ONNX Runtime pipeline (letterbox → decode → NMS)
reproduced Ultralytics' own PyTorch predictions on its bundled `bus.jpg` and `zidane.jpg`: same detections,
box IoU 0.95–0.995.

**Where end-to-end time goes:** for the trained model, inference is about two-thirds of it. For the fallback,
JPEG decode plus letterboxing (~20 ms) dominates the 5 ms network.

### Hosted latency

The hosted API runs on Render's free plan: **512 MB RAM and 0.1 CPU**. These numbers come from the production
image under the same limits (`docker run --memory 512m --cpus 0.1`):

| Setting | Cold start | Leaf detection (3 runs) | Peak memory |
|---|---:|---|---:|
| ONNX Runtime default threads | 57 s | 8.8 · 8.4 · 6.7 s | 207 MB |
| `FROSTGUARD_ORT_THREADS=1` (used in [`render.yaml`](render.yaml)) | **29 s** | **3.6 · 1.7 · 1.2 s** | 194 MB |

On a tenth of a core, ONNX Runtime's default worker threads spend the CPU quota busy-waiting.
One pinned thread with spinning disabled is about 5× faster. On the deployed Render service, the API reports
**0.55–0.8 s** model inference per leaf once warm (27 Sep 2026); Render's free CPU evidently bursts above the
0.1 CPU simulated here. A 48 h risk assessment takes about 2.5 s at this
CPU share. The free service sleeps after 15 minutes idle; the web app shows a "waking up" state until the
first request succeeds.

## Reaching < 15 ms

In rough order of effort:

1. **Serve at 320 px.** Re-export with `imgsz=320` (13.5 ms measured above) and re-check mAP in the notebook.
   Scab lesions are small, so accuracy has to be weighed on the validation split.
2. **OpenVINO Execution Provider** (Intel CPUs): `pip install onnxruntime-openvino`, then pass
   `providers=["OpenVINOExecutionProvider"]` to `ScabVisionDetector`. No model change needed.
3. **INT8 static quantisation** (QDQ via `onnxruntime.quantization.quantize_static`, calibrated on about
   100 validation images). Typically 2–3× faster on AVX-512 VNNI / AMX CPUs; re-validate mAP afterwards.
4. **Desktop or server CPU.** These runs used a 1.1 GHz-base, 15 W mobile part.

## Reproduce

```bash
python scripts/download_model.py                                   # trained model from the release
python scripts/benchmark.py --onnx models/scab_detector.onnx --threads 0 1 --json bench.json
```
