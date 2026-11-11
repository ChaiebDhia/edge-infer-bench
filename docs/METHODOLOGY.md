# Methodology

What `edge_bench` measures, how, and what it deliberately does not claim.

## What is timed

- Only `Ort::Session::Run()` is inside the timed region. Model load, session creation, input generation and
  output post-processing are excluded.
- `--warmup` untimed runs happen first (allocator arenas, thread-pool spin-up, instruction/data caches).
- Each of `--iters` runs is timed individually with `std::chrono::steady_clock`; the JSON reports mean, stddev, min,
  p50, p95, p99 and max. **p50 is the headline number, p95 shows tail behaviour, and the coefficient of variation
  (stddev / mean) tells you how noisy that row was.**
- Throughput is `iters * batch / wall_time` over the whole timed loop.
- Peak RSS is `getrusage(RUSAGE_SELF).ru_maxrss` for the whole process, so it includes the ONNX Runtime library and
  the model weights, not only activations.

## Controlled variables

| Variable | Setting |
|---|---|
| Execution provider | ONNX Runtime default CPU provider |
| Graph optimisation | `ORT_ENABLE_ALL` |
| Execution mode | sequential; `intra_op` threads = `--threads`, `inter_op` threads = 1 |
| Input | seeded `N(0, 1)` tensor, identical on every run (`--seed`, default 42) |
| Batch | 1 (latency-oriented, the on-device case) |
| Model bytes | exported once in CI and shared, so x64 and arm64 run byte-identical files |
| ONNX Runtime | pinned version, tarball verified against a SHA-256 in `scripts/fetch_ort.sh` |

Every result file also records the architecture, CPU model string, which ML-relevant instruction extensions the
kernel exposes (for example Arm `asimddp` dot-product, `i8mm`, `bf16`, `sve`; x86 `avx512_vnni`, `amx_int8`),
the ONNX Runtime version and the compiler. A latency number is only meaningful next to that context.

## Why the regression gate uses a ratio

CI runs on shared cloud VMs. Absolute latency moves between runs because of neighbouring tenants, frequency
behaviour and scheduling. The gate therefore compares the **INT8/FP32 p50 latency ratio measured inside the same
job**, because noise tends to scale both measurements together. It fails the build if the INT8 speedup falls more
than 25% below the committed per-architecture baseline (`baselines/<arch>.json`, recorded with
`python/compare.py --write-baseline`). Absolute milliseconds are reported but never gated.

The tolerance is deliberately loose. It exists to catch real regressions (a broken quantization path, an accidental
debug build), not to detect small changes.

## Accuracy is measured separately, and only on real data

The CI models use random weights: latency depends on the graph and shapes, not on the weight values, so no dataset is
needed. That also means **CI says nothing about accuracy**. The accuracy cost of INT8 is measured with
`python/eval_accuracy.py` on a labelled image folder, using a model you trained and calibrated on real images
(`--calib-dir`). Random-tensor calibration is only acceptable for the latency benchmark.

## What this project does not claim

- It does not claim any speedup before a CI run has measured one; numbers belong to a specific run and machine.
- It does not claim which low-level kernels ONNX Runtime selected on a given CPU (for example whether a dot-product
  or matrix-multiply path was used). It records which extensions the CPU *exposes*, not which ones the runtime *used*.
  Confirming that needs profiling (`perf`) or runtime logs, which is future work.
- One model family (EfficientNet-B3), one runtime (ONNX Runtime CPU provider), batch size 1. Conclusions do not
  automatically transfer to other models, runtimes or batch sizes.
- Cloud VMs are not representative of phones or embedded boards.
