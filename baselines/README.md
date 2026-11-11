# Baselines

`<arch>.json` holds the INT8-over-FP32 speedup per thread count that the CI gate compares against
(`aarch64.json`, `x86_64.json`). They are recorded from a real CI run, never by hand:

1. Download the `results-arm64` and `results-x64` artifacts from a green run into `results/`.
2. `python python/compare.py results --write-baseline`
3. Commit the generated files.

Until a baseline exists for an architecture the gate prints a notice and skips it.
