"""Validate the schema and sanity of one edge_bench result file."""

import json
import math
import sys


def validate(r: dict) -> list:
    errors = []
    for key in ("label", "threads", "iters", "latency_ms", "throughput_ips", "peak_rss_mb", "output", "system"):
        if key not in r:
            errors.append(f"missing key: {key}")
    if errors:
        return errors
    lat = r["latency_ms"]
    if not all(
        isinstance(lat[k], (int, float)) and math.isfinite(lat[k]) and lat[k] >= 0
        for k in ("mean", "min", "p50", "p95", "p99", "max")
    ):
        errors.append("latency values must be finite and non-negative")
    elif not (lat["min"] <= lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]):
        errors.append("latency percentiles are not monotonic")
    if not r["throughput_ips"] or r["throughput_ips"] <= 0:
        errors.append("throughput must be positive")
    if r["output"].get("all_finite") is not True:
        errors.append("model produced non-finite outputs")
    if r["system"].get("arch") not in ("aarch64", "x86_64"):
        errors.append("unrecognised architecture")
    return errors


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        problems = validate(json.load(f))
    if problems:
        print("INVALID:", *problems, sep="\n  ")
        sys.exit(1)
    print("result OK")
