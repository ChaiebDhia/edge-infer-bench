"""Summarise benchmark results and gate on regressions.

Reads <results>/<arch>/<label>_t<threads>.json (see run_suite.py), prints a
Markdown report, and fails (exit 1) if any result is invalid or if the INT8
speedup over FP32 dropped more than --tolerance below the committed baseline
for that architecture.

The gate uses the INT8/FP32 latency *ratio* measured in the same job, not
absolute milliseconds: shared cloud VMs shift absolute latency from run to run,
but they shift the numerator and denominator together, so the ratio is far more
stable. Absolute numbers are reported but never gated.
"""

import argparse
import json
import pathlib
import sys

from check_result import validate


def load(results_dir: pathlib.Path) -> dict:
    """{arch: {label: {threads: result}}}"""
    data: dict = {}
    for f in sorted(results_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        problems = validate(r)
        if problems:
            raise SystemExit(f"{f}: invalid result: {'; '.join(problems)}")
        data.setdefault(f.parent.name, {}).setdefault(r["label"], {})[r["threads"]] = r
    return data


def speedups(runs: dict) -> dict:
    fp32, int8 = runs.get("fp32", {}), runs.get("int8", {})
    return {t: fp32[t]["latency_ms"]["p50"] / int8[t]["latency_ms"]["p50"] for t in sorted(set(fp32) & set(int8))}


def cv(r: dict) -> float:
    lat = r["latency_ms"]
    return 100.0 * lat["stddev"] / lat["mean"] if lat["mean"] else 0.0


def report(data: dict) -> str:
    lines = ["# Benchmark report", ""]
    for arch, runs in data.items():
        any_run = next(iter(next(iter(runs.values())).values()))
        sysinfo = any_run["system"]
        lines += [
            f"## {arch}",
            "",
            f"- CPU: `{sysinfo['cpu_model'] or 'unknown'}`",
            f"- ML-relevant extensions exposed: `{', '.join(sysinfo['features']) or 'none detected'}`",
            f"- ONNX Runtime {sysinfo['ort_version']}, compiler {sysinfo['compiler']}",
            "",
            "| threads | FP32 p50 (ms) | INT8 p50 (ms) | INT8 speedup | FP32 p95 | INT8 p95 "
            "| FP32 CV% | INT8 CV% | INT8 img/s | INT8 RSS (MB) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        sp = speedups(runs)
        for t, s in sp.items():
            a, b = runs["fp32"][t], runs["int8"][t]
            lines.append(
                f"| {t} | {a['latency_ms']['p50']:.2f} | {b['latency_ms']['p50']:.2f} | {s:.2f}x | "
                f"{a['latency_ms']['p95']:.2f} | {b['latency_ms']['p95']:.2f} | {cv(a):.1f} | {cv(b):.1f} | "
                f"{b['throughput_ips']:.1f} | {b['peak_rss_mb']:.0f} |"
            )
        if "fp32" in runs and "int8" in runs:
            t0 = min(set(runs["fp32"]) & set(runs["int8"]))
            ratio = runs["int8"][t0]["model_bytes"] / runs["fp32"][t0]["model_bytes"]
            lines += ["", f"INT8 model file is {ratio:.0%} of the FP32 size.", ""]
    if len(data) > 1:
        lines += [
            "## Cross-architecture INT8 speedup (same model bytes on every machine)",
            "",
            "| threads | " + " | ".join(data) + " |",
            "|---:|" + "---:|" * len(data),
        ]
        all_t = sorted({t for runs in data.values() for t in speedups(runs)})
        for t in all_t:
            row = [f"{speedups(runs)[t]:.2f}x" if t in speedups(runs) else "-" for runs in data.values()]
            lines.append(f"| {t} | " + " | ".join(row) + " |")
        lines.append("")
    lines += [
        "> Cloud VMs are noisy and share hardware. CV% (stddev/mean) shows how noisy each row was;",
        "> treat single-digit-percent differences between runs as noise. See docs/METHODOLOGY.md.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--baseline-dir", default="baselines")
    ap.add_argument("--tolerance", type=float, default=0.25)
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--summary", default=None, help="also write the report to this file")
    args = ap.parse_args()

    data = load(pathlib.Path(args.results))
    if not data:
        print("no results found", file=sys.stderr)
        return 1
    text = report(data)
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text + "\n")

    base_dir = pathlib.Path(args.baseline_dir)
    failed = False
    for arch, runs in data.items():
        sp = speedups(runs)
        base_file = base_dir / f"{arch}.json"
        if args.write_baseline:
            base_dir.mkdir(parents=True, exist_ok=True)
            base_file.write_text(
                json.dumps({"int8_speedup": {str(t): round(s, 3) for t, s in sp.items()}}, indent=2) + "\n"
            )
            print(f"wrote {base_file}")
            continue
        if not base_file.exists():
            print(f"NOTICE: no baseline for {arch}; gate skipped (record one with --write-baseline)")
            continue
        base = json.loads(base_file.read_text())["int8_speedup"]
        for t, expected in base.items():
            got = sp.get(int(t))
            if got is None:
                print(f"FAIL {arch} threads={t}: no result to compare")
                failed = True
            elif got < expected * (1.0 - args.tolerance):
                print(
                    f"FAIL {arch} threads={t}: INT8 speedup {got:.2f}x "
                    f"< {expected:.2f}x baseline - {args.tolerance:.0%}"
                )
                failed = True
            else:
                print(f"ok   {arch} threads={t}: INT8 speedup {got:.2f}x (baseline {expected:.2f}x)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
