"""Run edge_bench for every (model, thread-count) pair and write one JSON per run.

Output layout: <out>/<arch>/<label>_t<threads>.json
"""

import argparse
import os
import pathlib
import platform
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, help="path to the edge_bench binary")
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out) / platform.machine()
    out.mkdir(parents=True, exist_ok=True)
    available = os.cpu_count() or 1
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--model expects LABEL=PATH, got {spec!r}")
        for t in args.threads:
            if t > available:
                print(f"skip {label} threads={t}: only {available} CPUs")
                continue
            dest = out / f"{label}_t{t}.json"
            cmd = [
                args.bench,
                "--model",
                path,
                "--label",
                label,
                "--threads",
                str(t),
                "--iters",
                str(args.iters),
                "--warmup",
                str(args.warmup),
                "--out",
                str(dest),
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
