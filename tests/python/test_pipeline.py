import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

import check_result  # noqa: E402
import compare  # noqa: E402


def make_result(label: str, threads: int, p50: float, arch: str = "aarch64") -> dict:
    return {
        "label": label,
        "model": f"{label}.onnx",
        "model_bytes": 1000 if label == "int8" else 4000,
        "threads": threads,
        "batch": 1,
        "iters": 100,
        "warmup": 10,
        "latency_ms": {
            "mean": p50,
            "stddev": p50 * 0.02,
            "min": p50 * 0.9,
            "p50": p50,
            "p95": p50 * 1.1,
            "p99": p50 * 1.2,
            "max": p50 * 1.3,
        },
        "throughput_ips": 1000.0 / p50,
        "peak_rss_mb": 100.0,
        "output": {"top1": 1, "checksum": 1.0, "classes": 10, "all_finite": True},
        "system": {
            "arch": arch,
            "cpu_model": "test",
            "features": ["asimd"],
            "ort_version": "1.22.0",
            "compiler": "gcc",
        },
    }


def write_results(root: pathlib.Path, fp32_p50: float, int8_p50: float, arch: str = "aarch64") -> None:
    d = root / arch
    d.mkdir(parents=True)
    for t in (1, 2):
        for label, p50 in (("fp32", fp32_p50), ("int8", int8_p50)):
            (d / f"{label}_t{t}.json").write_text(json.dumps(make_result(label, t, p50, arch)))


def test_valid_result_passes():
    assert check_result.validate(make_result("fp32", 1, 10.0)) == []


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda r: r.pop("latency_ms"), "missing key"),
        (lambda r: r["output"].update(all_finite=False), "non-finite"),
        (lambda r: r["latency_ms"].update(p95=0.1), "monotonic"),
        (lambda r: r.update(throughput_ips=0), "throughput"),
        (lambda r: r["system"].update(arch="mips"), "architecture"),
    ],
)
def test_invalid_results_are_rejected(mutate, fragment):
    r = copy.deepcopy(make_result("fp32", 1, 10.0))
    mutate(r)
    assert any(fragment in e for e in check_result.validate(r))


def test_speedup_is_fp32_over_int8_latency(tmp_path):
    write_results(tmp_path, fp32_p50=20.0, int8_p50=10.0)
    runs = compare.load(tmp_path)["aarch64"]
    assert compare.speedups(runs) == {1: 2.0, 2: 2.0}


def test_gate_passes_within_tolerance_and_fails_on_regression(tmp_path, monkeypatch, capsys):
    results, baselines = tmp_path / "results", tmp_path / "baselines"
    write_results(results, fp32_p50=20.0, int8_p50=10.0)  # 2.0x
    baselines.mkdir()
    (baselines / "aarch64.json").write_text(json.dumps({"int8_speedup": {"1": 2.2, "2": 2.2}}))
    argv = ["compare.py", str(results), "--baseline-dir", str(baselines), "--tolerance", "0.25"]
    monkeypatch.setattr(sys, "argv", argv)
    assert compare.main() == 0  # 2.0 >= 2.2 * 0.75

    (baselines / "aarch64.json").write_text(json.dumps({"int8_speedup": {"1": 3.0, "2": 3.0}}))
    assert compare.main() == 1  # 2.0 < 3.0 * 0.75
    assert "FAIL aarch64" in capsys.readouterr().out


def test_missing_baseline_skips_gate(tmp_path, monkeypatch):
    write_results(tmp_path / "r", 20.0, 10.0)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(tmp_path / "r"), "--baseline-dir", str(tmp_path / "none")])
    assert compare.main() == 0


def test_write_baseline_roundtrip(tmp_path, monkeypatch):
    write_results(tmp_path / "r", 30.0, 10.0)
    base = tmp_path / "b"
    monkeypatch.setattr(
        sys, "argv", ["compare.py", str(tmp_path / "r"), "--baseline-dir", str(base), "--write-baseline"]
    )
    assert compare.main() == 0
    assert json.loads((base / "aarch64.json").read_text())["int8_speedup"] == {"1": 3.0, "2": 3.0}


def test_cross_architecture_table_when_two_arches(tmp_path):
    write_results(tmp_path, 20.0, 10.0, "aarch64")
    write_results(tmp_path, 20.0, 15.0, "x86_64")
    text = compare.report(compare.load(tmp_path))
    assert "Cross-architecture" in text and "2.00x" in text and "1.33x" in text
