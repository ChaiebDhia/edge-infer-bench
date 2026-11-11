"""Accuracy cost of quantization on a labelled image folder.

Layout: <root>/<class_name>/*.jpg. Classes are indexed by sorted folder name
(the torchvision ImageFolder convention), so the class order must match how
the model was trained. Reports top-1 for each model and how often the INT8
prediction agrees with the FP32 one.
"""

import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort

from quantize_int8 import load_image


def predict(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return session.run(None, {session.get_inputs()[0].name: x})[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args()

    root = pathlib.Path(args.data)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class folders under {root}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = [(p, i) for i, c in enumerate(classes) for p in sorted((root / c).iterdir()) if p.suffix.lower() in exts]
    if not samples:
        raise SystemExit("no images found")

    s32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])
    ok32 = ok8 = agree = 0
    cos = []
    for path, label in samples:
        x = load_image(path, args.size)
        a, b = predict(s32, x), predict(s8, x)
        ok32 += int(a.argmax() == label)
        ok8 += int(b.argmax() == label)
        agree += int(a.argmax() == b.argmax())
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))

    n = len(samples)
    result = {
        "images": n,
        "classes": len(classes),
        "top1_fp32": ok32 / n,
        "top1_int8": ok8 / n,
        "top1_drop_points": 100.0 * (ok32 - ok8) / n,
        "fp32_int8_agreement": agree / n,
        "mean_logit_cosine": float(np.mean(cos)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
