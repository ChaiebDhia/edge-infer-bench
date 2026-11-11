"""Post-training static INT8 quantization (QDQ format) with ONNX Runtime.

Calibration data comes from --calib-dir (an image folder) if given, otherwise
from seeded random tensors. Random calibration is fine for the CI latency
benchmark but NOT for judging accuracy: use real images for that.
"""

import argparse
import pathlib

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_image(path: pathlib.Path, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - IMAGENET_MEAN) / IMAGENET_STD)[None]


class Reader(CalibrationDataReader):
    def __init__(self, input_name: str, size: int, calib_dir, samples: int, seed: int):
        self._name = input_name
        if calib_dir:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = sorted(p for p in pathlib.Path(calib_dir).rglob("*") if p.suffix.lower() in exts)
            if not files:
                raise SystemExit(f"no images found under {calib_dir}")
            rng = np.random.default_rng(seed)
            rng.shuffle(files)
            self._it = iter([load_image(p, size) for p in files[:samples]])
        else:
            rng = np.random.default_rng(seed)
            self._it = iter([rng.standard_normal((1, 3, size, size)).astype(np.float32) for _ in range(samples)])

    def get_next(self):
        batch = next(self._it, None)
        return None if batch is None else {self._name: batch}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, required=True, help="square input resolution")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_name = onnx.load(args.model).graph.input[0].name
    quantize_static(
        args.model,
        args.out,
        Reader(input_name, args.size, args.calib_dir, args.samples, args.seed),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"quantized {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
