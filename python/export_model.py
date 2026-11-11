"""Export a torchvision classifier to ONNX (FP32).

Weights are random by default: latency depends on the graph and tensor shapes,
not on the values, so CI needs no dataset. To benchmark your own trained model
pass --checkpoint (a state_dict saved with torch.save) and --num-classes.
"""

import argparse

import torch
import torchvision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="efficientnet_b3")
    ap.add_argument("--size", type=int, default=300, help="square input resolution")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--checkpoint", default=None, help="optional state_dict (.pt/.pth)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = getattr(torchvision.models, args.arch)(weights=None, num_classes=args.num_classes)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"exported {args.arch} ({args.size}px, {args.num_classes} classes) -> {args.out}")


if __name__ == "__main__":
    main()
