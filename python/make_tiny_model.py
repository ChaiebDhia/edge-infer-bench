"""Build a tiny conv-net ONNX model with only the `onnx` package (no torch).

Used by the CTest end-to-end test so the C++ harness is exercised on every
machine, including the Arm runners, in well under a second.
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(size: int = 32, classes: int = 10) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.2, "w1")
    b1 = numpy_helper.from_array(np.zeros(8, np.float32), "b1")
    wf = numpy_helper.from_array(rng.standard_normal((8, classes)).astype(np.float32) * 0.2, "wf")
    bf = numpy_helper.from_array(np.zeros(classes, np.float32), "bf")
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("GlobalAveragePool", ["r1"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "wf", "bf"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", classes])],
        initializer=[w1, b1, wf, bf],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # keep compatible with the pinned ONNX Runtime
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    onnx.save(build(), ap.parse_args().out)
