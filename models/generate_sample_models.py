#!/usr/bin/env python3
"""Generates the sample ONNX models used by tests, configs and benchmarks.

    pip install onnx numpy
    python models/generate_sample_models.py            # small models (committed)
    python models/generate_sample_models.py --large    # also a larger MLP for benchmarking

Every model is a classifier MLP with a dynamic batch dimension:

    input  "input"          float32 [N, in_dim]
    output "probabilities"  float32 [N, classes]   (softmax)
    output "logits"         float32 [N, classes]

v1 and v2 share the architecture but use different weights, which is what an
A/B test between two training runs looks like.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))


def build_mlp(name: str, in_dim: int, hidden: list, classes: int, seed: int) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    prev, prev_dim = "input", in_dim
    for i, h in enumerate(hidden):
        w = (rng.standard_normal((prev_dim, h)) * np.sqrt(2.0 / prev_dim)).astype(np.float32)
        b = np.zeros(h, dtype=np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes.append(helper.make_node("Gemm", [prev, f"w{i}", f"b{i}"], [f"h{i}"], name=f"fc{i}"))
        nodes.append(helper.make_node("Relu", [f"h{i}"], [f"a{i}"], name=f"relu{i}"))
        prev, prev_dim = f"a{i}", h
    w = (rng.standard_normal((prev_dim, classes)) * np.sqrt(1.0 / prev_dim)).astype(np.float32)
    b = (rng.standard_normal(classes) * 0.01).astype(np.float32)
    inits += [numpy_helper.from_array(w, "w_out"), numpy_helper.from_array(b, "b_out")]
    nodes.append(helper.make_node("Gemm", [prev, "w_out", "b_out"], ["logits"], name="fc_out"))
    nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1, name="softmax"))

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_dim])],
        [
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", classes]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", classes]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="mie-sample", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # readable by ONNX Runtime >= 1.15
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="also build a ~5 MB MLP for benchmarking")
    parser.add_argument("--out", default=os.path.join(HERE, "sample"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = {
        "mlp_v1.onnx": build_mlp("mlp_v1", 16, [64, 64], 4, seed=1),
        "mlp_v2.onnx": build_mlp("mlp_v2", 16, [64, 64], 4, seed=2),
    }
    if args.large:
        models["mlp_large.onnx"] = build_mlp("mlp_large", 256, [1024, 1024], 10, seed=3)

    for filename, model in models.items():
        path = os.path.join(args.out, filename)
        onnx.save(model, path)
        print(f"wrote {path} ({os.path.getsize(path) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
