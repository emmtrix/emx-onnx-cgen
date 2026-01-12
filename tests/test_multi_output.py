from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from onnx2c.compiler import Compiler, CompilerOptions


def test_multi_output_graph_compile_and_run() -> None:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, [2, 2]
    )
    add_output = helper.make_tensor_value_info(
        "out_add", TensorProto.FLOAT, [2, 2]
    )
    mul_output = helper.make_tensor_value_info(
        "out_mul", TensorProto.FLOAT, [2, 2]
    )
    add_node = helper.make_node(
        "Add", inputs=["in0", "in0"], outputs=["out_add"]
    )
    mul_node = helper.make_node(
        "Mul", inputs=["in0", "in0"], outputs=["out_mul"]
    )
    graph = helper.make_graph(
        [add_node, mul_node],
        "multi_output",
        [input_info],
        [add_output, mul_output],
    )
    model = helper.make_model(graph)
    compiler = Compiler(
        CompilerOptions(template_dir=Path("templates"), model_name="multi")
    )
    generated = compiler.compile(model)
    assert "void multi(" in generated
    assert "out_add[2][2]" in generated
    assert "out_mul[2][2]" in generated
    inputs = {"in0": np.ones((2, 2), dtype=np.float32)}
    outputs = compiler.run(model, inputs)
    np.testing.assert_allclose(outputs["out_add"], 2.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(outputs["out_mul"], 1.0, rtol=1e-6, atol=1e-6)
