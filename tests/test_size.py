from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_size_model() -> onnx.ModelProto:
    input_shape = [2, 3, 4]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.INT64, [])
    node = helper.make_node("Size", inputs=["in0"], outputs=[output.name])
    graph = helper.make_graph([node], "size_graph", [input_info], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def test_size_run() -> None:
    model = _make_size_model()
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    outputs = compiler.run(model, {"in0": data})
    expected = np.array(data.size, dtype=np.int64)
    np.testing.assert_array_equal(outputs["out"], expected)
