from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_constant_of_shape_model() -> onnx.ModelProto:
    shape_values = np.array([2, 2], dtype=np.int64)
    shape_tensor = helper.make_tensor(
        "shape",
        TensorProto.INT64,
        dims=shape_values.shape,
        vals=shape_values.tolist(),
    )
    value_tensor = helper.make_tensor(
        "fill",
        TensorProto.FLOAT,
        dims=[1],
        vals=[2.5],
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, shape_values.tolist()
    )
    node = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=[output.name],
        value=value_tensor,
    )
    graph = helper.make_graph(
        [node],
        "constant_of_shape_graph",
        [],
        [output],
        initializer=[shape_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def test_constant_of_shape_run() -> None:
    model = _make_constant_of_shape_model()
    compiler = Compiler()
    outputs = compiler.run(model, {})
    expected = np.full((2, 2), 2.5, dtype=np.float32)
    np.testing.assert_allclose(outputs["out"], expected)
