from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from emx_onnx_cgen.onnx_import import import_onnx


def _make_constant_model() -> tuple[onnx.ModelProto, np.ndarray]:
    const_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = helper.make_tensor(
        "const_tensor",
        TensorProto.FLOAT,
        dims=const_values.shape,
        vals=const_values.flatten().tolist(),
    )
    node = helper.make_node("Constant", inputs=[], outputs=["const_out"], value=tensor)
    output = helper.make_tensor_value_info(
        "const_out", TensorProto.FLOAT, const_values.shape
    )
    graph = helper.make_graph([node], "const_graph", [], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model, const_values


def test_import_constant_creates_initializer() -> None:
    model, const_values = _make_constant_model()
    graph = import_onnx(model)
    assert not graph.nodes
    assert len(graph.initializers) == 1
    initializer = graph.initializers[0]
    assert initializer.name == "const_out"
    np.testing.assert_array_equal(initializer.data, const_values)
