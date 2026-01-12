from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_mixed_dtype_model() -> onnx.ModelProto:
    float_input = helper.make_tensor_value_info(
        "float_in", TensorProto.FLOAT, [2, 2]
    )
    int_input = helper.make_tensor_value_info("int_in", TensorProto.INT32, [2, 2])
    output = helper.make_tensor_value_info("out", TensorProto.INT32, [2, 2])
    bias = helper.make_tensor(
        "bias", TensorProto.INT32, [2, 2], [1, 2, 3, 4]
    )
    add = helper.make_node("Add", ["int_in", "bias"], ["out"])
    graph = helper.make_graph(
        [add], "mixed_dtype_graph", [float_input, int_input], [output], [bias]
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=11,
    )


def test_compile_supports_mixed_dtypes() -> None:
    model = _make_mixed_dtype_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    assert "const float float_in[restrict 2][2]" in generated
    assert "const int32_t int_in[restrict 2][2]" in generated
    assert "int32_t out[restrict 2][2]" in generated


def test_mixed_dtype_model_matches_onnxruntime() -> None:
    model = _make_mixed_dtype_model()
    compiler = Compiler()
    float_input = np.random.rand(2, 2).astype(np.float32)
    int_input = np.array([[5, -2], [7, 0]], dtype=np.int32)

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = sess.run(None, {"float_in": float_input, "int_in": int_input})

    compiled = compiler.run(
        model, {"float_in": float_input, "int_in": int_input}
    )
    np.testing.assert_array_equal(compiled["out"], ort_out)
