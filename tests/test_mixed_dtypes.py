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


def _make_cast_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 2]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.INT32, [2, 2]
    )
    node = helper.make_node(
        "Cast",
        ["input"],
        ["output"],
        to=TensorProto.INT32,
    )
    graph = helper.make_graph(
        [node],
        "cast_graph",
        [input_info],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=11,
    )


def _make_castlike_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 2]
    )
    like_info = helper.make_tensor_value_info(
        "like", TensorProto.INT32, [2, 2]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.INT32, [2, 2]
    )
    node = helper.make_node(
        "CastLike",
        ["input", "like"],
        ["output"],
    )
    graph = helper.make_graph(
        [node],
        "castlike_graph",
        [input_info, like_info],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19)],
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


def test_cast_matches_numpy() -> None:
    model = _make_cast_model()
    compiler = Compiler()
    input_data = np.array([[1.25, -2.5], [3.0, 4.75]], dtype=np.float32)
    compiled = compiler.run(model, {"input": input_data})
    expected = input_data.astype(np.int32)
    np.testing.assert_array_equal(compiled["output"], expected)


def test_castlike_matches_numpy() -> None:
    model = _make_castlike_model()
    compiler = Compiler()
    input_data = np.array([[1.25, -2.5], [3.0, 4.75]], dtype=np.float32)
    like_data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    compiled = compiler.run(
        model, {"input": input_data, "like": like_data}
    )
    expected = input_data.astype(np.int32)
    np.testing.assert_array_equal(compiled["output"], expected)
