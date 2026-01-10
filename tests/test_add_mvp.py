from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from onnx2c import Compiler
from golden_utils import assert_golden


def _make_add_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3, 4])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node("Add", inputs=["a", "b"], outputs=["out"])
    graph = helper.make_graph([node], "add_graph", [input_a, input_b], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_mul_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Mul", inputs=["a", "b"], outputs=["out"])
    graph = helper.make_graph([node], "mul_graph", [input_a, input_b], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_mul_add_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
    input_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    mul_node = helper.make_node("Mul", inputs=["a", "b"], outputs=["mul_out"])
    add_node = helper.make_node("Add", inputs=["mul_out", "c"], outputs=["out"])
    graph = helper.make_graph(
        [mul_node, add_node],
        "mul_add_graph",
        [input_a, input_b, input_c],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_tanh_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Tanh", inputs=["x"], outputs=["out"])
    graph = helper.make_graph([node], "tanh_graph", [input_x], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def test_codegen_golden_tanh() -> None:
    model = _make_tanh_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "tanh_model.c"
    assert_golden(generated, golden_path)


def test_tanh_matches_onnxruntime() -> None:
    model = _make_tanh_model()
    compiler = Compiler()
    input_x = np.random.rand(2, 3).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"x": input_x})

    compiled = compiler.run(model, {"x": input_x})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)


def test_codegen_golden_add() -> None:
    model = _make_add_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "add_model.c"
    assert_golden(generated, golden_path)


def test_add_matches_onnxruntime() -> None:
    model = _make_add_model()
    compiler = Compiler()
    input_a = np.random.rand(2, 3, 4).astype(np.float32)
    input_b = np.random.rand(2, 3, 4).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"a": input_a, "b": input_b})

    compiled = compiler.run(model, {"a": input_a, "b": input_b})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)


def test_codegen_golden_mul() -> None:
    model = _make_mul_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "mul_model.c"
    assert_golden(generated, golden_path)


def test_mul_matches_onnxruntime() -> None:
    model = _make_mul_model()
    compiler = Compiler()
    input_a = np.random.rand(2, 3).astype(np.float32)
    input_b = np.random.rand(2, 3).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"a": input_a, "b": input_b})

    compiled = compiler.run(model, {"a": input_a, "b": input_b})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)


def test_codegen_golden_mul_add() -> None:
    model = _make_mul_add_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "mul_add_model.c"
    assert_golden(generated, golden_path)


def test_mul_add_matches_onnxruntime() -> None:
    model = _make_mul_add_model()
    compiler = Compiler()
    input_a = np.random.rand(2, 3).astype(np.float32)
    input_b = np.random.rand(2, 3).astype(np.float32)
    input_c = np.random.rand(2, 3).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"a": input_a, "b": input_b, "c": input_c})

    compiled = compiler.run(model, {"a": input_a, "b": input_b, "c": input_c})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)
