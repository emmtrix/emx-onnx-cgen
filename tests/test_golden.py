from __future__ import annotations

# NOTE: This file contains only golden codegen tests.

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from typing import Callable
from onnx import TensorProto, helper

from emx_onnx_cgen import Compiler
from emx_onnx_cgen.compiler import CompilerOptions
from golden_utils import assert_golden


def _make_binary_model(op_type: str, shape: list[int]) -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, shape)
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    node = helper.make_node(op_type, inputs=["a", "b"], outputs=["out"])
    graph = helper.make_graph(
        [node], f"{op_type.lower()}_graph", [input_a, input_b], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_mod_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Mod", inputs=["a", "b"], outputs=["out"], fmod=1
    )
    graph = helper.make_graph([node], "mod_graph", [input_a, input_b], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


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


def _make_mul_add_relu_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
    input_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    mul_node = helper.make_node("Mul", inputs=["a", "b"], outputs=["mul_out"])
    add_node = helper.make_node("Add", inputs=["mul_out", "c"], outputs=["add_out"])
    relu_node = helper.make_node("Relu", inputs=["add_out"], outputs=["out"])
    graph = helper.make_graph(
        [mul_node, add_node, relu_node],
        "mul_add_relu_graph",
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


def _make_large_temp_model() -> onnx.ModelProto:
    shape = [1, 257]
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, shape)
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    add_node = helper.make_node("Add", inputs=["a", "b"], outputs=["tmp"])
    relu_node = helper.make_node("Relu", inputs=["tmp"], outputs=["out"])
    graph = helper.make_graph(
        [add_node, relu_node],
        "large_temp_graph",
        [input_a, input_b],
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


def _make_relu_model() -> onnx.ModelProto:
    input_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Relu", inputs=["x"], outputs=["out"])
    graph = helper.make_graph([node], "relu_graph", [input_x], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_matmul_model() -> onnx.ModelProto:
    input_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    input_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])
    node = helper.make_node("MatMul", inputs=["a", "b"], outputs=["out"])
    graph = helper.make_graph([node], "matmul_graph", [input_a, input_b], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_add_initializer_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    weight_values = np.linspace(0.1, 0.6, num=6, dtype=np.float32).reshape(2, 3)
    weight_initializer = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        dims=[2, 3],
        vals=weight_values.flatten().tolist(),
    )
    weight_info = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", inputs=["in0", "weight"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "add_init_graph",
        [input_info, weight_info],
        [output],
        initializer=[weight_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _mark_dynamic_dims(
    model: onnx.ModelProto,
    *,
    input_dim_params: list[str],
    output_dim_params: list[str],
) -> onnx.ModelProto:
    for value_info in model.graph.input:
        for dim, dim_param in zip(
            value_info.type.tensor_type.shape.dim, input_dim_params
        ):
            dim.dim_param = dim_param
    for value_info in model.graph.output:
        for dim, dim_param in zip(
            value_info.type.tensor_type.shape.dim, output_dim_params
        ):
            dim.dim_param = dim_param
    return model


def test_codegen_golden_tanh() -> None:
    model = _make_tanh_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "tanh_model.c"
    assert_golden(generated, golden_path)


def test_codegen_golden_relu() -> None:
    model = _make_relu_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "relu_model.c"
    assert_golden(generated, golden_path)


def test_codegen_golden_dynamic_dims() -> None:
    model = _mark_dynamic_dims(
        _make_relu_model(),
        input_dim_params=["N", "C"],
        output_dim_params=["N", "C"],
    )
    compiler = Compiler(
        CompilerOptions(
            template_dir=Path("templates"),
            model_name="dynamic_dims_model",
        )
    )
    generated = compiler.compile(model)
    golden_path = (
        Path(__file__).parent / "golden" / "dynamic_dims_model.c"
    )
    assert_golden(generated, golden_path)


def test_tanh_matches_onnxruntime() -> None:
    model = _make_tanh_model()
    compiler = Compiler()
    input_x = np.random.rand(2, 3).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"x": input_x})

    compiled = compiler.run(model, {"x": input_x})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)


def test_relu_matches_onnxruntime() -> None:
    model = _make_relu_model()
    compiler = Compiler()
    input_x = np.random.randn(2, 3).astype(np.float32)

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


def test_codegen_golden_add_no_restrict() -> None:
    model = _make_add_model()
    compiler = Compiler(
        CompilerOptions(template_dir=Path("templates"), restrict_arrays=False)
    )
    generated = compiler.compile(model)
    golden_path = (
        Path(__file__).parent / "golden" / "add_model_no_restrict.c"
    )
    assert_golden(generated, golden_path)


def test_codegen_golden_add_initializer() -> None:
    model = _make_add_initializer_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "add_initializer_model.c"
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


@pytest.mark.parametrize(
    "op_type, expected_fn",
    [
        ("Sub", lambda left, right: left - right),
        ("Div", lambda left, right: left / right),
        ("Pow", np.power),
        ("Min", np.minimum),
        ("Max", np.maximum),
        ("Mean", lambda left, right: (left + right) * 0.5),
        ("Sum", lambda left, right: left + right),
        ("PRelu", lambda left, right: np.where(left > 0.0, left, right * left)),
    ],
    ids=["Sub", "Div", "Pow", "Min", "Max", "Mean", "Sum", "PRelu"],
)
def test_binary_ops_match_numpy(
    op_type: str, expected_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    model = _make_binary_model(op_type, [2, 3])
    compiler = Compiler()
    left = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    right = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    if op_type in {"Div", "Pow"}:
        left = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    if op_type == "PRelu":
        left = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
        right = np.random.uniform(0.1, 1.0, size=(2, 3)).astype(np.float32)
    expected = expected_fn(left, right)
    compiled = compiler.run(model, {"a": left, "b": right})
    np.testing.assert_allclose(compiled["out"], expected, rtol=1e-4, atol=1e-5)


def test_mod_matches_numpy() -> None:
    model = _make_mod_model()
    compiler = Compiler()
    left = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    right = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    expected = np.fmod(left, right)
    compiled = compiler.run(model, {"a": left, "b": right})
    np.testing.assert_allclose(compiled["out"], expected, rtol=1e-4, atol=1e-5)


def test_codegen_golden_mul_add() -> None:
    model = _make_mul_add_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "mul_add_model.c"
    assert_golden(generated, golden_path)


def test_codegen_golden_mul_add_relu() -> None:
    model = _make_mul_add_relu_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "mul_add_relu_model.c"
    assert_golden(generated, golden_path)


def test_codegen_golden_large_temp_static() -> None:
    model = _make_large_temp_model()
    compiler = Compiler(
        CompilerOptions(
            template_dir=Path("templates"),
            model_name="large_temp_static_model",
        )
    )
    generated = compiler.compile(model)
    golden_path = (
        Path(__file__).parent / "golden" / "large_temp_static_model.c"
    )
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


def test_mul_add_relu_matches_onnxruntime() -> None:
    model = _make_mul_add_relu_model()
    compiler = Compiler()
    input_a = np.random.rand(2, 3).astype(np.float32)
    input_b = np.random.rand(2, 3).astype(np.float32)
    input_c = np.random.rand(2, 3).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"a": input_a, "b": input_b, "c": input_c})

    compiled = compiler.run(model, {"a": input_a, "b": input_b, "c": input_c})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)


def test_codegen_golden_matmul() -> None:
    model = _make_matmul_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "matmul_model.c"
    assert_golden(generated, golden_path)


def test_codegen_includes_testbench() -> None:
    model = _make_add_model()
    options = CompilerOptions(template_dir=Path("templates"), emit_testbench=True)
    compiler = Compiler(options)
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / "add_model_testbench.c"
    assert_golden(generated, golden_path)


def test_matmul_matches_onnxruntime() -> None:
    model = _make_matmul_model()
    compiler = Compiler()
    input_a = np.random.rand(2, 3).astype(np.float32)
    input_b = np.random.rand(3, 4).astype(np.float32)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"a": input_a, "b": input_b})

    compiled = compiler.run(model, {"a": input_a, "b": input_b})
    np.testing.assert_allclose(compiled["out"], ort_out, rtol=1e-4, atol=1e-5)
