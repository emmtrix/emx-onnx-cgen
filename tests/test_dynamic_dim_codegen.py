"""Kernels whose tensors have a named dynamic dimension (``dim_param``).

Each model is the smallest graph for one pattern of the Qwen2.5-0.5B decoder
(onnx-community/Qwen2.5-0.5B-Instruct) with a dynamic ``past_sequence_length``.
ONNX shape inference already names every dimension here, so these models test
lowering and code generation only; the generated code must use the runtime
dimension wherever the shape is dynamic and match ONNX Runtime for several
sizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from dynamic_dim_verify import verify_for_dims
from emx_onnx_cgen.compiler import Compiler, CompilerOptions


def _make_model(nodes, inputs, outputs, initializers=()) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes, "dynamic_dim_graph", inputs, outputs, list(initializers)
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _float(name: str, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _kv_cache_concat_model() -> onnx.ModelProto:
    """``present = Concat(past, new, axis=1)``: the KV cache grows by one."""

    return _make_model(
        [helper.make_node("Concat", ["past", "new"], ["present"], axis=1)],
        [_float("past", [2, "p", 3]), _float("new", [2, 1, 3])],
        [_float("present", [2, "p + 1", 3])],
    )


def _attention_matmul_model() -> onnx.ModelProto:
    """``probs @ values`` reduces over the dynamic KV length."""

    return _make_model(
        [helper.make_node("MatMul", ["probs", "values"], ["out"])],
        [_float("probs", [2, 1, "n"]), _float("values", [2, "n", 4])],
        [_float("out", [2, 1, 4])],
    )


def _repeat_kv_expand_model() -> onnx.ModelProto:
    """``Expand`` of the KV heads; the dynamic axis is kept (target dim 1)."""

    target = numpy_helper.from_array(np.array([2, 3, 1, 4], dtype=np.int64), "target")
    return _make_model(
        [helper.make_node("Expand", ["kv", "target"], ["out"])],
        [_float("kv", [2, 1, "n", 4])],
        [_float("out", [2, 3, "n", 4])],
        [target],
    )


def _mask_from_shape_model() -> onnx.ModelProto:
    """``x + ConstantOfShape(Shape(x))`` and an Unsqueeze, as for the causal mask."""

    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    return _make_model(
        [
            helper.make_node("Shape", ["x"], ["x_shape"]),
            helper.make_node("ConstantOfShape", ["x_shape"], ["ones"], value=one),
            helper.make_node("Add", ["x", "ones"], ["shifted"]),
            helper.make_node("Unsqueeze", ["shifted", "axes"], ["y"]),
        ],
        [_float("x", [2, "n"])],
        [_float("y", [1, 2, "n"])],
        [axes],
    )


def _where_with_scalar_model() -> onnx.ModelProto:
    """``Where(x > 0, x, 0)`` broadcasts a scalar against a dynamic shape."""

    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
    return _make_model(
        [
            helper.make_node("Greater", ["x", "zero"], ["positive"]),
            helper.make_node("Where", ["positive", "x", "zero"], ["y"]),
        ],
        [_float("x", [1, "n"])],
        [_float("y", [1, "n"])],
        [zero],
    )


_MODELS = {
    "kv_cache_concat": (_kv_cache_concat_model, [{"p": 1}, {"p": 5}]),
    "attention_matmul": (_attention_matmul_model, [{"n": 1}, {"n": 6}]),
    "repeat_kv_expand": (_repeat_kv_expand_model, [{"n": 2}, {"n": 7}]),
    "mask_from_shape": (_mask_from_shape_model, [{"n": 1}, {"n": 5}]),
    "where_with_scalar": (_where_with_scalar_model, [{"n": 3}, {"n": 6}]),
}


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_generated_code_has_no_placeholder_dims(case: str) -> None:
    builder, _ = _MODELS[case]
    generated = Compiler(CompilerOptions()).compile(builder())
    for placeholder in ("< -1;", "[-1]", "= -1LL;", "dim_0", "dim_1", "dim_2"):
        assert placeholder not in generated, placeholder


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_verify_against_onnxruntime(case: str, tmp_path: Path) -> None:
    builder, dim_sets = _MODELS[case]
    verify_for_dims(builder(), tmp_path, dim_sets)


def test_derived_dim_is_an_expression_of_its_parameter() -> None:
    # `p + 1` depends on the parameter `p`, so it has no parameter of its own.
    generated = Compiler(CompilerOptions()).compile(_kv_cache_concat_model())
    assert "void model(int p, const float past[restrict 2][p][3]," in generated
    assert "float present[restrict 2][(p + 1)][3])" in generated


def test_dim_param_that_is_no_identifier_is_sanitized() -> None:
    model = _make_model(
        [helper.make_node("Relu", ["x"], ["y"])],
        [_float("x", ["n + 1"])],
        [_float("y", ["n + 1"])],
    )
    generated = Compiler(CompilerOptions()).compile(model)
    assert "void model(int n_1, const float x[restrict n_1]" in generated


def test_negative_runtime_dims_are_rejected() -> None:
    generated = Compiler(CompilerOptions()).compile(_attention_matmul_model())
    assert "    if (n < 0) {\n        return;\n    }" in generated
