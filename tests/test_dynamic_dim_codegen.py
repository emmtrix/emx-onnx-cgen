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
from emx_onnx_cgen.errors import ShapeInferenceError
from emx_onnx_cgen.ir.op_base import BroadcastingOpBase
from emx_onnx_cgen.lowering.common import _shape_values_from_input
from emx_onnx_cgen.onnx_import import import_onnx


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


def _colliding_dim_names_model() -> onnx.ModelProto:
    """``n + 1`` and ``n - 1`` both sanitize to ``n_1`` (no plain ``n`` dim)."""

    return _make_model(
        [
            helper.make_node("Relu", ["x"], ["x_out"]),
            helper.make_node("Relu", ["y"], ["y_out"]),
        ],
        [_float("x", ["n + 1"]), _float("y", ["n - 1"])],
        [_float("x_out", ["n + 1"]), _float("y_out", ["n - 1"])],
    )


def _tensor_named_like_dim_model() -> onnx.ModelProto:
    """Input tensor ``p`` with dim ``p``: the parameter is renamed to ``p_dim``."""

    return _make_model(
        [helper.make_node("Concat", ["p", "new"], ["present"], axis=0)],
        [_float("p", ["p", 3]), _float("new", [1, 3])],
        [_float("present", ["p + 1", 3])],
    )


_MODELS = {
    "kv_cache_concat": (_kv_cache_concat_model, [{"p": 1}, {"p": 5}]),
    "colliding_dim_names": (
        _colliding_dim_names_model,
        [{"n + 1": 3, "n - 1": 1}, {"n + 1": 6, "n - 1": 4}],
    ),
    "tensor_named_like_dim": (_tensor_named_like_dim_model, [{"p": 2}, {"p": 4}]),
    "attention_matmul": (_attention_matmul_model, [{"n": 1}, {"n": 6}]),
    "repeat_kv_expand": (_repeat_kv_expand_model, [{"n": 2}, {"n": 7}]),
    "mask_from_shape": (_mask_from_shape_model, [{"n": 1}, {"n": 5}]),
    "where_with_scalar": (_where_with_scalar_model, [{"n": 3}, {"n": 6}]),
}


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_generated_code_has_no_placeholder_dims(case: str) -> None:
    builder, _ = _MODELS[case]
    generated = Compiler(CompilerOptions()).compile(builder())
    for placeholder in ("< -1;", "[-1]", "= -1LL;", "[dim_", " dim_0"):
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


def test_derived_dim_uses_the_renamed_parameter() -> None:
    generated = Compiler(CompilerOptions()).compile(_tensor_named_like_dim_model())
    assert "void model(int p_dim, const float p[restrict p_dim][3]," in generated
    assert "float present[restrict (p_dim + 1)][3])" in generated


def test_colliding_sanitized_dim_names_stay_distinct() -> None:
    generated = Compiler(CompilerOptions()).compile(_colliding_dim_names_model())
    assert "void model(int n_1, int n_1_dim," in generated


def test_expand_rejects_a_dynamic_dim_that_changes_its_name() -> None:
    # x[1, a] expanded to y's shape [3, b]: a may be 1 at runtime (broadcast),
    # which the kernel cannot express, so a and b must be the same dim.
    model = _make_model(
        [
            helper.make_node("Shape", ["y"], ["y_shape"]),
            helper.make_node("Expand", ["x", "y_shape"], ["out"]),
        ],
        [_float("x", [1, "a"]), _float("y", [3, "b"])],
        [_float("out", [3, "b"])],
    )
    with pytest.raises(Exception, match="must keep its dim_param"):
        Compiler(CompilerOptions()).compile(model)


def test_zero_size_dims_broadcast_like_other_sizes() -> None:
    assert BroadcastingOpBase.broadcast_shapes((0,), (0,)) == (0,)
    assert BroadcastingOpBase.broadcast_shapes((2, 0), (1,)) == (2, 0)
    with pytest.raises(ShapeInferenceError):
        BroadcastingOpBase.broadcast_shapes((2, 0), (3,))


def test_static_element_of_a_partly_dynamic_shape_resolves() -> None:
    # Shape(x)[1] of x[N, 6] is the constant 6 although axis 0 is dynamic.
    index = numpy_helper.from_array(np.array([1], dtype=np.int64), "index")
    model = _make_model(
        [
            helper.make_node("Shape", ["x"], ["x_shape"]),
            helper.make_node("Gather", ["x_shape", "index"], ["width"], axis=0),
            helper.make_node("Gather", ["x_shape", "zero"], ["batch"], axis=0),
        ],
        [_float("x", ["N", 6])],
        [
            helper.make_tensor_value_info("width", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("batch", TensorProto.INT64, [1]),
        ],
        [index, numpy_helper.from_array(np.array([0], dtype=np.int64), "zero")],
    )
    graph = import_onnx(model)
    assert _shape_values_from_input(graph, "width", None) == [6]
    assert _shape_values_from_input(graph, "batch", None) is None
