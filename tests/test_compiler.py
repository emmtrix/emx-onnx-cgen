from __future__ import annotations

import onnx
from onnx import TensorProto, helper
import pytest

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.errors import ShapeInferenceError


def _make_split_dynamic_dim_model() -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node(
                    "Split",
                    ["x"],
                    ["y0", "y1"],
                    axis=1,
                    num_outputs=2,
                    name="split_dynamic",
                )
            ],
            name="split_dynamic_graph",
            inputs=[
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["N", 2, 3]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "y0", TensorProto.FLOAT, ["N", 1, 3]
                ),
                helper.make_tensor_value_info(
                    "y1", TensorProto.FLOAT, ["N", 1, 3]
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def _make_constant_of_shape_via_neg_model() -> onnx.ModelProto:
    minus_one = helper.make_tensor("minus_one", TensorProto.INT64, [1], [-1])
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node("Neg", ["minus_one"], ["shape_len"]),
                helper.make_node("ConstantOfShape", ["shape_len"], ["out"]),
            ],
            name="constant_of_shape_via_neg_graph",
            inputs=[],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N"])
            ],
            initializer=[minus_one],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _make_range_from_shape_gather_model() -> onnx.ModelProto:
    zero = helper.make_tensor("zero", TensorProto.INT64, [1], [0])
    one = helper.make_tensor("one", TensorProto.INT64, [1], [1])
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node("Shape", ["x"], ["x_shape"]),
                helper.make_node("Gather", ["x_shape", "zero"], ["limit"]),
                helper.make_node("Squeeze", ["zero", "zero"], ["start"]),
                helper.make_node("Squeeze", ["one", "zero"], ["delta"]),
                helper.make_node("Range", ["start", "limit", "delta"], ["out"]),
            ],
            name="range_from_shape_gather_graph",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
            ],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.INT64, ["N"])
            ],
            initializer=[zero, one],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _make_range_with_float_constants_model() -> onnx.ModelProto:
    start = helper.make_tensor("start", TensorProto.FLOAT, [], [0.0])
    limit = helper.make_tensor("limit", TensorProto.FLOAT, [], [3.0])
    delta = helper.make_tensor("delta", TensorProto.FLOAT, [], [1.0])
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node("Range", ["start", "limit", "delta"], ["out"]),
            ],
            name="range_float_constants_graph",
            inputs=[],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N"])
            ],
            initializer=[start, limit, delta],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _make_range_from_rank_size_model() -> onnx.ModelProto:
    zero = helper.make_tensor("zero", TensorProto.INT64, [1], [0])
    one = helper.make_tensor("one", TensorProto.INT64, [1], [1])
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node("Shape", ["x"], ["x_shape"]),
                helper.make_node("Size", ["x_shape"], ["rank"]),
                helper.make_node("Squeeze", ["zero", "zero"], ["start"]),
                helper.make_node("Squeeze", ["one", "zero"], ["delta"]),
                helper.make_node("Range", ["start", "rank", "delta"], ["out"]),
            ],
            name="range_from_rank_size_graph",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
            ],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.INT64, ["N"])
            ],
            initializer=[zero, one],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _make_reshape_target_from_rank_size_model() -> onnx.ModelProto:
    zero = helper.make_tensor("zero", TensorProto.INT64, [1], [0])
    return helper.make_model(
        helper.make_graph(
            nodes=[
                helper.make_node("Shape", ["shape_src"], ["shape_src_shape"]),
                helper.make_node("Size", ["shape_src_shape"], ["rank"]),
                helper.make_node("Unsqueeze", ["rank", "zero"], ["target_shape"]),
                helper.make_node("Reshape", ["data", "target_shape"], ["out"]),
            ],
            name="reshape_target_from_rank_size_graph",
            inputs=[
                helper.make_tensor_value_info("shape_src", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("data", TensorProto.FLOAT, [3]),
            ],
            outputs=[
                helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N"])
            ],
            initializer=[zero],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def test_compile_debug_lowering_failure_context_disabled_by_default() -> None:
    compiler = Compiler(CompilerOptions())

    with pytest.raises(ShapeInferenceError, match="Split does not support dynamic dims") as exc_info:
        compiler.compile(_make_split_dynamic_dim_model())

    assert "Lowering debug context:" not in str(exc_info.value)


def test_compile_debug_lowering_failure_context_enabled() -> None:
    compiler = Compiler(CompilerOptions(debug_lowering_failures=True))

    with pytest.raises(ShapeInferenceError, match="Split does not support dynamic dims") as exc_info:
        compiler.compile(_make_split_dynamic_dim_model())

    message = str(exc_info.value)
    assert "Lowering debug context:" in message
    assert "node_index: 0" in message
    assert "op_type: Split" in message
    assert "name: split_dynamic" in message
    assert "x: tensor[dtype=float, shape=(-1, 2, 3), dim_params=('N', None, None)]" in message
    assert "y0: tensor[dtype=float, shape=(-1, 1, 3), dim_params=('N', None, None)]" in message


def test_compile_resolves_constant_of_shape_from_negated_int_list() -> None:
    generated = Compiler(CompilerOptions()).compile(_make_constant_of_shape_via_neg_model())

    assert "OpType: ConstantOfShape" in generated


def test_compile_resolves_range_length_from_shape_gather_chain() -> None:
    generated = Compiler(CompilerOptions()).compile(_make_range_from_shape_gather_model())

    assert "OpType: Range" in generated


def test_compile_allows_range_shape_inference_with_float_constants() -> None:
    generated = Compiler(CompilerOptions()).compile(_make_range_with_float_constants_model())

    assert "OpType: Range" in generated


def test_compile_resolves_range_length_from_rank_size_chain() -> None:
    generated = Compiler(CompilerOptions()).compile(_make_range_from_rank_size_model())

    assert "OpType: Range" in generated


def test_compile_resolves_reshape_target_from_rank_size_chain() -> None:
    generated = Compiler(CompilerOptions()).compile(_make_reshape_target_from_rank_size_model())

    assert "OpType: Reshape" in generated
