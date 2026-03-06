from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import DetOp
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("Det")
def lower_det(graph: Graph, node: Node) -> DetOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Det must have 1 input and 1 output")
    input_name = node.inputs[0]
    output_name = node.outputs[0]

    input_value = graph.find_value(input_name)
    output_value = graph.find_value(output_name)
    input_shape = input_value.type.shape
    output_shape = output_value.type.shape
    input_dtype = input_value.type.dtype
    output_dtype = output_value.type.dtype

    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            f"Det expects matching input/output dtypes, got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if input_dtype not in {
        ScalarType.F16,
        ScalarType.F32,
        ScalarType.F64,
        ScalarType.BF16,
    }:
        raise UnsupportedOpError(
            f"Det does not support dtype {input_dtype.onnx_name}; expected float16, float, double, or bfloat16"
        )

    rank = len(input_shape)
    if rank < 2:
        raise ShapeInferenceError(
            f"Det expects input rank >= 2, got rank {rank} with shape {input_shape}"
        )

    matrix_rows = input_shape[-2]
    matrix_cols = input_shape[-1]
    if matrix_rows != matrix_cols:
        raise ShapeInferenceError(
            f"Det expects square matrices on the last two dimensions, got shape {input_shape}"
        )
    if matrix_rows < 0:
        raise ShapeInferenceError(
            "Det does not support dynamic matrix dimensions; export with static shapes"
        )

    expected_output_shape = input_shape[:-2]
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"Det output shape must be {expected_output_shape}, got {output_shape}"
        )

    return DetOp(
        input0=input_name,
        output=output_name,
    )
