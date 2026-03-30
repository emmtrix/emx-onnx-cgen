from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import InverseOp
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("Inverse")
def lower_inverse(graph: Graph, node: Node) -> InverseOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Inverse must have 1 input and 1 output")
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
            f"Inverse expects matching input/output dtypes, got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if input_dtype not in {ScalarType.F16, ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(
            f"Inverse does not support dtype {input_dtype.onnx_name}; expected float16, float, or double"
        )

    rank = len(input_shape)
    if rank < 2:
        raise ShapeInferenceError(
            f"Inverse expects input rank >= 2, got rank {rank} with shape {input_shape}"
        )

    matrix_rows = input_shape[-2]
    matrix_cols = input_shape[-1]
    if matrix_rows != matrix_cols:
        raise ShapeInferenceError(
            f"Inverse expects square matrices on the last two dimensions, got shape {input_shape}"
        )
    if matrix_rows < 0:
        raise ShapeInferenceError(
            "Inverse does not support dynamic matrix dimensions; export with static shapes"
        )

    if output_shape != input_shape:
        raise ShapeInferenceError(
            f"Inverse output shape must match input shape {input_shape}, got {output_shape}"
        )

    return InverseOp(
        input0=input_name,
        output=output_name,
    )
