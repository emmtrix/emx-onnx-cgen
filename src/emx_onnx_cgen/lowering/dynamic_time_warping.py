from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import DynamicTimeWarpingOp
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("DynamicTimeWarping")
def lower_dynamic_time_warping(graph: Graph, node: Node) -> DynamicTimeWarpingOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("DynamicTimeWarping must have 1 input and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]

    input_val = graph.find_value(input_name)
    output_val = graph.find_value(output_name)

    input_dtype = input_val.type.dtype
    if input_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(
            f"DynamicTimeWarping: input must be float32 or float64, got {input_dtype.onnx_name}"
        )

    input_shape = input_val.type.shape
    if len(input_shape) != 2:
        raise ShapeInferenceError(
            f"DynamicTimeWarping: input must have rank 2, got shape {input_shape}"
        )

    rows, cols = input_shape
    if rows < 0 or cols < 0:
        raise ShapeInferenceError(
            "DynamicTimeWarping does not support dynamic input shapes; export with static shapes"
        )

    output_shape = output_val.type.shape
    if len(output_shape) != 2 or output_shape[0] != 2:
        raise ShapeInferenceError(
            f"DynamicTimeWarping: output must have shape [2, path_len], got {output_shape}"
        )
    path_len = output_shape[1]
    if path_len < 0:
        raise ShapeInferenceError(
            "DynamicTimeWarping does not support dynamic output path length; export with static shapes"
        )

    return DynamicTimeWarpingOp(
        input0=input_name,
        output=output_name,
        rows=rows,
        cols=cols,
        path_len=path_len,
    )
