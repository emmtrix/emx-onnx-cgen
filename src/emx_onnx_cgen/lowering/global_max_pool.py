from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import ReduceOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("GlobalMaxPool")
def lower_global_max_pool(graph: Graph, node: Node) -> ReduceOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("GlobalMaxPool must have 1 input and 1 output")
    if node.attrs:
        raise UnsupportedOpError("GlobalMaxPool has unsupported attributes")
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            "GlobalMaxPool expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if op_dtype not in {ScalarType.F16, ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(
            "GlobalMaxPool supports float16, float, and double inputs only"
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError(
            "GlobalMaxPool expects input rank of at least 3"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (input_shape[0], input_shape[1]) + (
        1,
    ) * (len(input_shape) - 2)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "GlobalMaxPool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    axes = tuple(range(2, len(input_shape)))
    return ReduceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        axes=axes,
        axes_input=None,
        axes_input_shape=None,
        axes_input_dtype=None,
        keepdims=True,
        noop_with_empty_axes=False,
        reduce_kind="max",
        reduce_count=None,
        dtype=op_dtype,
    )
