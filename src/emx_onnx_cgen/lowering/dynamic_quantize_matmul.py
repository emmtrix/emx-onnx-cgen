from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import DynamicQuantizeMatMulOp
from ..ir.context import GraphContext
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("DynamicQuantizeMatMul")
def lower_dynamic_quantize_matmul(
    graph: Graph, node: Node
) -> DynamicQuantizeMatMulOp:
    if len(node.inputs) != 5 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "DynamicQuantizeMatMul must have 5 inputs and 1 output"
        )

    # Parse optional inputs (empty string means absent).
    b_zero_name: str | None = node.inputs[3] if node.inputs[3] else None
    bias_name: str | None = node.inputs[4] if node.inputs[4] else None

    # A (input 0) must be float.
    input_a_dtype = _value_dtype(graph, node.inputs[0], node)
    if not input_a_dtype.is_float:
        raise UnsupportedOpError("DynamicQuantizeMatMul requires float A input")

    # B (input 1) must be uint8 or int8.
    input_b_dtype = _value_dtype(graph, node.inputs[1], node)
    if input_b_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "DynamicQuantizeMatMul supports uint8/int8 B input only"
        )

    # b_scale must be float.
    b_scale_dtype = _value_dtype(graph, node.inputs[2], node)
    if not b_scale_dtype.is_float:
        raise UnsupportedOpError("DynamicQuantizeMatMul b_scale must be float")

    # b_zero_point dtype must match B dtype.
    b_zero_shape: tuple[int, ...] | None = None
    b_zero_per_column = False
    if b_zero_name is not None:
        b_zero_dtype = _value_dtype(graph, b_zero_name, node)
        if b_zero_dtype != input_b_dtype:
            raise UnsupportedOpError(
                "DynamicQuantizeMatMul b_zero_point dtype must match B"
            )
        b_zero_shape_raw = _value_shape(graph, b_zero_name, node)
        b_zero_shape = b_zero_shape_raw

    # Resolve matrix shapes.
    input_a_shape = _value_shape(graph, node.inputs[0], node)
    input_b_shape = _value_shape(graph, node.inputs[1], node)
    if len(input_a_shape) != 2 or len(input_b_shape) != 2:
        raise UnsupportedOpError(
            f"DynamicQuantizeMatMul supports 2D inputs only, "
            f"got {input_a_shape} x {input_b_shape}"
        )
    m, k_left = input_a_shape
    k_right, n = input_b_shape
    if k_left != k_right:
        raise ShapeInferenceError(
            f"DynamicQuantizeMatMul inner dimensions must match, "
            f"got {k_left} and {k_right}"
        )
    output_shape = (m, n)

    # Validate b_zero_point shape now that we know N.
    if b_zero_name is not None and b_zero_shape is not None:
        if b_zero_shape in {(), (1,)}:
            pass  # scalar
        elif len(b_zero_shape) == 1 and b_zero_shape[0] == n:
            b_zero_per_column = True
        else:
            raise UnsupportedOpError(
                f"DynamicQuantizeMatMul b_zero_point must be scalar or shape [{n}], "
                f"got {b_zero_shape}"
            )

    # b_scale may be scalar or per-column [N].
    b_scale_shape = _value_shape(graph, node.inputs[2], node)
    b_scale_per_column = False
    if b_scale_shape in {(), (1,)}:
        pass  # scalar
    elif len(b_scale_shape) == 1 and b_scale_shape[0] == n:
        b_scale_per_column = True
    else:
        raise UnsupportedOpError(
            f"DynamicQuantizeMatMul b_scale must be scalar or shape [{n}], "
            f"got {b_scale_shape}"
        )

    # bias must be 1D [N] if present.
    bias_shape: tuple[int, ...] | None = None
    if bias_name is not None:
        bias_shape = _value_shape(graph, bias_name, node)
        if len(bias_shape) != 1 or bias_shape[0] != n:
            raise UnsupportedOpError(
                f"DynamicQuantizeMatMul bias must have shape [{n}], "
                f"got {bias_shape}"
            )

    lowered = DynamicQuantizeMatMulOp(
        input_a=node.inputs[0],
        input_b=node.inputs[1],
        b_scale=node.inputs[2],
        b_zero_point=b_zero_name,
        bias=bias_name,
        output=node.outputs[0],
        input_b_dtype=input_b_dtype,
        b_scale_dtype=b_scale_dtype,
        b_scale_shape=b_scale_shape,
        b_zero_shape=b_zero_shape,
        bias_shape=bias_shape,
        b_scale_per_column=b_scale_per_column,
        b_zero_per_column=b_zero_per_column,
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
        graph.set_dtype(node.outputs[0], ScalarType.F32)

    return lowered
