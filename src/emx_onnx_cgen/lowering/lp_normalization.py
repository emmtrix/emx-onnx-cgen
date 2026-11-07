from __future__ import annotations

from ..ir.ops import LpNormalizationOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from ..validation import normalize_axis
from .common import node_dtype, value_shape
from .registry import register_lowering


@register_lowering("LpNormalization")
def lower_lp_normalization(graph: Graph, node: Node) -> LpNormalizationOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LpNormalization must have 1 input and 1 output")
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "LpNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    axis = normalize_axis(int(node.attrs.get("axis", -1)), input_shape, node)
    p = int(node.attrs.get("p", 2))
    if p not in {1, 2}:
        raise UnsupportedOpError("LpNormalization only supports p=1 or p=2")
    return LpNormalizationOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=input_shape,
        axis=axis,
        p=p,
        dtype=op_dtype,
    )
