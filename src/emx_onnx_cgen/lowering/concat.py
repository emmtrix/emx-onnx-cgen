from __future__ import annotations

from ..ir.ops import ConcatOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering
from ..validation import validate_concat_shapes


@register_lowering("Concat")
def lower_concat(graph: Graph, node: Node) -> ConcatOp:
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Concat must have at least 1 input and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    output_shape = _value_shape(graph, node.outputs[0], node)
    input_shapes = tuple(_value_shape(graph, name, node) for name in node.inputs)
    axis = validate_concat_shapes(
        input_shapes,
        output_shape,
        int(node.attrs.get("axis", 0)),
    )
    return ConcatOp(
        inputs=node.inputs,
        output=node.outputs[0],
        axis=axis,
        input_shapes=input_shapes,
        output_shape=output_shape,
        dtype=op_dtype,
    )
