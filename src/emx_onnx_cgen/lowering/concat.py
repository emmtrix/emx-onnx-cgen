from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import ConcatOp
from .common import node_dtype as _node_dtype
from .common import value_has_dim_params as _value_has_dim_params
from .common import value_shape as _value_shape
from .registry import register_lowering
from ..validation import normalize_concat_axis, validate_concat_shapes


@register_lowering("Concat")
def lower_concat(graph: Graph, node: Node) -> ConcatOp:
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Concat must have at least 1 input and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if _value_has_dim_params(graph, node.outputs[0]):
        output_shape = ()
    input_shapes = tuple(_value_shape(graph, name, node) for name in node.inputs)
    axis = int(node.attrs.get("axis", 0))
    if output_shape:
        axis = validate_concat_shapes(
            input_shapes,
            output_shape,
            axis,
        )
    else:
        ranks = {len(shape) for shape in input_shapes}
        if len(ranks) != 1:
            raise UnsupportedOpError(
                f"Concat inputs must have matching ranks, got {input_shapes}"
            )
        rank = ranks.pop()
        axis = normalize_concat_axis(axis, rank)
        base_shape = list(input_shapes[0])
        axis_dim = 0
        for shape in input_shapes:
            if len(shape) != rank:
                raise UnsupportedOpError(
                    f"Concat inputs must have matching ranks, got {input_shapes}"
                )
            for dim_index, dim in enumerate(shape):
                if dim_index == axis:
                    continue
                if dim != base_shape[dim_index]:
                    raise UnsupportedOpError(
                        "Concat inputs must match on non-axis dimensions, "
                        f"got {input_shapes}"
                    )
            axis_dim += shape[axis]
        base_shape[axis] = axis_dim
        output_shape = tuple(base_shape)
        if isinstance(graph, GraphContext):
            graph.set_shape(node.outputs[0], output_shape)
    return ConcatOp(
        inputs=node.inputs,
        output=node.outputs[0],
        axis=axis,
        input_shapes=input_shapes,
        output_shape=output_shape,
        dtype=op_dtype,
    )
