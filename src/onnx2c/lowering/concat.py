from __future__ import annotations

from ..codegen.c_emitter import ConcatOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("Concat")
def lower_concat(graph: Graph, node: Node) -> ConcatOp:
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Concat must have at least 1 input and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    output_shape = _value_shape(graph, node.outputs[0], node)
    input_shapes = tuple(_value_shape(graph, name, node) for name in node.inputs)
    ranks = {len(shape) for shape in input_shapes}
    if len(ranks) != 1:
        raise ShapeInferenceError(
            f"Concat inputs must have matching ranks, got {input_shapes}"
        )
    rank = ranks.pop()
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(
            f"Concat axis out of range for rank {rank}: {axis}"
        )
    base_shape = list(input_shapes[0])
    axis_dim = 0
    for shape in input_shapes:
        if len(shape) != rank:
            raise ShapeInferenceError(
                f"Concat inputs must have matching ranks, got {input_shapes}"
            )
        for dim_index, dim in enumerate(shape):
            if dim_index == axis:
                continue
            if dim != base_shape[dim_index]:
                raise ShapeInferenceError(
                    "Concat inputs must match on non-axis dimensions, "
                    f"got {input_shapes}"
                )
        axis_dim += shape[axis]
    base_shape[axis] = axis_dim
    expected_output_shape = tuple(base_shape)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Concat output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return ConcatOp(
        inputs=node.inputs,
        output=node.outputs[0],
        axis=axis,
        input_shapes=input_shapes,
        output_shape=output_shape,
        dtype=op_dtype,
    )
