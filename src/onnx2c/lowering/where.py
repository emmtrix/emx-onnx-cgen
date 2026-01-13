from __future__ import annotations

from ..codegen.c_emitter import WhereOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _broadcast_shape(shapes: tuple[tuple[int, ...], ...], node: Node) -> tuple[int, ...]:
    if not shapes:
        return ()
    max_rank = max(len(shape) for shape in shapes)
    padded = [
        (1,) * (max_rank - len(shape)) + shape
        for shape in shapes
    ]
    broadcast: list[int] = []
    for dims in zip(*padded):
        dim = max(dims)
        if any(item not in (1, dim) for item in dims):
            raise ShapeInferenceError(
                f"{node.op_type} inputs must be broadcastable, got {shapes}"
            )
        broadcast.append(dim)
    return tuple(broadcast)


@register_lowering("Where")
def lower_where(graph: Graph, node: Node) -> WhereOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Where must have 3 inputs and 1 output")
    condition_name, x_name, y_name = node.inputs
    output_name = node.outputs[0]
    condition_dtype = _value_dtype(graph, condition_name, node)
    if condition_dtype != "bool":
        raise UnsupportedOpError(
            f"Where expects bool condition, got {condition_dtype}"
        )
    x_dtype = _value_dtype(graph, x_name, node)
    y_dtype = _value_dtype(graph, y_name, node)
    output_dtype = _value_dtype(graph, output_name, node)
    if x_dtype != y_dtype or output_dtype != x_dtype:
        raise UnsupportedOpError(
            "Where expects matching input/output dtypes, "
            f"got {x_dtype}, {y_dtype}, {output_dtype}"
        )
    condition_shape = _value_shape(graph, condition_name, node)
    x_shape = _value_shape(graph, x_name, node)
    y_shape = _value_shape(graph, y_name, node)
    output_shape = _value_shape(graph, output_name, node)
    broadcast_shape = _broadcast_shape(
        (condition_shape, x_shape, y_shape),
        node,
    )
    if output_shape != broadcast_shape:
        raise ShapeInferenceError(
            f"Where output shape must be {broadcast_shape}, got {output_shape}"
        )
    return WhereOp(
        condition=condition_name,
        input_x=x_name,
        input_y=y_name,
        output=output_name,
        condition_shape=condition_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        output_shape=output_shape,
        dtype=output_dtype,
    )
