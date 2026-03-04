from __future__ import annotations

from ..ir.ops import MatMulOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from .common import value_shape
from .registry import register_lowering


def _broadcast_batch_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    left_rev = list(reversed(left))
    right_rev = list(reversed(right))
    for index in range(max(len(left_rev), len(right_rev))):
        left_dim = left_rev[index] if index < len(left_rev) else 1
        right_dim = right_rev[index] if index < len(right_rev) else 1
        if left_dim == right_dim:
            result.append(left_dim)
            continue
        if left_dim == 1:
            result.append(right_dim)
            continue
        if right_dim == 1:
            result.append(left_dim)
            continue
        raise ShapeInferenceError(
            f"MatMul batch dimensions are not broadcastable: {left} vs {right}"
        )
    return tuple(reversed(result))


@register_lowering("MatMul")
def lower_matmul(graph: Graph, node: Node) -> MatMulOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
    left_shape = value_shape(graph, node.inputs[0], node)
    right_shape = value_shape(graph, node.inputs[1], node)
    if len(left_shape) == 0 or len(right_shape) == 0:
        raise UnsupportedOpError("MatMul does not support scalar inputs")
    left_batch = left_shape[:-2] if len(left_shape) > 1 else ()
    right_batch = right_shape[:-2] if len(right_shape) > 1 else ()
    left_k = left_shape[-1]
    right_k = right_shape[-2] if len(right_shape) > 1 else right_shape[0]
    if left_k != right_k:
        raise ShapeInferenceError(
            f"MatMul inner dimensions must match, got {left_shape} and {right_shape}"
        )
    batch_shape = _broadcast_batch_shape(left_batch, right_batch)
    if len(left_shape) == 1 and len(right_shape) == 1:
        output_shape: tuple[int, ...] = ()
    elif len(left_shape) == 1:
        output_shape = (*batch_shape, right_shape[-1])
    elif len(right_shape) == 1:
        output_shape = (*batch_shape, left_shape[-2])
    else:
        output_shape = (*batch_shape, left_shape[-2], right_shape[-1])
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
    return MatMulOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
    )
