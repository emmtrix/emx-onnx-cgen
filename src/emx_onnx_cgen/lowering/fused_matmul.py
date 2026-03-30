from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import FusedMatMulOp
from .common import value_shape
from .registry import register_lowering


def _apply_transpositions(
    shape: tuple[int, ...],
    trans: bool,
    trans_batch: bool,
) -> tuple[int, ...]:
    """Return the effective shape after batch and matrix transpositions.

    ``trans_batch`` applies a cyclic left rotation to all dims except the last
    (the K dimension): ``[d0, d1, ..., d_{n-2}, k] -> [d1, ..., d_{n-2}, d0, k]``.
    This moves the first dimension to the position just before K, making it the
    effective matrix-row dimension.  ``trans`` then swaps the last two dims.
    """
    if len(shape) < 2:
        return shape
    if trans_batch:
        # Cyclic left rotation of all dims except the last (k).
        shape = shape[1:-1] + (shape[0],) + (shape[-1],)
    if trans:
        shape = shape[:-2] + (shape[-1], shape[-2])
    return shape


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
            f"FusedMatMul batch dimensions are not broadcastable: {left} vs {right}"
        )
    return tuple(reversed(result))


@register_lowering("FusedMatMul")
def lower_fused_matmul(graph: Graph, node: Node) -> FusedMatMulOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("FusedMatMul must have 2 inputs and 1 output")

    alpha = float(node.attrs.get("alpha", 1.0))
    trans_a = bool(int(node.attrs.get("transA", 0)))
    trans_b = bool(int(node.attrs.get("transB", 0)))
    trans_batch_a = bool(int(node.attrs.get("transBatchA", 0)))
    trans_batch_b = bool(int(node.attrs.get("transBatchB", 0)))

    left_shape = value_shape(graph, node.inputs[0], node)
    right_shape = value_shape(graph, node.inputs[1], node)

    if len(left_shape) == 0 or len(right_shape) == 0:
        raise UnsupportedOpError("FusedMatMul does not support scalar inputs")

    eff_left = _apply_transpositions(left_shape, trans_a, trans_batch_a)
    eff_right = _apply_transpositions(right_shape, trans_b, trans_batch_b)

    left_batch = eff_left[:-2] if len(eff_left) > 1 else ()
    right_batch = eff_right[:-2] if len(eff_right) > 1 else ()
    left_k = eff_left[-1]
    right_k = eff_right[-2] if len(eff_right) > 1 else eff_right[0]

    if left_k != right_k:
        raise ShapeInferenceError(
            f"FusedMatMul inner dimensions must match after transposition, "
            f"got effective shapes {eff_left} and {eff_right} "
            f"(original: {left_shape} and {right_shape})"
        )

    batch_shape = _broadcast_batch_shape(left_batch, right_batch)
    if len(eff_left) == 1 and len(eff_right) == 1:
        output_shape: tuple[int, ...] = ()
    elif len(eff_left) == 1:
        output_shape = (*batch_shape, eff_right[-1])
    elif len(eff_right) == 1:
        output_shape = (*batch_shape, eff_left[-2])
    else:
        output_shape = (*batch_shape, eff_left[-2], eff_right[-1])

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)

    return FusedMatMulOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
        alpha=alpha,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_batch_a=trans_batch_a,
        trans_batch_b=trans_batch_b,
    )
