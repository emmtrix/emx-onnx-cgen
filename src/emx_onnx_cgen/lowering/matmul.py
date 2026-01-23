from __future__ import annotations

from dataclasses import dataclass

from ..ir.ops import MatMulOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class MatMulSpec:
    input0_shape: tuple[int, ...]
    input1_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    input0_batch_shape: tuple[int, ...]
    input1_batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    left_vector: bool
    right_vector: bool


def resolve_matmul_spec(graph: Graph, node: Node) -> MatMulSpec:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[1], node)
    if len(input0_shape) < 1 or len(input1_shape) < 1:
        raise UnsupportedOpError(
            "MatMul inputs must be at least 1D, "
            f"got {input0_shape} x {input1_shape}"
        )
    left_vector = len(input0_shape) == 1
    right_vector = len(input1_shape) == 1
    input0_effective = (1, input0_shape[0]) if left_vector else input0_shape
    input1_effective = (input1_shape[0], 1) if right_vector else input1_shape
    m, k_left = input0_effective[-2], input0_effective[-1]
    k_right, n = input1_effective[-2], input1_effective[-1]
    if k_left != k_right:
        raise ShapeInferenceError(
            f"MatMul inner dimensions must match, got {k_left} and {k_right}"
        )
    batch_shape, input0_batch_shape, input1_batch_shape = (
        _broadcast_batch_shapes(
            input0_effective[:-2], input1_effective[:-2], node
        )
    )
    if left_vector and right_vector:
        output_shape = batch_shape
    elif left_vector:
        output_shape = batch_shape + (n,)
    elif right_vector:
        output_shape = batch_shape + (m,)
    else:
        output_shape = batch_shape + (m, n)
    expected_output_shape = _value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            "MatMul output shape must be "
            f"{output_shape}, got {expected_output_shape}"
        )
    return MatMulSpec(
        input0_shape=input0_shape,
        input1_shape=input1_shape,
        output_shape=output_shape,
        batch_shape=batch_shape,
        input0_batch_shape=input0_batch_shape,
        input1_batch_shape=input1_batch_shape,
        m=m,
        n=n,
        k=k_left,
        left_vector=left_vector,
        right_vector=right_vector,
    )


def _broadcast_batch_shapes(
    left: tuple[int, ...], right: tuple[int, ...], node: Node
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    max_rank = max(len(left), len(right))
    left_padded = (1,) * (max_rank - len(left)) + left
    right_padded = (1,) * (max_rank - len(right)) + right
    broadcast_shape = []
    for left_dim, right_dim in zip(left_padded, right_padded):
        if not (left_dim == right_dim or left_dim == 1 or right_dim == 1):
            raise ShapeInferenceError(
                "MatMul batch dimensions must be broadcastable, "
                f"got {left} x {right}"
            )
        broadcast_shape.append(max(left_dim, right_dim))
    return tuple(broadcast_shape), left_padded, right_padded


@register_lowering("MatMul")
def lower_matmul(graph: Graph, node: Node) -> MatMulOp:
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    spec = resolve_matmul_spec(graph, node)
    return MatMulOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
        input0_shape=spec.input0_shape,
        input1_shape=spec.input1_shape,
        output_shape=spec.output_shape,
        batch_shape=spec.batch_shape,
        input0_batch_shape=spec.input0_batch_shape,
        input1_batch_shape=spec.input1_batch_shape,
        m=spec.m,
        n=spec.n,
        k=spec.k,
        left_vector=spec.left_vector,
        right_vector=spec.right_vector,
        dtype=op_dtype,
    )
