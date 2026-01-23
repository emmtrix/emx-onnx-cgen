from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import QLinearMatMulOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class QLinearMatMulSpec:
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


def resolve_qlinear_matmul_spec(graph: Graph, node: Node) -> QLinearMatMulSpec:
    if len(node.inputs) != 8 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "QLinearMatMul must have 8 inputs and 1 output"
        )
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[3], node)
    if len(input0_shape) < 1 or len(input1_shape) < 1:
        raise UnsupportedOpError(
            "QLinearMatMul inputs must be at least 1D, "
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
            "QLinearMatMul inner dimensions must match, "
            f"got {k_left} and {k_right}"
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
            "QLinearMatMul output shape must be "
            f"{output_shape}, got {expected_output_shape}"
        )
    return QLinearMatMulSpec(
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
                "QLinearMatMul batch dimensions must be broadcastable, "
                f"got {left} x {right}"
            )
        broadcast_shape.append(max(left_dim, right_dim))
    return tuple(broadcast_shape), left_padded, right_padded


def _ensure_scalar_input(
    graph: Graph, name: str, node: Node, label: str
) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearMatMul {label} must be scalar, got shape {shape}"
        )
    return shape


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not dtype.is_float:
        raise UnsupportedOpError(
            f"QLinearMatMul {label} must be float16/float/double"
        )


@register_lowering("QLinearMatMul")
def lower_qlinear_matmul(graph: Graph, node: Node) -> QLinearMatMulOp:
    spec = resolve_qlinear_matmul_spec(graph, node)
    input0_dtype = _value_dtype(graph, node.inputs[0], node)
    input1_dtype = _value_dtype(graph, node.inputs[3], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input0_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "QLinearMatMul supports uint8/int8 inputs only"
        )
    if input1_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "QLinearMatMul supports uint8/int8 inputs only"
        )
    if output_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "QLinearMatMul supports uint8/int8 outputs only"
        )
    input0_scale_dtype = _value_dtype(graph, node.inputs[1], node)
    input1_scale_dtype = _value_dtype(graph, node.inputs[4], node)
    output_scale_dtype = _value_dtype(graph, node.inputs[6], node)
    _ensure_scale_dtype(input0_scale_dtype, "a_scale")
    _ensure_scale_dtype(input1_scale_dtype, "b_scale")
    _ensure_scale_dtype(output_scale_dtype, "y_scale")
    input0_zero_dtype = _value_dtype(graph, node.inputs[2], node)
    input1_zero_dtype = _value_dtype(graph, node.inputs[5], node)
    output_zero_dtype = _value_dtype(graph, node.inputs[7], node)
    if input0_zero_dtype != input0_dtype:
        raise UnsupportedOpError(
            "QLinearMatMul a_zero_point dtype must match a"
        )
    if input1_zero_dtype != input1_dtype:
        raise UnsupportedOpError(
            "QLinearMatMul b_zero_point dtype must match b"
        )
    if output_zero_dtype != output_dtype:
        raise UnsupportedOpError(
            "QLinearMatMul y_zero_point dtype must match y"
        )
    input0_scale_shape = _ensure_scalar_input(
        graph, node.inputs[1], node, "a_scale"
    )
    input1_scale_shape = _ensure_scalar_input(
        graph, node.inputs[4], node, "b_scale"
    )
    output_scale_shape = _ensure_scalar_input(
        graph, node.inputs[6], node, "y_scale"
    )
    input0_zero_shape = _ensure_scalar_input(
        graph, node.inputs[2], node, "a_zero_point"
    )
    input1_zero_shape = _ensure_scalar_input(
        graph, node.inputs[5], node, "b_zero_point"
    )
    output_zero_shape = _ensure_scalar_input(
        graph, node.inputs[7], node, "y_zero_point"
    )
    return QLinearMatMulOp(
        input0=node.inputs[0],
        input0_scale=node.inputs[1],
        input0_zero_point=node.inputs[2],
        input1=node.inputs[3],
        input1_scale=node.inputs[4],
        input1_zero_point=node.inputs[5],
        output_scale=node.inputs[6],
        output_zero_point=node.inputs[7],
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
        input0_dtype=input0_dtype,
        input1_dtype=input1_dtype,
        dtype=output_dtype,
        input0_scale_dtype=input0_scale_dtype,
        input1_scale_dtype=input1_scale_dtype,
        output_scale_dtype=output_scale_dtype,
        input0_scale_shape=input0_scale_shape,
        input1_scale_shape=input1_scale_shape,
        output_scale_shape=output_scale_shape,
        input0_zero_shape=input0_zero_shape,
        input1_zero_shape=input1_zero_shape,
        output_zero_shape=output_zero_shape,
    )
