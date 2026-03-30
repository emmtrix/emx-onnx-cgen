from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MatMulInteger16Op
from .matmul_integer import resolve_matmul_integer_spec
from .common import value_dtype as _value_dtype
from .registry import register_lowering


@register_lowering("MatMulInteger16")
def lower_matmul_integer16(graph: Graph, node: Node) -> MatMulInteger16Op:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "MatMulInteger16 must have exactly 2 inputs and 1 output"
        )
    input0_dtype = _value_dtype(graph, node.inputs[0], node)
    input1_dtype = _value_dtype(graph, node.inputs[1], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input0_dtype != ScalarType.I16:
        raise UnsupportedOpError(
            f"MatMulInteger16 expects int16 A input, got {input0_dtype}"
        )
    if input1_dtype != ScalarType.I16:
        raise UnsupportedOpError(
            f"MatMulInteger16 expects int16 B input, got {input1_dtype}"
        )
    if output_dtype != ScalarType.I32:
        raise UnsupportedOpError(
            f"MatMulInteger16 expects int32 output, got {output_dtype}"
        )
    spec = resolve_matmul_integer_spec(graph, node)
    return MatMulInteger16Op(
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
        dtype=output_dtype,
    )
