from __future__ import annotations

import math

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MatMulBnb4Op
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering

_SUPPORTED_QUANT_TYPES = {0, 1}  # 0 = FP4, 1 = NF4


def _int_attr(node: Node, name: str) -> int:
    val = node.attrs.get(name)
    if val is None:
        raise UnsupportedOpError(
            f"MatMulBnb4 requires attribute '{name}'"
        )
    return int(val)


@register_lowering("MatMulBnb4")
def lower_matmul_bnb4(graph: Graph, node: Node) -> MatMulBnb4Op:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "MatMulBnb4 must have exactly 3 inputs (A, B, absmax) "
            "and 1 output"
        )

    k = _int_attr(node, "K")
    n = _int_attr(node, "N")
    block_size = _int_attr(node, "block_size")
    quant_type = _int_attr(node, "quant_type")

    if quant_type not in _SUPPORTED_QUANT_TYPES:
        raise UnsupportedOpError(
            f"MatMulBnb4 quant_type must be in {sorted(_SUPPORTED_QUANT_TYPES)}, "
            f"got {quant_type}"
        )

    input0_shape = _value_shape(graph, node.inputs[0], node)
    if len(input0_shape) < 2:
        raise UnsupportedOpError(
            f"MatMulBnb4 input A must be at least 2-D, got shape {input0_shape}"
        )
    m = input0_shape[-2]
    k_from_shape = input0_shape[-1]
    if k_from_shape != k:
        raise ShapeInferenceError(
            f"MatMulBnb4 K attribute ({k}) does not match "
            f"input A last dimension ({k_from_shape})"
        )

    # Only 2-D inputs are supported (matching ORT's constraint on MatMulBnb4).
    if len(input0_shape) != 2:
        raise UnsupportedOpError(
            f"MatMulBnb4 only supports 2-D input A, got shape {input0_shape}"
        )

    output_shape = (m, n)

    expected_output_shape = _value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            f"MatMulBnb4 output shape must be {output_shape}, "
            f"got {expected_output_shape}"
        )

    numel = k * n
    n_blocks = math.ceil(numel / block_size)
    packed_size = math.ceil(numel / 2)

    input0_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    b_dtype = _value_dtype(graph, node.inputs[1], node)
    absmax_dtype = _value_dtype(graph, node.inputs[2], node)

    if not input0_dtype.is_float:
        raise UnsupportedOpError(
            f"MatMulBnb4 input A must be float, got {input0_dtype.onnx_name}"
        )
    if output_dtype != input0_dtype:
        raise UnsupportedOpError(
            f"MatMulBnb4 output dtype ({output_dtype.onnx_name}) "
            f"must match A dtype ({input0_dtype.onnx_name})"
        )

    return MatMulBnb4Op(
        input0=node.inputs[0],
        input1=node.inputs[1],
        absmax=node.inputs[2],
        output=node.outputs[0],
        k=k,
        n=n,
        block_size=block_size,
        quant_type=quant_type,
        input0_shape=input0_shape,
        output_shape=output_shape,
        m=m,
        input0_dtype=input0_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        absmax_dtype=absmax_dtype,
        n_blocks=n_blocks,
        packed_size=packed_size,
    )
