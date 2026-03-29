from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MatMulNBitsOp
from .common import _find_initializer
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


_SUPPORTED_BITS = {2, 4, 8}


@dataclass(frozen=True)
class MatMulNBitsSpec:
    input0_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    n_blocks_per_col: int
    blob_size: int


def _resolve_matmul_nbits_spec(
    graph: Graph, node: Node, *, bits: int, block_size: int, k: int, n: int
) -> MatMulNBitsSpec:
    input0_shape = _value_shape(graph, node.inputs[0], node)
    if len(input0_shape) < 2:
        raise UnsupportedOpError(
            f"MatMulNBits input A must be at least 2-D, got shape {input0_shape}"
        )
    m = input0_shape[-2]
    k_from_shape = input0_shape[-1]
    if k_from_shape != k:
        raise ShapeInferenceError(
            f"MatMulNBits K attribute ({k}) does not match "
            f"input A last dimension ({k_from_shape})"
        )
    batch_shape = input0_shape[:-2]
    output_shape = batch_shape + (m, n)

    expected_output_shape = _value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            f"MatMulNBits output shape must be {output_shape}, "
            f"got {expected_output_shape}"
        )

    n_blocks_per_col = math.ceil(k / block_size)
    blob_size = block_size * bits // 8
    return MatMulNBitsSpec(
        input0_shape=input0_shape,
        output_shape=output_shape,
        batch_shape=batch_shape,
        m=m,
        n=n,
        k=k,
        n_blocks_per_col=n_blocks_per_col,
        blob_size=blob_size,
    )


def _int_attr(node: Node, name: str) -> int:
    val = node.attrs.get(name)
    if val is None:
        raise UnsupportedOpError(f"MatMulNBits requires attribute '{name}'")
    return int(val)


def _optional_input(node: Node, index: int) -> str | None:
    if index < len(node.inputs) and node.inputs[index]:
        return node.inputs[index]
    return None


def _validate_canonical_g_idx(
    graph: Graph,
    *,
    g_idx: str,
    block_size: int,
    n_blocks_per_col: int,
) -> None:
    initializer = _find_initializer(graph, g_idx)
    if initializer is None:
        raise UnsupportedOpError(
            "MatMulNBits g_idx (input 4) must be a constant initializer"
        )
    if initializer.type.dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            "MatMulNBits g_idx (input 4) must be int32 or int64, "
            f"got {initializer.type.dtype.onnx_name}"
        )

    values = initializer.data.reshape(-1)
    expected_len = n_blocks_per_col * block_size
    if len(values) != expected_len:
        raise UnsupportedOpError(
            "MatMulNBits g_idx (input 4) must have length "
            f"{expected_len}, got {len(values)}"
        )

    for index, value in enumerate(values):
        expected_group = index // block_size
        if int(value) != expected_group:
            raise UnsupportedOpError(
                "MatMulNBits only supports canonical g_idx where each block index "
                f"is repeated {block_size} times; got g_idx[{index}]={int(value)} "
                f"but expected {expected_group}"
            )


@register_lowering("MatMulNBits")
def lower_matmul_nbits(graph: Graph, node: Node) -> MatMulNBitsOp:
    if len(node.inputs) < 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "MatMulNBits must have at least 3 inputs and exactly 1 output"
        )

    bits = _int_attr(node, "bits")
    block_size = _int_attr(node, "block_size")
    k = _int_attr(node, "K")
    n = _int_attr(node, "N")
    accuracy_level = int(node.attrs.get("accuracy_level", 0))

    if bits not in _SUPPORTED_BITS:
        raise UnsupportedOpError(
            f"MatMulNBits supports bits in {sorted(_SUPPORTED_BITS)}, got {bits}"
        )

    spec = _resolve_matmul_nbits_spec(
        graph, node, bits=bits, block_size=block_size, k=k, n=n
    )

    input0_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    b_dtype = _value_dtype(graph, node.inputs[1], node)
    scales_dtype = _value_dtype(graph, node.inputs[2], node)

    if not input0_dtype.is_float:
        raise UnsupportedOpError(
            f"MatMulNBits input A must be float, got {input0_dtype.onnx_name}"
        )
    if output_dtype != input0_dtype:
        raise UnsupportedOpError(
            f"MatMulNBits output dtype ({output_dtype.onnx_name}) "
            f"must match A dtype ({input0_dtype.onnx_name})"
        )

    zero_points = _optional_input(node, 3)
    g_idx = _optional_input(node, 4)
    bias = _optional_input(node, 5)

    if g_idx is not None:
        _validate_canonical_g_idx(
            graph,
            g_idx=g_idx,
            block_size=block_size,
            n_blocks_per_col=spec.n_blocks_per_col,
        )

    zero_points_dtype: ScalarType | None = None
    zero_points_packed = False
    if zero_points is not None:
        zero_points_dtype = _value_dtype(graph, zero_points, node)
        if zero_points_dtype in {ScalarType.U8, ScalarType.I8}:
            zero_points_packed = True
        elif not zero_points_dtype.is_float:
            raise UnsupportedOpError(
                f"MatMulNBits zero_points must be uint8 or float, "
                f"got {zero_points_dtype.onnx_name}"
            )

    bias_dtype: ScalarType | None = None
    if bias is not None:
        bias_dtype = _value_dtype(graph, bias, node)

    return MatMulNBitsOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        scales=node.inputs[2],
        zero_points=zero_points,
        bias=bias,
        output=node.outputs[0],
        bits=bits,
        block_size=block_size,
        k=k,
        n=n,
        accuracy_level=accuracy_level,
        n_blocks_per_col=spec.n_blocks_per_col,
        blob_size=spec.blob_size,
        input0_shape=spec.input0_shape,
        output_shape=spec.output_shape,
        batch_shape=spec.batch_shape,
        m=spec.m,
        input0_dtype=input0_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        scales_dtype=scales_dtype,
        zero_points_dtype=zero_points_dtype,
        zero_points_packed=zero_points_packed,
        bias_dtype=bias_dtype,
    )
