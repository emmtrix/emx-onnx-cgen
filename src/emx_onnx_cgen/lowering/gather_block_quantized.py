from __future__ import annotations

import math

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import GatherBlockQuantizedOp
from .common import (
    optional_name,
    value_dtype as _value_dtype,
    value_shape as _value_shape,
)
from .registry import register_lowering

# Valid data dtypes for GatherBlockQuantized.
# UINT8 with bits=4 is the "packed nibble" case (2 quantized values per byte).
# INT4/UINT4 store values in 4-bit types directly (unpacked).
# All wider integer types (INT8, INT16, INT32, INT64, etc.) store one
# quantized value per element and are never packed.
_VALID_DATA_DTYPES = {
    ScalarType.U4,
    ScalarType.I4,
    ScalarType.U8,
    ScalarType.I8,
    ScalarType.U16,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.U32,
    ScalarType.I64,
    ScalarType.U64,
}

# Valid index dtypes for GatherBlockQuantized.
_VALID_INDEX_DTYPES = {ScalarType.I16, ScalarType.I32, ScalarType.I64}


def _normalize_gather_axis(axis: int, rank: int) -> int:
    """Normalize gather_axis; clamp out-of-range values to 0 (ORT behaviour)."""
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return 0
    return axis


def _normalize_quantize_axis(axis: int, rank: int) -> int:
    """Normalize quantize_axis; clamp out-of-range values to rank-1 (ORT behaviour)."""
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return rank - 1
    return axis


@register_lowering("GatherBlockQuantized")
def lower_gather_block_quantized(graph: Graph, node: Node) -> GatherBlockQuantizedOp:
    if len(node.inputs) not in {3, 4} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "GatherBlockQuantized must have 3 or 4 inputs and 1 output"
        )

    bits = int(node.attrs.get("bits", 0))
    block_size = int(node.attrs.get("block_size", 0))
    gather_axis = int(node.attrs.get("gather_axis", 0))
    quantize_axis = int(node.attrs.get("quantize_axis", 0))

    if bits <= 0:
        raise UnsupportedOpError(
            f"GatherBlockQuantized bits must be > 0, got {bits}"
        )
    if block_size <= 0:
        raise UnsupportedOpError(
            f"GatherBlockQuantized block_size must be > 0, got {block_size}"
        )

    data_shape = _value_shape(graph, node.inputs[0], node)
    indices_shape = _value_shape(graph, node.inputs[1], node)

    data_dtype = _value_dtype(graph, node.inputs[0], node)
    indices_dtype = _value_dtype(graph, node.inputs[1], node)
    scales_dtype = _value_dtype(graph, node.inputs[2], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)

    if data_dtype not in _VALID_DATA_DTYPES:
        raise UnsupportedOpError(
            f"GatherBlockQuantized unsupported data dtype {data_dtype.onnx_name}"
        )
    if indices_dtype not in _VALID_INDEX_DTYPES:
        raise UnsupportedOpError("GatherBlockQuantized indices must be INT16, INT32 or INT64")
    if not scales_dtype.is_float:
        raise UnsupportedOpError("GatherBlockQuantized scales must be float")
    if output_dtype != scales_dtype:
        raise UnsupportedOpError(
            "GatherBlockQuantized output dtype must match scales dtype"
        )

    rank = len(data_shape)
    gather_axis = _normalize_gather_axis(gather_axis, rank)
    quantize_axis = _normalize_quantize_axis(quantize_axis, rank)

    # Determine whether data is packed: only UINT8 storing sub-byte values is
    # packed (2 nibbles per byte for bits=4).  All other integer types store
    # one quantized value per element regardless of bits.
    data_bits = data_dtype.bits
    packed = data_dtype == ScalarType.U8 and bits < data_bits
    values_per_element = data_bits // bits if packed else 1

    # For packed UINT8 data, ORT always gathers on axis 0 regardless of the
    # model's gather_axis attribute.
    if packed and gather_axis != 0:
        gather_axis = 0

    # Logical size along quantize_axis (unpacked).
    logical_quantize_dim = data_shape[quantize_axis] * values_per_element

    # Number of quantization blocks along quantize_axis.
    n_blocks = math.ceil(logical_quantize_dim / block_size)

    zero_point_name = optional_name(node.inputs, 3)
    zero_points_packed = False
    if zero_point_name is not None:
        zero_point_dtype = _value_dtype(graph, zero_point_name, node)
        zero_point_shape = _value_shape(graph, zero_point_name, node)
        if zero_point_dtype != data_dtype:
            raise UnsupportedOpError(
                "GatherBlockQuantized zero_points dtype must match data dtype"
            )
        # When data is packed, zero_points may also be packed.
        if packed:
            expected_zp_shape = list(data_shape)
            expected_zp_shape[quantize_axis] = math.ceil(n_blocks / values_per_element)
            if zero_point_shape == tuple(expected_zp_shape):
                zero_points_packed = True
            # else: leave zero_points_packed=False (unpacked zero_points accepted)

    # Output shape: standard Gather formula on the *logical* (unpacked) shape.
    logical_data_shape = list(data_shape)
    logical_data_shape[quantize_axis] = logical_quantize_dim
    output_shape = (
        tuple(logical_data_shape[:gather_axis])
        + indices_shape
        + tuple(logical_data_shape[gather_axis + 1 :])
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)

    return GatherBlockQuantizedOp(
        data=node.inputs[0],
        indices=node.inputs[1],
        scales=node.inputs[2],
        zero_points=zero_point_name,
        output=node.outputs[0],
        gather_axis=gather_axis,
        quantize_axis=quantize_axis,
        block_size=block_size,
        bits=bits,
        packed=packed,
        values_per_element=values_per_element,
        logical_quantize_dim=logical_quantize_dim,
        n_blocks=n_blocks,
        zero_points_packed=zero_points_packed,
    )
