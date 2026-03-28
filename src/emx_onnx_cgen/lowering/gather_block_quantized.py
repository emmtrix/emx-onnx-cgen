from __future__ import annotations

import math

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import GatherBlockQuantizedOp
from ..validation import normalize_axis
from .common import (
    optional_name,
    value_dtype as _value_dtype,
    value_shape as _value_shape,
)
from .registry import register_lowering

_SUPPORTED_BITS = {4, 8}

_QUANTIZED_DTYPES = {
    ScalarType.U4,
    ScalarType.I4,
    ScalarType.U8,
    ScalarType.I8,
    ScalarType.U16,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.U32,
}


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

    if bits not in _SUPPORTED_BITS:
        raise UnsupportedOpError(
            f"GatherBlockQuantized supports bits in {sorted(_SUPPORTED_BITS)}, "
            f"got {bits}"
        )
    if block_size <= 0:
        raise UnsupportedOpError(
            f"GatherBlockQuantized block_size must be > 0, got {block_size}"
        )

    data_shape = _value_shape(graph, node.inputs[0], node)
    indices_shape = _value_shape(graph, node.inputs[1], node)
    scales_shape = _value_shape(graph, node.inputs[2], node)

    data_dtype = _value_dtype(graph, node.inputs[0], node)
    indices_dtype = _value_dtype(graph, node.inputs[1], node)
    scales_dtype = _value_dtype(graph, node.inputs[2], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)

    if data_dtype not in _QUANTIZED_DTYPES:
        raise UnsupportedOpError(
            f"GatherBlockQuantized data dtype must be integer, "
            f"got {data_dtype.onnx_name}"
        )
    if indices_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError("GatherBlockQuantized indices must be int32 or int64")
    if not scales_dtype.is_float:
        raise UnsupportedOpError("GatherBlockQuantized scales must be float")
    if output_dtype != scales_dtype:
        raise UnsupportedOpError(
            "GatherBlockQuantized output dtype must match scales dtype"
        )

    gather_axis = normalize_axis(gather_axis, data_shape, node)
    quantize_axis = normalize_axis(quantize_axis, data_shape, node)

    # Determine whether data is packed (e.g. UINT8 storing 4-bit values).
    data_bits = data_dtype.bits
    packed = data_bits > bits
    values_per_element = data_bits // bits if packed else 1

    # Logical size along quantize_axis (unpacked).
    logical_quantize_dim = data_shape[quantize_axis] * values_per_element

    # Number of quantization blocks along quantize_axis.
    n_blocks = math.ceil(logical_quantize_dim / block_size)

    # Expected scales shape: same as data except quantize_axis has n_blocks.
    expected_scales_shape = list(data_shape)
    expected_scales_shape[quantize_axis] = n_blocks
    if scales_shape != tuple(expected_scales_shape):
        raise ShapeInferenceError(
            f"GatherBlockQuantized scales shape {scales_shape} does not match "
            f"expected {tuple(expected_scales_shape)} "
            f"(data_shape={data_shape}, quantize_axis={quantize_axis}, "
            f"block_size={block_size}, packed={packed})"
        )

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
            elif zero_point_shape == tuple(expected_scales_shape):
                zero_points_packed = False
            else:
                raise ShapeInferenceError(
                    f"GatherBlockQuantized zero_points shape {zero_point_shape} "
                    f"does not match expected packed {tuple(expected_zp_shape)} "
                    f"or unpacked {tuple(expected_scales_shape)}"
                )
        else:
            if zero_point_shape != scales_shape:
                raise ShapeInferenceError(
                    f"GatherBlockQuantized zero_points shape {zero_point_shape} "
                    f"must match scales shape {scales_shape}"
                )

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
