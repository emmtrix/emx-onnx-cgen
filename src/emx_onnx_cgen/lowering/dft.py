from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import DFTOp
from ..lowering.common import optional_name, value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_DFT_DTYPES = {
    ScalarType.F32,
    ScalarType.F64,
}


def _find_initializer(graph: Graph | GraphContext, name: str) -> Initializer | None:
    if isinstance(graph, GraphContext):
        return graph.initializer(name)
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_scalar_int_initializer(
    graph: Graph | GraphContext,
    name: str,
    node: Node,
    label: str,
) -> int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be int32 or int64, "
            f"got {initializer.type.dtype.onnx_name}"
        )
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if values.size != 1:
        raise UnsupportedOpError(f"{node.op_type} {label} input must be a scalar")
    return int(values[0])


def _normalize_dft_axis(axis: int, input_shape: tuple[int, ...], node: Node) -> int:
    rank = len(input_shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank - 1:
        raise ShapeInferenceError(
            f"{node.op_type} axis {axis} is out of range for rank {rank}. "
            "The last axis stores complex lanes and cannot be transformed."
        )
    return axis


@register_lowering("DFT")
def lower_dft(graph: Graph | GraphContext, node: Node) -> DFTOp:
    if len(node.inputs) < 1 or len(node.inputs) > 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("DFT must have 1 to 3 inputs and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if len(input_shape) < 2:
        raise ShapeInferenceError("DFT input rank must be at least 2")
    if len(output_shape) != len(input_shape):
        raise ShapeInferenceError(
            f"DFT output rank must match input rank, got {len(output_shape)} and {len(input_shape)}"
        )
    if any(dim < 0 for dim in input_shape + output_shape):
        raise ShapeInferenceError("DFT does not support dynamic dimensions")

    input_complex_lanes = input_shape[-1]
    if input_complex_lanes not in {1, 2}:
        raise ShapeInferenceError(
            f"DFT input last dimension must be 1 or 2, got {input_complex_lanes}"
        )
    if output_shape[-1] != 2:
        raise ShapeInferenceError(
            f"DFT output last dimension must be 2, got {output_shape[-1]}"
        )

    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "DFT expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if input_dtype not in _SUPPORTED_DFT_DTYPES:
        raise UnsupportedOpError(
            f"DFT supports only float and double, got {input_dtype.onnx_name}"
        )

    inverse = bool(int(node.attrs.get("inverse", 0)))
    onesided = bool(int(node.attrs.get("onesided", 0)))
    if onesided and input_complex_lanes != 1:
        raise UnsupportedOpError(
            "DFT onesided output requires real input (last dim = 1)"
        )

    axis_value = int(node.attrs.get("axis", -2))
    axis_name = optional_name(node.inputs, 2)
    if axis_name is not None:
        axis_const = _read_scalar_int_initializer(graph, axis_name, node, "axis")
        if axis_const is None:
            raise UnsupportedOpError(
                "DFT axis input must be a constant scalar for code generation"
            )
        axis_value = axis_const
    axis = _normalize_dft_axis(axis_value, input_shape, node)

    dft_length_name = optional_name(node.inputs, 1)
    if dft_length_name is None:
        dft_length = input_shape[axis]
    else:
        dft_length_const = _read_scalar_int_initializer(
            graph, dft_length_name, node, "dft_length"
        )
        if dft_length_const is None:
            raise UnsupportedOpError(
                "DFT dft_length input must be a constant scalar for code generation"
            )
        dft_length = dft_length_const
    if dft_length <= 0:
        raise ShapeInferenceError(f"DFT dft_length must be > 0, got {dft_length}")

    expected_axis_dim = dft_length // 2 + 1 if onesided else dft_length
    if output_shape[axis] != expected_axis_dim:
        raise ShapeInferenceError(
            f"DFT output axis dimension must be {expected_axis_dim}, got {output_shape[axis]}"
        )
    for index, (input_dim, output_dim) in enumerate(zip(input_shape, output_shape)):
        if index in {axis, len(input_shape) - 1}:
            continue
        if input_dim != output_dim:
            raise ShapeInferenceError(
                "DFT input/output dimensions must match outside transform axis, "
                f"got input {input_shape} and output {output_shape}"
            )

    return DFTOp(
        input0=input_name,
        output=output_name,
        axis=axis,
        dft_length=dft_length,
        inverse=inverse,
        onesided=onesided,
        input_is_complex=input_complex_lanes == 2,
    )
