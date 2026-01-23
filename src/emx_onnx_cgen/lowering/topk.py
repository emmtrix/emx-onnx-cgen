from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import TopKOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import shape_product, value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_k(graph: Graph, name: str, node: Node) -> int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} k input must be int64 or int32"
        )
    data = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if data.size != 1:
        raise ShapeInferenceError(
            f"{node.op_type} k input must contain a single value"
        )
    k = int(data[0])
    if k <= 0:
        raise ShapeInferenceError(
            f"{node.op_type} k must be a positive value, got {k}"
        )
    return k


def _topk_dtype_supported(dtype: ScalarType) -> bool:
    return not dtype.is_bool


def lower_topk(graph: Graph, node: Node) -> TopKOp:
    if node.op_type != "TopK":
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if len(node.inputs) != 2 or len(node.outputs) != 2:
        raise UnsupportedOpError(
            f"{node.op_type} must have 2 inputs and 2 outputs"
        )
    input_name = node.inputs[0]
    k_name = node.inputs[1]
    output_values = node.outputs[0]
    output_indices = node.outputs[1]
    input_shape = value_shape(graph, input_name, node)
    shape_product(input_shape)
    axis = int(node.attrs.get("axis", -1))
    axis = normalize_axis(axis, input_shape, node)
    k = _read_k(graph, k_name, node)
    axis_dim = input_shape[axis]
    values_shape = value_shape(graph, output_values, node)
    indices_shape = value_shape(graph, output_indices, node)
    if values_shape != indices_shape:
        raise ShapeInferenceError(
            f"{node.op_type} values and indices output shapes must match, "
            f"got {values_shape} and {indices_shape}"
        )
    if k is None:
        k_shape = value_shape(graph, k_name, node)
        if len(k_shape) != 1 or k_shape[0] != 1:
            raise ShapeInferenceError(
                f"{node.op_type} k input must be a 1-element tensor"
            )
        if axis >= len(values_shape):
            raise ShapeInferenceError(
                f"{node.op_type} axis {axis} exceeds output rank {len(values_shape)}"
            )
        k = values_shape[axis]
        if k <= 0:
            raise ShapeInferenceError(
                f"{node.op_type} k must be a positive value, got {k}"
            )
    if k > axis_dim:
        raise ShapeInferenceError(
            f"{node.op_type} k {k} exceeds axis dimension {axis_dim}"
        )
    output_shape_expected = list(input_shape)
    output_shape_expected[axis] = k
    output_shape = tuple(output_shape_expected)
    if values_shape != output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} values output shape must be {output_shape}, got {values_shape}"
        )
    if indices_shape != output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} indices output shape must be {output_shape}, got {indices_shape}"
        )
    input_dtype = value_dtype(graph, input_name, node)
    if not _topk_dtype_supported(input_dtype):
        raise UnsupportedOpError(
            f"{node.op_type} does not support dtype {input_dtype.onnx_name}"
        )
    values_dtype = value_dtype(graph, output_values, node)
    if values_dtype != input_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} values output dtype must be {input_dtype.onnx_name}"
        )
    indices_dtype = value_dtype(graph, output_indices, node)
    if indices_dtype != ScalarType.I64:
        raise UnsupportedOpError(
            f"{node.op_type} indices output dtype must be int64"
        )
    largest = bool(int(node.attrs.get("largest", 1)))
    sorted_output = bool(int(node.attrs.get("sorted", 1)))
    return TopKOp(
        input0=input_name,
        k_input=k_name,
        output_values=output_values,
        output_indices=output_indices,
        axis=axis,
        k=k,
        largest=largest,
        sorted=sorted_output,
    )


register_lowering("TopK")(lower_topk)
