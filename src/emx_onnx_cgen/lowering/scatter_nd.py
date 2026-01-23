from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import ScatterNDOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering

_ALLOWED_REDUCTIONS = {"none", "add", "mul", "min", "max"}


@register_lowering("ScatterND")
def lower_scatternd(graph: Graph, node: Node) -> ScatterNDOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("ScatterND must have 3 inputs and 1 output")
    data_name, indices_name, updates_name = node.inputs
    output_name = node.outputs[0]
    data_shape = value_shape(graph, data_name, node)
    indices_shape = value_shape(graph, indices_name, node)
    updates_shape = value_shape(graph, updates_name, node)
    output_shape = value_shape(graph, output_name, node)
    if output_shape != data_shape:
        raise ShapeInferenceError(
            "ScatterND output shape must match data shape, "
            f"got {output_shape} vs {data_shape}"
        )
    if len(indices_shape) < 1:
        raise ShapeInferenceError("ScatterND indices must have rank >= 1")
    index_depth = indices_shape[-1]
    if index_depth <= 0:
        raise ShapeInferenceError(
            "ScatterND indices final dimension must be >= 1"
        )
    if index_depth > len(data_shape):
        raise ShapeInferenceError(
            "ScatterND indices final dimension must be <= data rank, "
            f"got {index_depth} vs {len(data_shape)}"
        )
    expected_updates_shape = indices_shape[:-1] + data_shape[index_depth:]
    if updates_shape != expected_updates_shape:
        raise ShapeInferenceError(
            "ScatterND updates shape must be "
            f"{expected_updates_shape}, got {updates_shape}"
        )
    data_dtype = value_dtype(graph, data_name, node)
    updates_dtype = value_dtype(graph, updates_name, node)
    if updates_dtype != data_dtype:
        raise UnsupportedOpError(
            "ScatterND updates dtype must match data dtype, "
            f"got {updates_dtype.onnx_name} vs {data_dtype.onnx_name}"
        )
    indices_dtype = value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "ScatterND indices must be int32 or int64, "
            f"got {indices_dtype.onnx_name}"
        )
    reduction_attr = node.attrs.get("reduction", "none")
    if isinstance(reduction_attr, bytes):
        reduction = reduction_attr.decode()
    else:
        reduction = str(reduction_attr)
    if reduction not in _ALLOWED_REDUCTIONS:
        raise UnsupportedOpError(
            "ScatterND reduction must be one of "
            f"{sorted(_ALLOWED_REDUCTIONS)}, got {reduction}"
        )
    return ScatterNDOp(
        data=data_name,
        indices=indices_name,
        updates=updates_name,
        output=output_name,
        data_shape=data_shape,
        indices_shape=indices_shape,
        updates_shape=updates_shape,
        output_shape=output_shape,
        reduction=reduction,
        dtype=data_dtype,
        indices_dtype=indices_dtype,
    )
