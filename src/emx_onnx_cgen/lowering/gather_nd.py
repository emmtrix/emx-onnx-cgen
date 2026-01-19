from __future__ import annotations

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import GatherNDOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("GatherND")
def lower_gather_nd(graph: Graph, node: Node) -> GatherNDOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GatherND must have 2 inputs and 1 output")
    data_name, indices_name = node.inputs
    output_name = node.outputs[0]
    data_shape = _value_shape(graph, data_name, node)
    indices_shape = _value_shape(graph, indices_name, node)
    output_shape = _value_shape(graph, output_name, node)
    if len(indices_shape) < 1:
        raise ShapeInferenceError("GatherND indices must have rank >= 1")
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims < 0:
        raise ShapeInferenceError(
            f"GatherND batch_dims must be >= 0, got {batch_dims}"
        )
    if batch_dims > len(indices_shape) - 1:
        raise ShapeInferenceError(
            "GatherND batch_dims must be <= indices rank - 1, "
            f"got {batch_dims} vs {len(indices_shape) - 1}"
        )
    if batch_dims > len(data_shape):
        raise ShapeInferenceError(
            "GatherND batch_dims must be <= data rank, "
            f"got {batch_dims} vs {len(data_shape)}"
        )
    if tuple(data_shape[:batch_dims]) != tuple(indices_shape[:batch_dims]):
        raise ShapeInferenceError(
            "GatherND batch_dims must match on data/indices, "
            f"got {data_shape} vs {indices_shape}"
        )
    index_depth = indices_shape[-1]
    if index_depth <= 0:
        raise ShapeInferenceError(
            "GatherND indices final dimension must be >= 1"
        )
    if index_depth > len(data_shape) - batch_dims:
        raise ShapeInferenceError(
            "GatherND indices final dimension must be <= data rank - "
            f"batch_dims, got {index_depth} vs {len(data_shape) - batch_dims}"
        )
    expected_output_shape = indices_shape[:-1] + data_shape[
        batch_dims + index_depth :
    ]
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "GatherND output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    data_dtype = _value_dtype(graph, data_name, node)
    indices_dtype = _value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "GatherND indices must be int32 or int64, "
            f"got {indices_dtype.onnx_name}"
        )
    return GatherNDOp(
        data=data_name,
        indices=indices_name,
        output=output_name,
        batch_dims=batch_dims,
        data_shape=data_shape,
        indices_shape=indices_shape,
        output_shape=output_shape,
        dtype=data_dtype,
        indices_dtype=indices_dtype,
    )
