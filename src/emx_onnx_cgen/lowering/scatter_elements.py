from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ScatterElementsOp
from ..validation import normalize_axis
from .common import value_dtype, value_shape
from .registry import register_lowering

_ALLOWED_REDUCTIONS = {"none", "add", "mul", "min", "max"}


@register_lowering("ScatterElements")
def lower_scatter_elements(graph: Graph, node: Node) -> ScatterElementsOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("ScatterElements must have 3 inputs and 1 output")

    data_name, indices_name, updates_name = node.inputs
    output_name = node.outputs[0]

    data_shape = value_shape(graph, data_name, node)
    indices_shape = value_shape(graph, indices_name, node)
    updates_shape = value_shape(graph, updates_name, node)
    output_shape = value_shape(graph, output_name, node)

    if output_shape != data_shape:
        raise ShapeInferenceError(
            "ScatterElements output shape must match data shape, "
            f"got {output_shape} vs {data_shape}"
        )
    if indices_shape != updates_shape:
        raise ShapeInferenceError(
            "ScatterElements indices and updates shapes must match, "
            f"got {indices_shape} and {updates_shape}"
        )
    if len(indices_shape) != len(data_shape):
        raise ShapeInferenceError(
            "ScatterElements indices rank must match data rank, "
            f"got {len(indices_shape)} and {len(data_shape)}"
        )

    axis = normalize_axis(int(node.attrs.get("axis", 0)), data_shape, node)

    data_dtype = value_dtype(graph, data_name, node)
    updates_dtype = value_dtype(graph, updates_name, node)
    if updates_dtype != data_dtype:
        raise UnsupportedOpError(
            "ScatterElements updates dtype must match data dtype, "
            f"got {updates_dtype.onnx_name} vs {data_dtype.onnx_name}"
        )

    indices_dtype = value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            f"ScatterElements indices must be int32 or int64, got {indices_dtype.onnx_name}"
        )

    reduction_attr = node.attrs.get("reduction", "none")
    if isinstance(reduction_attr, bytes):
        reduction = reduction_attr.decode()
    else:
        reduction = str(reduction_attr)
    if reduction not in _ALLOWED_REDUCTIONS:
        raise UnsupportedOpError(
            "ScatterElements reduction must be one of "
            f"{sorted(_ALLOWED_REDUCTIONS)}, got {reduction}"
        )

    return ScatterElementsOp(
        data=data_name,
        indices=indices_name,
        updates=updates_name,
        output=output_name,
        axis=axis,
        reduction=reduction,
    )
