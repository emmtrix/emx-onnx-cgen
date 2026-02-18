from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import UniqueOp
from ..lowering.common import value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@register_lowering("Unique")
def lower_unique(graph: Graph, node: Node) -> UniqueOp:
    if len(node.inputs) != 1 or len(node.outputs) != 4:
        raise UnsupportedOpError("Unique must have 1 input and 4 outputs")
    input_name = node.inputs[0]
    y_name, indices_name, inverse_indices_name, counts_name = node.outputs
    input_shape = value_shape(graph, input_name, node)
    y_shape = value_shape(graph, y_name, node)
    indices_shape = value_shape(graph, indices_name, node)
    inverse_shape = value_shape(graph, inverse_indices_name, node)
    counts_shape = value_shape(graph, counts_name, node)
    input_dtype = value_dtype(graph, input_name, node)
    y_dtype = value_dtype(graph, y_name, node)
    indices_dtype = value_dtype(graph, indices_name, node)
    inverse_dtype = value_dtype(graph, inverse_indices_name, node)
    counts_dtype = value_dtype(graph, counts_name, node)
    if y_dtype != input_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} Y output dtype must match input dtype {input_dtype.onnx_name}"
        )
    if indices_dtype.onnx_name != "int64":
        raise UnsupportedOpError(f"{node.op_type} indices output dtype must be int64")
    if inverse_dtype.onnx_name != "int64":
        raise UnsupportedOpError(
            f"{node.op_type} inverse_indices output dtype must be int64"
        )
    if counts_dtype.onnx_name != "int64":
        raise UnsupportedOpError(f"{node.op_type} counts output dtype must be int64")
    sorted_attr = bool(int(node.attrs.get("sorted", 1)))
    axis_attr = node.attrs.get("axis")
    axis: int | None = None
    if axis_attr is not None:
        axis = normalize_axis(int(axis_attr), input_shape, node)
        if len(y_shape) != len(input_shape):
            raise UnsupportedOpError(
                f"{node.op_type} Y rank must match input rank when axis is set"
            )
        for dim_index, (input_dim, output_dim) in enumerate(zip(input_shape, y_shape)):
            if dim_index == axis:
                continue
            if input_dim != output_dim:
                raise UnsupportedOpError(
                    f"{node.op_type} Y shape must match input outside axis {axis}"
                )
        expected = y_shape[axis]
        if indices_shape != (expected,) or counts_shape != (expected,):
            raise UnsupportedOpError(
                f"{node.op_type} indices and counts must have shape ({expected},)"
            )
        if inverse_shape != (input_shape[axis],):
            raise UnsupportedOpError(
                f"{node.op_type} inverse_indices must have shape ({input_shape[axis]},)"
            )
    else:
        if len(y_shape) != 1:
            raise UnsupportedOpError(f"{node.op_type} Y output must be rank-1")
        unique_count = y_shape[0]
        if indices_shape != (unique_count,) or counts_shape != (unique_count,):
            raise UnsupportedOpError(
                f"{node.op_type} indices and counts must have shape ({unique_count},)"
            )
        element_count = 1
        for dim in input_shape:
            element_count *= dim
        if inverse_shape != (element_count,):
            raise UnsupportedOpError(
                f"{node.op_type} inverse_indices must have shape ({element_count},)"
            )
    return UniqueOp(
        input0=input_name,
        y=y_name,
        indices=indices_name,
        inverse_indices=inverse_indices_name,
        counts=counts_name,
        axis=axis,
        sorted=sorted_attr,
    )
