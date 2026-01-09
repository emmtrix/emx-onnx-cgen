from __future__ import annotations

from typing import Iterable

import onnx

from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Node, TensorType, Value

_ONNX_TO_DTYPE = {
    onnx.TensorProto.FLOAT: "float",
}


def _tensor_type(value_info: onnx.ValueInfoProto) -> TensorType:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ShapeInferenceError(f"Missing elem_type for {value_info.name}")
    dtype = _ONNX_TO_DTYPE.get(tensor_type.elem_type)
    if dtype is None:
        raise UnsupportedOpError(
            f"Unsupported elem_type {tensor_type.elem_type} for {value_info.name}"
        )
    shape = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ShapeInferenceError(f"Dynamic dim for {value_info.name}")
        shape.append(dim.dim_value)
    return TensorType(dtype=dtype, shape=tuple(shape))


def _values(value_infos: Iterable[onnx.ValueInfoProto]) -> tuple[Value, ...]:
    return tuple(Value(name=vi.name, type=_tensor_type(vi)) for vi in value_infos)


def import_onnx(model: onnx.ModelProto) -> Graph:
    graph = model.graph
    nodes = tuple(
        Node(
            op_type=node.op_type,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
        )
        for node in graph.node
    )
    inputs = _values(graph.input)
    outputs = _values(graph.output)
    return Graph(inputs=inputs, outputs=outputs, nodes=nodes)
