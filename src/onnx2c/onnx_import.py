from __future__ import annotations

from typing import Iterable

import onnx
from onnx import numpy_helper

from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Initializer, Node, TensorType, Value

_ONNX_TO_DTYPE = {
    onnx.TensorProto.FLOAT: "float",
}


def _format_elem_type(elem_type: int) -> str:
    try:
        name = onnx.TensorProto.DataType.Name(elem_type)
    except ValueError:
        name = "UNKNOWN"
    return f"{elem_type} ({name})"


def _tensor_type(value_info: onnx.ValueInfoProto) -> TensorType:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ShapeInferenceError(f"Missing elem_type for {value_info.name}")
    dtype = _ONNX_TO_DTYPE.get(tensor_type.elem_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(tensor_type.elem_type)} for {value_info.name}. "
            "Supported elem_types: "
            f"{', '.join(_format_elem_type(elem) for elem in _ONNX_TO_DTYPE)}. "
            "Hint: export the model with float32 tensors."
        )
    shape = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ShapeInferenceError(f"Dynamic dim for {value_info.name}")
        shape.append(dim.dim_value)
    return TensorType(dtype=dtype, shape=tuple(shape))


def _values(value_infos: Iterable[onnx.ValueInfoProto]) -> tuple[Value, ...]:
    return tuple(Value(name=vi.name, type=_tensor_type(vi)) for vi in value_infos)


def _initializer(value: onnx.TensorProto) -> Initializer:
    dtype = _ONNX_TO_DTYPE.get(value.data_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(value.data_type)} for initializer {value.name}. "
            "Supported elem_types: "
            f"{', '.join(_format_elem_type(elem) for elem in _ONNX_TO_DTYPE)}. "
            "Hint: export the model with float32 initializers."
        )
    data = numpy_helper.to_array(value)
    if dtype == "float" and data.dtype != "float32":
        data = data.astype("float32", copy=False)
    return Initializer(
        name=value.name,
        type=TensorType(dtype=dtype, shape=tuple(data.shape)),
        data=data,
    )


def import_onnx(model: onnx.ModelProto) -> Graph:
    graph = model.graph
    initializers = tuple(_initializer(value) for value in graph.initializer)
    initializer_names = {initializer.name for initializer in initializers}
    nodes = tuple(
        Node(
            op_type=node.op_type,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
        )
        for node in graph.node
    )
    inputs = _values(
        value_info for value_info in graph.input if value_info.name not in initializer_names
    )
    outputs = _values(graph.output)
    return Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers,
    )
