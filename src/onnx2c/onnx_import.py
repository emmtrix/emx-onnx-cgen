from __future__ import annotations

from typing import Iterable

import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference

from .dtypes import ONNX_TO_DTYPE
from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Initializer, Node, TensorType, Value


def _normalize_initializer_data(dtype: str, data: object) -> np.ndarray:
    if isinstance(data, (onnx.TensorProto, onnx.SparseTensorProto)):
        array = numpy_helper.to_array(data)
    elif isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data)
    if dtype == "float" and array.dtype != "float32":
        array = array.astype("float32", copy=False)
    if dtype == "bool" and array.dtype != "bool":
        array = array.astype("bool", copy=False)
    if dtype == "int64" and array.dtype != "int64":
        array = array.astype("int64", copy=False)
    if dtype == "int32" and array.dtype != "int32":
        array = array.astype("int32", copy=False)
    if dtype == "int16" and array.dtype != "int16":
        array = array.astype("int16", copy=False)
    if dtype == "int8" and array.dtype != "int8":
        array = array.astype("int8", copy=False)
    return array


def _format_elem_type(elem_type: int) -> str:
    try:
        name = onnx.TensorProto.DataType.Name(elem_type)
    except ValueError:
        name = "UNKNOWN"
    return f"{elem_type} ({name})"


def _tensor_type(value_info: onnx.ValueInfoProto) -> TensorType:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ShapeInferenceError(f"Missing elem_type for tensor '{value_info.name}'")
    dtype = ONNX_TO_DTYPE.get(tensor_type.elem_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(tensor_type.elem_type)} for tensor '{value_info.name}'."
        )
    shape = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ShapeInferenceError(f"Dynamic dim for tensor '{value_info.name}'")
        shape.append(dim.dim_value)
    return TensorType(dtype=dtype, shape=tuple(shape))


def _values(value_infos: Iterable[onnx.ValueInfoProto]) -> tuple[Value, ...]:
    return tuple(Value(name=vi.name, type=_tensor_type(vi)) for vi in value_infos)


def _initializer(value: onnx.TensorProto) -> Initializer:
    dtype = ONNX_TO_DTYPE.get(value.data_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(value.data_type)} for initializer '{value.name}'. "
            "Hint: export the model with float32 initializers."
        )
    data = _normalize_initializer_data(dtype, value)
    return Initializer(
        name=value.name,
        type=TensorType(dtype=dtype, shape=tuple(data.shape)),
        data=data,
    )


def _node_attrs(node: onnx.NodeProto) -> dict[str, object]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _constant_initializer(node: onnx.NodeProto) -> Initializer:
    if len(node.output) != 1:
        raise UnsupportedOpError("Constant must have exactly one output")
    attrs = _node_attrs(node)
    output_name = node.output[0]
    if "value" in attrs:
        tensor = attrs["value"]
        dtype = ONNX_TO_DTYPE.get(tensor.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(dtype=dtype, shape=tuple(data.shape)),
            data=data,
        )
    if "sparse_value" in attrs:
        tensor = attrs["sparse_value"]
        dtype = ONNX_TO_DTYPE.get(tensor.values.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.values.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(dtype=dtype, shape=tuple(data.shape)),
            data=data,
        )
    if "value_float" in attrs or "value_floats" in attrs:
        values = attrs.get("value_floats", attrs.get("value_float"))
        data = _normalize_initializer_data("float", values)
        return Initializer(
            name=output_name,
            type=TensorType(dtype="float", shape=tuple(data.shape)),
            data=data,
        )
    if "value_int" in attrs or "value_ints" in attrs:
        values = attrs.get("value_ints", attrs.get("value_int"))
        data = _normalize_initializer_data("int64", values)
        return Initializer(
            name=output_name,
            type=TensorType(dtype="int64", shape=tuple(data.shape)),
            data=data,
        )
    if "value_string" in attrs or "value_strings" in attrs:
        raise UnsupportedOpError(
            f"Constant '{output_name}' has unsupported string values"
        )
    raise UnsupportedOpError(f"Constant '{output_name}' requires a value attribute")


def import_onnx(model: onnx.ModelProto) -> Graph:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # pragma: no cover - onnx inference errors
        raise ShapeInferenceError("ONNX shape inference failed") from exc
    graph = model.graph
    base_initializers = [_initializer(value) for value in graph.initializer]
    constant_initializers: list[Initializer] = []
    input_names = {value_info.name for value_info in graph.input}
    output_names = {value_info.name for value_info in graph.output}
    nodes: list[Node] = []
    for node in graph.node:
        if node.op_type == "Constant":
            constant_initializers.append(_constant_initializer(node))
            continue
        nodes.append(
            Node(
                op_type=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attrs=_node_attrs(node),
            )
        )
    initializers = tuple(base_initializers + constant_initializers)
    initializer_names = {initializer.name for initializer in initializers}
    inputs = _values(
        value_info for value_info in graph.input if value_info.name not in initializer_names
    )
    outputs = _values(graph.output)
    values = _values(
        value_info
        for value_info in graph.value_info
        if value_info.name
        not in initializer_names | input_names | output_names
    )
    return Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers,
        values=values,
    )
