from __future__ import annotations

from typing import Iterable, Mapping

import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference

from shared.scalar_types import ScalarType

from .dtypes import scalar_type_from_onnx
from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Initializer, Node, TensorType, Value


def _normalize_initializer_data(dtype: ScalarType, data: object) -> np.ndarray:
    if isinstance(data, (onnx.TensorProto, onnx.SparseTensorProto)):
        array = numpy_helper.to_array(data)
    elif isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data)
    return array.astype(dtype.np_dtype, copy=False)


def _format_elem_type(elem_type: int) -> str:
    try:
        name = onnx.TensorProto.DataType.Name(elem_type)
    except ValueError:
        name = "UNKNOWN"
    return f"{elem_type} ({name})"


def _unsupported_value_type(value_info: onnx.ValueInfoProto) -> UnsupportedOpError:
    value_kind = value_info.type.WhichOneof("value")
    if value_kind is None:
        value_kind = "unknown"
    return UnsupportedOpError(
        f"Unsupported value type '{value_kind}' for '{value_info.name}'. "
        "Hint: export the model with tensor inputs/outputs."
    )


def _tensor_type(
    value_info: onnx.ValueInfoProto,
    *,
    dim_param_override: tuple[str | None, ...] | None = None,
) -> TensorType:
    if value_info.type.WhichOneof("value") != "tensor_type":
        raise _unsupported_value_type(value_info)
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ShapeInferenceError(f"Missing elem_type for tensor '{value_info.name}'")
    dtype = scalar_type_from_onnx(tensor_type.elem_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(tensor_type.elem_type)} for tensor '{value_info.name}'."
        )
    shape = []
    dim_params = []
    for dim_index, dim in enumerate(tensor_type.shape.dim):
        dim_param = dim.dim_param if dim.HasField("dim_param") else ""
        if (
            dim_param_override is not None
            and dim_index < len(dim_param_override)
            and dim_param_override[dim_index]
        ):
            dim_param = dim_param_override[dim_index] or ""
        dim_params.append(dim_param or None)
        if not dim.HasField("dim_value"):
            if dim_param:
                shape.append(1)
                continue
            raise ShapeInferenceError(f"Dynamic dim for tensor '{value_info.name}'")
        shape.append(dim.dim_value)
    return TensorType(
        dtype=dtype,
        shape=tuple(shape),
        dim_params=tuple(dim_params),
    )


def _values(
    value_infos: Iterable[onnx.ValueInfoProto],
    *,
    dim_param_by_name: Mapping[str, tuple[str | None, ...]] | None = None,
) -> tuple[Value, ...]:
    dim_param_by_name = dim_param_by_name or {}
    return tuple(
        Value(
            name=vi.name,
            type=_tensor_type(
                vi, dim_param_override=dim_param_by_name.get(vi.name)
            ),
        )
        for vi in value_infos
    )


def _collect_dim_params(
    value_infos: Iterable[onnx.ValueInfoProto],
) -> dict[str, tuple[str | None, ...]]:
    dim_params: dict[str, tuple[str | None, ...]] = {}
    for value_info in value_infos:
        dims = []
        for dim in value_info.type.tensor_type.shape.dim:
            dim_param = dim.dim_param if dim.HasField("dim_param") else ""
            dims.append(dim_param or None)
        if any(dims):
            dim_params[value_info.name] = tuple(dims)
    return dim_params


def _value_info_complete(value_info: onnx.ValueInfoProto) -> bool:
    if value_info.type.WhichOneof("value") != "tensor_type":
        return False
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return False
    if not tensor_type.HasField("shape"):
        return False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            continue
        if dim.HasField("dim_param"):
            continue
        return False
    return True


def _needs_shape_inference(model: onnx.ModelProto) -> bool:
    graph = model.graph
    value_info_by_name = {
        value_info.name: value_info for value_info in graph.value_info
    }
    output_names = {value_info.name for value_info in graph.output}
    initializer_names = {initializer.name for initializer in graph.initializer}
    initializer_names.update(
        sparse_init.name for sparse_init in graph.sparse_initializer
    )
    for node in graph.node:
        for output in node.output:
            if not output:
                continue
            if output in output_names or output in value_info_by_name:
                continue
            return True
    for value_info in graph.value_info:
        if not _value_info_complete(value_info):
            return True
    for value_info in graph.output:
        if not _value_info_complete(value_info):
            return True
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        if not _value_info_complete(value_info):
            return True
    return False


def _initializer(value: onnx.TensorProto) -> Initializer:
    dtype = scalar_type_from_onnx(value.data_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(value.data_type)} for initializer '{value.name}'. "
            "Hint: export the model with float32 initializers."
        )
    data = _normalize_initializer_data(dtype, value)
    return Initializer(
        name=value.name,
        type=TensorType(
            dtype=dtype,
            shape=tuple(data.shape),
            dim_params=(None,) * len(data.shape),
        ),
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
        dtype = scalar_type_from_onnx(tensor.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=dtype,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "sparse_value" in attrs:
        tensor = attrs["sparse_value"]
        dtype = scalar_type_from_onnx(tensor.values.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.values.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=dtype,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "value_float" in attrs or "value_floats" in attrs:
        values = attrs.get("value_floats", attrs.get("value_float"))
        data = _normalize_initializer_data(ScalarType.F32, values)
        return Initializer(
            name=output_name,
        type=TensorType(
            dtype=ScalarType.F32,
            shape=tuple(data.shape),
            dim_params=(None,) * len(data.shape),
        ),
        data=data,
    )
    if "value_int" in attrs or "value_ints" in attrs:
        values = attrs.get("value_ints", attrs.get("value_int"))
        data = _normalize_initializer_data(ScalarType.I64, values)
        return Initializer(
            name=output_name,
        type=TensorType(
            dtype=ScalarType.I64,
            shape=tuple(data.shape),
            dim_params=(None,) * len(data.shape),
        ),
        data=data,
    )
    if "value_string" in attrs or "value_strings" in attrs:
        raise UnsupportedOpError(
            f"Constant '{output_name}' has unsupported string values"
        )
    raise UnsupportedOpError(f"Constant '{output_name}' requires a value attribute")


def import_onnx(model: onnx.ModelProto) -> Graph:
    dim_param_by_name = _collect_dim_params(
        tuple(model.graph.input) + tuple(model.graph.output)
    )
    opset_imports = tuple(
        (opset.domain, opset.version) for opset in model.opset_import
    )
    if _needs_shape_inference(model):
        try:
            model = shape_inference.infer_shapes(model, data_prop=True)
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
                name=node.name or None,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attrs=_node_attrs(node),
            )
        )
    initializers = tuple(base_initializers + constant_initializers)
    initializer_names = {initializer.name for initializer in initializers}
    inputs = _values(
        (
            value_info
            for value_info in graph.input
            if value_info.name not in initializer_names
        ),
        dim_param_by_name=dim_param_by_name,
    )
    outputs = _values(graph.output, dim_param_by_name=dim_param_by_name)
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
        opset_imports=opset_imports,
    )
