from __future__ import annotations

import numpy as np
from onnx import numpy_helper

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import LabelEncoderOp
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_KEY_DTYPES = {
    ScalarType.STRING,
    ScalarType.I64,
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I32,
    ScalarType.I16,
}

_SUPPORTED_VALUE_DTYPES = {
    ScalarType.STRING,
    ScalarType.I64,
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I32,
    ScalarType.I16,
}


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_strings(values: object | None) -> tuple[str, ...]:
    if values is None:
        return ()
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        return ()
    return tuple(_decode_string(item) for item in items)


def _decode_int64s(values: object | None) -> tuple[int, ...]:
    if values is None:
        return ()
    try:
        return tuple(int(v) for v in values)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _decode_floats(values: object | None) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        return tuple(float(v) for v in values)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _decode_tensor_attr(node: Node, attr_name: str) -> np.ndarray | None:
    attr_val = node.attrs.get(attr_name)
    if attr_val is None:
        return None
    return numpy_helper.to_array(attr_val)


@register_lowering("LabelEncoder")
def lower_label_encoder(graph: Graph, node: Node) -> LabelEncoderOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LabelEncoder must have 1 input and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)

    if input_dtype not in _SUPPORTED_KEY_DTYPES:
        raise UnsupportedOpError(
            f"LabelEncoder input dtype {input_dtype.onnx_name} is not supported"
        )
    if output_dtype not in _SUPPORTED_VALUE_DTYPES:
        raise UnsupportedOpError(
            f"LabelEncoder output dtype {output_dtype.onnx_name} is not supported"
        )

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if input_shape != output_shape:
        raise UnsupportedOpError(
            f"LabelEncoder input and output shapes must match, "
            f"got {input_shape} and {output_shape}"
        )

    keys_strings = _decode_strings(node.attrs.get("keys_strings"))
    keys_int64s = _decode_int64s(node.attrs.get("keys_int64s"))
    keys_floats = _decode_floats(node.attrs.get("keys_floats"))
    values_strings = _decode_strings(node.attrs.get("values_strings"))
    values_int64s = _decode_int64s(node.attrs.get("values_int64s"))
    values_floats = _decode_floats(node.attrs.get("values_floats"))

    keys_tensor = _decode_tensor_attr(node, "keys_tensor")
    values_tensor = _decode_tensor_attr(node, "values_tensor")
    default_tensor = _decode_tensor_attr(node, "default_tensor")

    if keys_tensor is not None and not keys_strings:
        if keys_tensor.dtype == object:
            keys_strings = tuple(str(s) for s in keys_tensor.reshape(-1))
        elif np.issubdtype(keys_tensor.dtype, np.integer):
            keys_int64s = tuple(int(v) for v in keys_tensor.reshape(-1))
        elif np.issubdtype(keys_tensor.dtype, np.floating):
            keys_floats = tuple(float(v) for v in keys_tensor.reshape(-1))

    if (
        values_tensor is not None
        and not values_strings
        and not values_int64s
        and not values_floats
    ):
        if values_tensor.dtype == object:
            values_strings = tuple(str(s) for s in values_tensor.reshape(-1))
        elif np.issubdtype(values_tensor.dtype, np.integer):
            values_int64s = tuple(int(v) for v in values_tensor.reshape(-1))
        elif np.issubdtype(values_tensor.dtype, np.floating):
            values_floats = tuple(float(v) for v in values_tensor.reshape(-1))

    default_string = ""
    default_int64 = -1
    default_float = -0.0

    if default_tensor is not None:
        arr = default_tensor.reshape(-1)
        if arr.size > 0:
            if arr.dtype == object:
                default_string = str(arr[0])
            elif np.issubdtype(arr.dtype, np.integer):
                default_int64 = int(arr[0])
            elif np.issubdtype(arr.dtype, np.floating):
                default_float = float(arr[0])

    raw_default_string = node.attrs.get("default_string")
    if raw_default_string is not None:
        default_string = _decode_string(raw_default_string)
    raw_default_int64 = node.attrs.get("default_int64")
    if raw_default_int64 is not None:
        default_int64 = int(raw_default_int64)
    raw_default_float = node.attrs.get("default_float")
    if raw_default_float is not None:
        default_float = float(raw_default_float)

    num_key_sets = sum(
        [
            len(keys_strings) > 0,
            len(keys_int64s) > 0,
            len(keys_floats) > 0,
        ]
    )
    if num_key_sets == 0:
        raise UnsupportedOpError("LabelEncoder requires at least one set of keys")
    if num_key_sets > 1:
        raise UnsupportedOpError("LabelEncoder requires exactly one set of keys")

    num_value_sets = sum(
        [
            len(values_strings) > 0,
            len(values_int64s) > 0,
            len(values_floats) > 0,
        ]
    )
    if num_value_sets == 0:
        raise UnsupportedOpError("LabelEncoder requires at least one set of values")
    if num_value_sets > 1:
        raise UnsupportedOpError("LabelEncoder requires exactly one set of values")

    if keys_strings:
        num_keys = len(keys_strings)
    elif keys_int64s:
        num_keys = len(keys_int64s)
    else:
        num_keys = len(keys_floats)

    if values_strings:
        num_values = len(values_strings)
    elif values_int64s:
        num_values = len(values_int64s)
    else:
        num_values = len(values_floats)

    if num_keys != num_values:
        raise UnsupportedOpError(
            f"LabelEncoder keys and values must have the same length, "
            f"got {num_keys} and {num_values}"
        )

    if output_dtype == ScalarType.STRING:
        raise UnsupportedOpError("LabelEncoder with string output is not supported")

    if keys_strings and input_dtype != ScalarType.STRING:
        raise UnsupportedOpError("LabelEncoder string keys require string input dtype")

    return LabelEncoderOp(
        input0=input_name,
        output=output_name,
        keys_strings=keys_strings,
        keys_int64s=keys_int64s,
        keys_floats=keys_floats,
        values_strings=values_strings,
        values_int64s=values_int64s,
        values_floats=values_floats,
        default_string=default_string,
        default_int64=default_int64,
        default_float=default_float,
    )
