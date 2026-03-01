from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import StringSplitOp
from .common import value_dtype, value_shape
from .registry import register_lowering


def _decode_string_attr(value: object | None, *, attr_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise UnsupportedOpError(f"StringSplit {attr_name} must be a string")


@register_lowering("StringSplit")
def lower_string_split(graph: Graph, node: Node) -> StringSplitOp:
    if len(node.inputs) != 1:
        raise UnsupportedOpError("StringSplit must have 1 input")
    if len(node.outputs) != 2:
        raise UnsupportedOpError("StringSplit must have 2 outputs")

    input_name = node.inputs[0]
    output_y_name = node.outputs[0]
    output_z_name = node.outputs[1]

    input_dtype = value_dtype(graph, input_name, node)
    if input_dtype.onnx_name != "string":
        raise UnsupportedOpError("StringSplit input must be a string tensor")

    output_y_dtype = value_dtype(graph, output_y_name, node)
    output_z_dtype = value_dtype(graph, output_z_name, node)
    if output_y_dtype.onnx_name != "string":
        raise UnsupportedOpError("StringSplit first output must be a string tensor")
    if output_z_dtype.onnx_name != "int64":
        raise UnsupportedOpError("StringSplit second output must be an int64 tensor")

    input_shape = value_shape(graph, input_name, node)
    output_y_shape = value_shape(graph, output_y_name, node)
    output_z_shape = value_shape(graph, output_z_name, node)

    expected_y_rank = len(input_shape) + 1
    if len(output_y_shape) != expected_y_rank:
        raise ShapeInferenceError(
            f"StringSplit output Y rank must be input rank + 1 "
            f"(expected {expected_y_rank}, got {len(output_y_shape)})"
        )
    if output_y_shape[-1] is None:
        raise UnsupportedOpError(
            "StringSplit output last dimension must be statically known"
        )
    if tuple(output_y_shape[:-1]) != tuple(input_shape):
        raise ShapeInferenceError(
            "StringSplit output Y shape must match input shape except for last dim"
        )
    if tuple(output_z_shape) != tuple(input_shape):
        raise ShapeInferenceError(
            "StringSplit output Z shape must match input shape"
        )

    delimiter = _decode_string_attr(node.attrs.get("delimiter"), attr_name="delimiter")

    maxsplit_raw = node.attrs.get("maxsplit")
    if maxsplit_raw is None:
        maxsplit = -1
    else:
        maxsplit = int(maxsplit_raw)
        if maxsplit < 0:
            raise UnsupportedOpError(
                f"StringSplit maxsplit must be non-negative, got {maxsplit}"
            )

    return StringSplitOp(
        input0=input_name,
        output_y=output_y_name,
        output_z=output_z_name,
        delimiter=delimiter,
        maxsplit=maxsplit,
    )
