from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import RegexFullMatchOp
from .common import value_dtype, value_shape
from .registry import register_lowering


def _decode_pattern(value: object | None) -> str:
    if value is None:
        raise UnsupportedOpError("RegexFullMatch requires a pattern attribute")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise UnsupportedOpError("RegexFullMatch pattern must be a string")


@register_lowering("RegexFullMatch")
def lower_regex_full_match(graph: Graph, node: Node) -> RegexFullMatchOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("RegexFullMatch must have 1 input and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if input_dtype.onnx_name != "string":
        raise UnsupportedOpError("RegexFullMatch input must be a string tensor")
    if output_dtype.onnx_name != "bool":
        raise UnsupportedOpError("RegexFullMatch output must be a bool tensor")

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if tuple(input_shape) != tuple(output_shape):
        raise ShapeInferenceError(
            f"RegexFullMatch output shape must be {input_shape}, got {output_shape}"
        )

    return RegexFullMatchOp(
        input0=input_name,
        output=output_name,
        pattern=_decode_pattern(node.attrs.get("pattern")),
    )
