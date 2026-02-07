from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import StringNormalizerOp
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_CASE_ACTIONS = {"LOWER", "UPPER", "NONE"}
_SUPPORTED_LOCALES = {"", "c", "en_us", "en-us", "posix"}


def _decode_string(value: object, *, attr_name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise UnsupportedOpError(f"StringNormalizer {attr_name} must be a string")


def _decode_stopwords(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise UnsupportedOpError(
            "StringNormalizer stopwords must be a list of strings"
        ) from exc
    return tuple(_decode_string(item, attr_name="stopwords item") for item in items)


@register_lowering("StringNormalizer")
def lower_string_normalizer(graph: Graph, node: Node) -> StringNormalizerOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("StringNormalizer must have 1 input and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if input_dtype.onnx_name != "string" or output_dtype.onnx_name != "string":
        raise UnsupportedOpError("StringNormalizer supports only string tensors")

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if len(input_shape) not in {1, 2}:
        raise UnsupportedOpError(
            f"StringNormalizer input rank must be 1 or 2, got {len(input_shape)}"
        )
    if len(input_shape) == 2 and input_shape[0] != 1:
        raise ShapeInferenceError("StringNormalizer 2D input must have shape [1, C]")
    if len(output_shape) != len(input_shape):
        raise ShapeInferenceError("StringNormalizer output rank must match input rank")
    if len(output_shape) == 2 and output_shape[0] != 1:
        raise ShapeInferenceError("StringNormalizer 2D output must have shape [1, C]")
    if input_shape and output_shape and output_shape[-1] > input_shape[-1]:
        raise ShapeInferenceError(
            "StringNormalizer output last dimension cannot exceed input last dimension"
        )

    case_change_action = _decode_string(
        node.attrs.get("case_change_action", "NONE"), attr_name="case_change_action"
    )
    if case_change_action not in _SUPPORTED_CASE_ACTIONS:
        raise UnsupportedOpError(
            "StringNormalizer case_change_action must be one of "
            f"{sorted(_SUPPORTED_CASE_ACTIONS)}, got {case_change_action}"
        )

    locale_attr = node.attrs.get("locale")
    if locale_attr is not None:
        locale = _decode_string(locale_attr, attr_name="locale").lower()
        if locale not in _SUPPORTED_LOCALES:
            raise UnsupportedOpError(
                "StringNormalizer locale is not supported (only C/en_US/POSIX)"
            )

    return StringNormalizerOp(
        input0=input_name,
        output=output_name,
        input_shape=input_shape,
        output_shape=output_shape,
        case_change_action=case_change_action,
        is_case_sensitive=bool(int(node.attrs.get("is_case_sensitive", 0))),
        stopwords=_decode_stopwords(node.attrs.get("stopwords")),
    )
