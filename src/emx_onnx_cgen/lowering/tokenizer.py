from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import TokenizerOp
from .common import value_dtype, value_shape
from .registry import register_lowering


def _decode_string(value: object, *, attr_name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise UnsupportedOpError(f"Tokenizer {attr_name} must be a string")


def _decode_strings(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise UnsupportedOpError(
            "Tokenizer separators must be a list of strings"
        ) from exc
    return tuple(_decode_string(item, attr_name="separators item") for item in items)


@register_lowering("Tokenizer")
def lower_tokenizer(graph: Graph, node: Node) -> TokenizerOp:
    if len(node.inputs) != 1:
        raise UnsupportedOpError("Tokenizer must have 1 input")
    if len(node.outputs) != 1:
        raise UnsupportedOpError("Tokenizer must have 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]

    input_dtype = value_dtype(graph, input_name, node)
    if input_dtype.onnx_name != "string":
        raise UnsupportedOpError("Tokenizer input must be a string tensor")

    output_dtype = value_dtype(graph, output_name, node)
    if output_dtype.onnx_name != "string":
        raise UnsupportedOpError("Tokenizer output must be a string tensor")

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)

    # Compute input element count (may be 0 for empty tensors)
    input_elem_count = 1
    for dim in input_shape:
        if dim is not None:
            input_elem_count *= dim
        else:
            input_elem_count = -1  # unknown, treat as non-zero
            break

    expected_out_rank = len(input_shape) + 1

    if input_elem_count == 0:
        # Empty input: ORT may produce a degenerate output shape. Accept it as-is
        # as long as the output also has 0 elements.
        output_elem_count = 1
        for dim in output_shape:
            if dim is not None:
                output_elem_count *= dim
            else:
                output_elem_count = -1
                break
        if output_elem_count not in (0, -1):
            raise ShapeInferenceError(
                "Tokenizer output must have 0 elements when input is empty"
            )
    else:
        if len(output_shape) != expected_out_rank:
            raise ShapeInferenceError(
                f"Tokenizer output rank must be input rank + 1 "
                f"(expected {expected_out_rank}, got {len(output_shape)})"
            )
        if tuple(output_shape[:-1]) != tuple(input_shape):
            raise ShapeInferenceError(
                "Tokenizer output leading dimensions must match input shape"
            )
        if output_shape[-1] is None:
            raise UnsupportedOpError(
                "Tokenizer output last dimension must be statically known"
            )

    mark = int(node.attrs.get("mark", 0))
    if mark not in (0, 1):
        raise UnsupportedOpError(f"Tokenizer mark must be 0 or 1, got {mark}")

    mincharnum = int(node.attrs.get("mincharnum", 1))
    if mincharnum < 1:
        raise UnsupportedOpError(f"Tokenizer mincharnum must be >= 1, got {mincharnum}")

    pad_value_raw = node.attrs.get("pad_value", "0xdeadbeaf")
    pad_value = _decode_string(pad_value_raw, attr_name="pad_value")

    separators = _decode_strings(node.attrs.get("separators"))

    tokenexp_raw = node.attrs.get("tokenexp")
    tokenexp = (
        _decode_string(tokenexp_raw, attr_name="tokenexp")
        if tokenexp_raw is not None
        else ""
    )

    return TokenizerOp(
        input0=input_name,
        output=output_name,
        mark=mark,
        mincharnum=mincharnum,
        pad_value=pad_value,
        separators=separators,
        tokenexp=tokenexp,
    )
