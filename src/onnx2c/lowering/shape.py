from __future__ import annotations

from ..codegen.c_emitter import ShapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


def _value_shape(graph: Graph, name: str, node: Node) -> tuple[int, ...]:
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _value_dtype(graph: Graph, name: str, node: Node) -> str:
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _normalize_slice_bounds(
    rank: int, *, start: int | None, end: int | None
) -> tuple[int, int]:
    normalized_start = 0 if start is None else int(start)
    normalized_end = rank if end is None else int(end)
    if normalized_start < 0:
        normalized_start += rank
    if normalized_end < 0:
        normalized_end += rank
    normalized_start = max(0, min(normalized_start, rank))
    normalized_end = max(0, min(normalized_end, rank))
    return normalized_start, normalized_end


@register_lowering("Shape")
def lower_shape(graph: Graph, node: Node) -> ShapeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Shape must have 1 input and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 1:
        raise ShapeInferenceError("Shape output must be 1D")
    if output_shape[0] <= 0:
        raise ShapeInferenceError("Shape output length must be positive")
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if output_dtype != "int64":
        raise UnsupportedOpError("Shape output dtype must be int64")
    start = node.attrs.get("start")
    end = node.attrs.get("end")
    start_index, end_index = _normalize_slice_bounds(
        len(input_shape), start=start, end=end
    )
    if end_index <= start_index:
        raise ShapeInferenceError("Shape start must be less than end")
    expected_shape = (end_index - start_index,)
    if expected_shape != output_shape:
        raise ShapeInferenceError(
            "Shape output shape must be "
            f"{expected_shape}, got {output_shape}"
        )
    return ShapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        values=input_shape[start_index:end_index],
        dtype=output_dtype,
        input_dtype=input_dtype,
    )
