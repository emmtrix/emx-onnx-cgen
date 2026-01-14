from __future__ import annotations

from ..codegen.c_emitter import ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import shape_product, value_dtype, value_shape
from .registry import register_lowering


def _normalize_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis > rank:
        raise UnsupportedOpError("Flatten axis must be within input rank")
    return axis


def _flatten_output_shape(
    input_shape: tuple[int, ...], axis: int
) -> tuple[int, int]:
    rank = len(input_shape)
    axis = _normalize_axis(axis, rank)
    if rank == 0:
        return (1, 1)
    for dim in input_shape:
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
    first = shape_product(input_shape[:axis]) if axis else 1
    second = shape_product(input_shape[axis:]) if axis < rank else 1
    return (first, second)


@register_lowering("Flatten")
def lower_flatten(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Flatten must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Flatten expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    axis = int(node.attrs.get("axis", 1))
    output_shape = _flatten_output_shape(input_shape, axis)
    expected_shape = value_shape(graph, node.outputs[0], node)
    if expected_shape and output_shape != expected_shape:
        raise ShapeInferenceError(
            "Flatten output shape must be "
            f"{output_shape}, got {expected_shape}"
        )
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=input_dtype,
        input_dtype=input_dtype,
    )
