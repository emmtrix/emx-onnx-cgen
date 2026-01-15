from __future__ import annotations

from .errors import ShapeInferenceError
from .ir.model import Node


def normalize_axis(axis: int, shape: tuple[int, ...], node: Node) -> int:
    if not shape:
        raise ShapeInferenceError(f"{node.op_type} does not support scalar inputs")
    rank = len(shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(
            f"{node.op_type} axis {axis} is out of range for rank {rank}"
        )
    return axis


def normalize_concat_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(
            f"Concat axis out of range for rank {rank}: {axis}"
        )
    return axis


def ensure_output_shape_matches_input(
    node: Node,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    if input_shape != output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {input_shape}, got {output_shape}"
        )


def validate_concat_shapes(
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
    axis: int,
) -> int:
    ranks = {len(shape) for shape in input_shapes}
    if len(ranks) != 1:
        raise ShapeInferenceError(
            f"Concat inputs must have matching ranks, got {input_shapes}"
        )
    rank = ranks.pop()
    axis = normalize_concat_axis(axis, rank)
    base_shape = list(input_shapes[0])
    axis_dim = 0
    for shape in input_shapes:
        if len(shape) != rank:
            raise ShapeInferenceError(
                f"Concat inputs must have matching ranks, got {input_shapes}"
            )
        for dim_index, dim in enumerate(shape):
            if dim_index == axis:
                continue
            if dim != base_shape[dim_index]:
                raise ShapeInferenceError(
                    "Concat inputs must match on non-axis dimensions, "
                    f"got {input_shapes}"
                )
        axis_dim += shape[axis]
    base_shape[axis] = axis_dim
    expected_output_shape = tuple(base_shape)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Concat output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return axis
