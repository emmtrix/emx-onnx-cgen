from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import SliceOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@dataclass(frozen=True)
class SliceSpec:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    starts: tuple[int, ...]
    steps: tuple[int, ...]


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_int_list(
    graph: Graph, name: str, node: Node, *, label: str
) -> list[int]:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be a constant initializer"
        )
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be int64 or int32"
        )
    data = np.array(initializer.data, dtype=np.int64).reshape(-1)
    return [int(value) for value in data]


def _resolve_inputs(
    graph: Graph, node: Node
) -> tuple[list[int], list[int], list[int] | None, list[int] | None]:
    if "starts" in node.attrs or "ends" in node.attrs:
        if len(node.inputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} with starts/ends attributes expects 1 input"
            )
        if "starts" not in node.attrs or "ends" not in node.attrs:
            raise UnsupportedOpError(
                f"{node.op_type} must specify both starts and ends"
            )
        starts = [int(value) for value in node.attrs.get("starts", [])]
        ends = [int(value) for value in node.attrs.get("ends", [])]
        axes_attr = node.attrs.get("axes")
        axes = [int(value) for value in axes_attr] if axes_attr else None
        steps = None
        return starts, ends, axes, steps
    if len(node.inputs) < 3:
        raise UnsupportedOpError(
            f"{node.op_type} expects at least 3 inputs"
        )
    starts = _read_int_list(graph, node.inputs[1], node, label="starts")
    ends = _read_int_list(graph, node.inputs[2], node, label="ends")
    axes = None
    steps = None
    if len(node.inputs) >= 4 and node.inputs[3]:
        axes = _read_int_list(graph, node.inputs[3], node, label="axes")
    if len(node.inputs) >= 5 and node.inputs[4]:
        steps = _read_int_list(graph, node.inputs[4], node, label="steps")
    return starts, ends, axes, steps


def _normalize_slices(
    input_shape: tuple[int, ...],
    starts: list[int],
    ends: list[int],
    axes: list[int] | None,
    steps: list[int] | None,
    node: Node,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rank = len(input_shape)
    if rank == 0:
        raise ShapeInferenceError(
            f"{node.op_type} does not support scalar inputs"
        )
    if len(starts) != len(ends):
        raise ShapeInferenceError(
            f"{node.op_type} starts and ends must have matching lengths"
        )
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    if len(axes) != len(starts) or len(steps) != len(starts):
        raise ShapeInferenceError(
            f"{node.op_type} axes and steps must match starts length"
        )
    normalized_starts = [0] * rank
    normalized_steps = [1] * rank
    output_shape = list(input_shape)
    seen_axes: set[int] = set()
    for index, axis in enumerate(axes):
        normalized_axis = normalize_axis(int(axis), input_shape, node)
        if normalized_axis in seen_axes:
            raise ShapeInferenceError(
                f"{node.op_type} axes must be unique"
            )
        seen_axes.add(normalized_axis)
        dim = input_shape[normalized_axis]
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
        step = int(steps[index])
        if step == 0:
            raise UnsupportedOpError(
                f"{node.op_type} steps must be non-zero"
            )
        if step < 0:
            raise UnsupportedOpError(
                f"{node.op_type} only supports positive steps"
            )
        start = int(starts[index])
        end = int(ends[index])
        if start < 0:
            start += dim
        if end < 0:
            end += dim
        start = max(0, min(start, dim))
        end = max(0, min(end, dim))
        if end <= start:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        length = (end - start + step - 1) // step
        if length <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        normalized_starts[normalized_axis] = start
        normalized_steps[normalized_axis] = step
        output_shape[normalized_axis] = length
    return (
        tuple(normalized_starts),
        tuple(normalized_steps),
        tuple(output_shape),
    )


def resolve_slice_spec(graph: Graph, node: Node) -> SliceSpec:
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Slice must have 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    if any(dim < 0 for dim in input_shape):
        raise ShapeInferenceError("Dynamic dims are not supported")
    if any(dim < 0 for dim in output_shape):
        raise ShapeInferenceError("Dynamic dims are not supported")
    starts, ends, axes, steps = _resolve_inputs(graph, node)
    normalized_starts, normalized_steps, computed_output_shape = _normalize_slices(
        input_shape, starts, ends, axes, steps, node
    )
    if output_shape and computed_output_shape != output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be "
            f"{computed_output_shape}, got {output_shape}"
        )
    return SliceSpec(
        input_shape=input_shape,
        output_shape=computed_output_shape,
        starts=normalized_starts,
        steps=normalized_steps,
    )


@register_lowering("Slice")
def lower_slice(graph: Graph, node: Node) -> SliceOp:
    spec = resolve_slice_spec(graph, node)
    dtype = value_dtype(graph, node.inputs[0], node)
    return SliceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=spec.input_shape,
        output_shape=spec.output_shape,
        starts=spec.starts,
        steps=spec.steps,
        dtype=dtype,
        input_dtype=dtype,
    )
