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


@dataclass(frozen=True)
class SliceInputs:
    starts: list[int] | None
    ends: list[int] | None
    axes: list[int] | None
    steps: list[int] | None
    starts_input: str | None
    ends_input: str | None
    axes_input: str | None
    steps_input: str | None
    starts_shape: tuple[int, ...] | None
    ends_shape: tuple[int, ...] | None
    axes_shape: tuple[int, ...] | None
    steps_shape: tuple[int, ...] | None
    starts_dtype: ScalarType | None
    ends_dtype: ScalarType | None
    axes_dtype: ScalarType | None
    steps_dtype: ScalarType | None


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


def _maybe_read_int_list(
    graph: Graph, name: str, node: Node, *, label: str
) -> list[int] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    return _read_int_list(graph, name, node, label=label)


def _validate_int_input(
    graph: Graph, name: str, node: Node, *, label: str
) -> tuple[tuple[int, ...], ScalarType]:
    dtype = value_dtype(graph, name, node)
    if dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be int64 or int32"
        )
    shape = value_shape(graph, name, node)
    if len(shape) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be a 1D tensor"
        )
    return shape, dtype


def _resolve_inputs(
    graph: Graph, node: Node
) -> SliceInputs:
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
        return SliceInputs(
            starts=starts,
            ends=ends,
            axes=axes,
            steps=steps,
            starts_input=None,
            ends_input=None,
            axes_input=None,
            steps_input=None,
            starts_shape=None,
            ends_shape=None,
            axes_shape=None,
            steps_shape=None,
            starts_dtype=None,
            ends_dtype=None,
            axes_dtype=None,
            steps_dtype=None,
        )
    if len(node.inputs) < 3:
        raise UnsupportedOpError(
            f"{node.op_type} expects at least 3 inputs"
        )
    starts_name = node.inputs[1]
    ends_name = node.inputs[2]
    axes_name = node.inputs[3] if len(node.inputs) >= 4 else ""
    steps_name = node.inputs[4] if len(node.inputs) >= 5 else ""
    starts = _maybe_read_int_list(graph, starts_name, node, label="starts")
    ends = _maybe_read_int_list(graph, ends_name, node, label="ends")
    axes = (
        _maybe_read_int_list(graph, axes_name, node, label="axes")
        if axes_name
        else None
    )
    steps = (
        _maybe_read_int_list(graph, steps_name, node, label="steps")
        if steps_name
        else None
    )
    if starts is not None and ends is not None:
        return SliceInputs(
            starts=starts,
            ends=ends,
            axes=axes,
            steps=steps,
            starts_input=None,
            ends_input=None,
            axes_input=None,
            steps_input=None,
            starts_shape=None,
            ends_shape=None,
            axes_shape=None,
            steps_shape=None,
            starts_dtype=None,
            ends_dtype=None,
            axes_dtype=None,
            steps_dtype=None,
        )
    if starts is None or ends is None:
        starts_shape, starts_dtype = _validate_int_input(
            graph, starts_name, node, label="starts"
        )
        ends_shape, ends_dtype = _validate_int_input(
            graph, ends_name, node, label="ends"
        )
        if starts_shape != ends_shape:
            raise ShapeInferenceError(
                f"{node.op_type} starts and ends must have matching shapes"
            )
        axes_shape = None
        axes_dtype = None
        steps_shape = None
        steps_dtype = None
        axes_input = None
        steps_input = None
        if axes_name:
            axes_shape, axes_dtype = _validate_int_input(
                graph, axes_name, node, label="axes"
            )
            if axes_shape != starts_shape:
                raise ShapeInferenceError(
                    f"{node.op_type} axes must match starts length"
                )
            axes_input = axes_name
        if steps_name:
            steps_shape, steps_dtype = _validate_int_input(
                graph, steps_name, node, label="steps"
            )
            if steps_shape != starts_shape:
                raise ShapeInferenceError(
                    f"{node.op_type} steps must match starts length"
                )
            steps_input = steps_name
        return SliceInputs(
            starts=None,
            ends=None,
            axes=None,
            steps=None,
            starts_input=starts_name,
            ends_input=ends_name,
            axes_input=axes_input,
            steps_input=steps_input,
            starts_shape=starts_shape,
            ends_shape=ends_shape,
            axes_shape=axes_shape,
            steps_shape=steps_shape,
            starts_dtype=starts_dtype,
            ends_dtype=ends_dtype,
            axes_dtype=axes_dtype,
            steps_dtype=steps_dtype,
        )
    raise UnsupportedOpError(
        f"{node.op_type} starts and ends inputs must both be constant initializers"
    )


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
        length = max(0, (end - start + step - 1) // step)
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
    inputs = _resolve_inputs(graph, node)
    if inputs.starts is None or inputs.ends is None:
        raise UnsupportedOpError(
            f"{node.op_type} starts/ends inputs must be constant for shape "
            "inference"
        )
    starts = inputs.starts
    ends = inputs.ends
    axes = inputs.axes
    steps = inputs.steps
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
    inputs = _resolve_inputs(graph, node)
    if inputs.starts is not None and inputs.ends is not None:
        normalized_starts, normalized_steps, computed_output_shape = _normalize_slices(
            input_shape, inputs.starts, inputs.ends, inputs.axes, inputs.steps, node
        )
        if output_shape and computed_output_shape != output_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be "
                f"{computed_output_shape}, got {output_shape}"
            )
        return SliceOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            input_shape=input_shape,
            output_shape=computed_output_shape,
            starts=normalized_starts,
            steps=normalized_steps,
            axes=None,
            starts_input=None,
            ends_input=None,
            axes_input=None,
            steps_input=None,
            starts_shape=None,
            ends_shape=None,
            axes_shape=None,
            steps_shape=None,
            starts_dtype=None,
            ends_dtype=None,
            axes_dtype=None,
            steps_dtype=None,
            dtype=input_dtype,
            input_dtype=input_dtype,
        )
    if len(output_shape) != len(input_shape):
        raise ShapeInferenceError(
            f"{node.op_type} output rank must match input rank"
        )
    if inputs.starts_shape is None or inputs.ends_shape is None:
        raise UnsupportedOpError(
            f"{node.op_type} starts and ends inputs must be provided"
        )
    if inputs.starts_shape != inputs.ends_shape:
        raise ShapeInferenceError(
            f"{node.op_type} starts and ends must have matching shapes"
        )
    starts_len = inputs.starts_shape[0]
    if starts_len > len(input_shape):
        raise ShapeInferenceError(
            f"{node.op_type} starts length exceeds input rank"
        )
    if starts_len == 0 and output_shape != input_shape:
        raise ShapeInferenceError(
            f"{node.op_type} empty starts expects output shape to match input"
        )
    return SliceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        starts=None,
        steps=None,
        axes=None,
        starts_input=inputs.starts_input,
        ends_input=inputs.ends_input,
        axes_input=inputs.axes_input,
        steps_input=inputs.steps_input,
        starts_shape=inputs.starts_shape,
        ends_shape=inputs.ends_shape,
        axes_shape=inputs.axes_shape,
        steps_shape=inputs.steps_shape,
        starts_dtype=inputs.starts_dtype,
        ends_dtype=inputs.ends_dtype,
        axes_dtype=inputs.axes_dtype,
        steps_dtype=inputs.steps_dtype,
        dtype=input_dtype,
        input_dtype=input_dtype,
    )
