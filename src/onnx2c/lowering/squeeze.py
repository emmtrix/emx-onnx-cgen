from __future__ import annotations

from ..codegen.c_emitter import ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
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


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _validate_shape(shape: tuple[int, ...], node: Node, label: str) -> None:
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError(
                f"{node.op_type} does not support dynamic or zero dims in {label}"
            )


def _normalize_axes(
    axes: list[int], input_rank: int, node: Node
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += input_rank
        if axis < 0 or axis >= input_rank:
            raise ShapeInferenceError(
                f"{node.op_type} axis {axis} is out of range for rank {input_rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ShapeInferenceError(f"{node.op_type} axes must be unique")
    return tuple(sorted(normalized))


def _resolve_axes(graph: Graph, node: Node) -> tuple[int, ...] | None:
    axes_attr = node.attrs.get("axes")
    axes_values: list[int] | None = None
    if len(node.inputs) == 2:
        axes_initializer = _find_initializer(graph, node.inputs[1])
        if axes_initializer is not None:
            if axes_initializer.type.dtype not in {"int64", "int32"}:
                raise UnsupportedOpError(
                    "Squeeze axes input must be int64 or int32, "
                    f"got {axes_initializer.type.dtype}"
                )
            axes_values = [int(value) for value in axes_initializer.data.reshape(-1)]
    elif axes_attr is not None:
        axes_values = [int(value) for value in axes_attr]
    if axes_values is None:
        return None
    return tuple(axes_values)


def _expected_output_shape(
    input_shape: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[int, ...]:
    axis_set = set(axes)
    return tuple(
        dim for index, dim in enumerate(input_shape) if index not in axis_set
    )


def _validate_output_shape_for_unknown_axes(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], node: Node
) -> None:
    output_index = 0
    for dim in input_shape:
        if output_index < len(output_shape) and dim == output_shape[output_index]:
            output_index += 1
            continue
        if dim != 1:
            raise ShapeInferenceError(
                "Squeeze output shape must remove only dimensions of size 1"
            )
    if output_index != len(output_shape):
        raise ShapeInferenceError(
            "Squeeze output shape must preserve input order while removing size-1 axes"
        )


@register_lowering("Squeeze")
def lower_squeeze(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.outputs) != 1 or len(node.inputs) not in {1, 2}:
        raise UnsupportedOpError("Squeeze must have 1 or 2 inputs and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    _validate_shape(input_shape, node, "input")
    _validate_shape(output_shape, node, "output")
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Squeeze expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    axes = _resolve_axes(graph, node)
    if axes is None:
        if len(node.inputs) == 2:
            axes_dtype = _value_dtype(graph, node.inputs[1], node)
            if axes_dtype not in {"int64", "int32"}:
                raise UnsupportedOpError(
                    "Squeeze axes input must be int64 or int32, "
                    f"got {axes_dtype}"
                )
            _validate_output_shape_for_unknown_axes(input_shape, output_shape, node)
        else:
            expected_shape = tuple(dim for dim in input_shape if dim != 1)
            if expected_shape != output_shape:
                raise ShapeInferenceError(
                    "Squeeze output shape must be "
                    f"{expected_shape}, got {output_shape}"
                )
    else:
        normalized_axes = _normalize_axes(list(axes), len(input_shape), node)
        for axis in normalized_axes:
            if input_shape[axis] != 1:
                raise ShapeInferenceError(
                    "Squeeze axes must target dimensions of size 1"
                )
        expected_shape = _expected_output_shape(input_shape, normalized_axes)
        if expected_shape != output_shape:
            raise ShapeInferenceError(
                "Squeeze output shape must be "
                f"{expected_shape}, got {output_shape}"
            )
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=input_dtype,
        input_dtype=input_dtype,
    )
