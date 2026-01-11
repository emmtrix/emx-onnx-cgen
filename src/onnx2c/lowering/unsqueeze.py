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
    axes: list[int], output_rank: int, node: Node
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += output_rank
        if axis < 0 or axis >= output_rank:
            raise ShapeInferenceError(
                f"{node.op_type} axis {axis} is out of range for rank {output_rank}"
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
                    "Unsqueeze axes input must be int64 or int32, "
                    f"got {axes_initializer.type.dtype}"
                )
            axes_values = [int(value) for value in axes_initializer.data.reshape(-1)]
    elif axes_attr is not None:
        axes_values = [int(value) for value in axes_attr]
    if axes_values is None and axes_attr is None and len(node.inputs) != 2:
        raise UnsupportedOpError("Unsqueeze requires axes")
    if axes_values is None:
        return None
    if not axes_values:
        raise UnsupportedOpError("Unsqueeze requires non-empty axes")
    return tuple(axes_values)


def _expected_output_shape(
    input_shape: tuple[int, ...], axes: tuple[int, ...], node: Node
) -> tuple[int, ...]:
    output_rank = len(input_shape) + len(axes)
    normalized_axes = _normalize_axes(list(axes), output_rank, node)
    output_dims: list[int] = []
    input_index = 0
    for axis in range(output_rank):
        if axis in normalized_axes:
            output_dims.append(1)
        else:
            output_dims.append(input_shape[input_index])
            input_index += 1
    return tuple(output_dims)


@register_lowering("Unsqueeze")
def lower_unsqueeze(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.outputs) != 1 or len(node.inputs) not in {1, 2}:
        raise UnsupportedOpError("Unsqueeze must have 1 or 2 inputs and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    _validate_shape(input_shape, node, "input")
    _validate_shape(output_shape, node, "output")
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Unsqueeze expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    axes = _resolve_axes(graph, node)
    if axes is None:
        if len(node.inputs) == 2:
            axes_dtype = _value_dtype(graph, node.inputs[1], node)
            if axes_dtype not in {"int64", "int32"}:
                raise UnsupportedOpError(
                    "Unsqueeze axes input must be int64 or int32, "
                    f"got {axes_dtype}"
                )
        if len(output_shape) <= len(input_shape):
            raise ShapeInferenceError(
                "Unsqueeze output rank must exceed input rank"
            )
        input_index = 0
        for dim in output_shape:
            if input_index < len(input_shape) and dim == input_shape[input_index]:
                input_index += 1
                continue
            if dim != 1:
                raise ShapeInferenceError(
                    "Unsqueeze output shape must insert ones only"
                )
        if input_index != len(input_shape):
            raise ShapeInferenceError(
                "Unsqueeze output shape must contain input shape in order"
            )
    else:
        expected_shape = _expected_output_shape(input_shape, axes, node)
        if expected_shape != output_shape:
            raise ShapeInferenceError(
                "Unsqueeze output shape must be "
                f"{expected_shape}, got {output_shape}"
            )
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=input_dtype,
    )
