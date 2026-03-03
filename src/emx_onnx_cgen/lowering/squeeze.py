from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Initializer, Node
from .common import reconcile_shape_with_dim_params
from .common import value_dim_params
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _validate_shape(shape: tuple[int, ...], node: Node, label: str) -> None:
    for dim in shape:
        if dim < 0:
            raise ShapeInferenceError(
                f"{node.op_type} does not support dynamic dims in {label}"
            )


def _normalize_axes(axes: list[int], input_rank: int, node: Node) -> tuple[int, ...]:
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
            if axes_initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
                raise UnsupportedOpError(
                    "Squeeze axes input must be int64 or int32, "
                    f"got {axes_initializer.type.dtype.onnx_name}"
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
    return tuple(dim for index, dim in enumerate(input_shape) if index not in axis_set)


def _expected_output_shape_without_axes(
    input_shape: tuple[int, ...], input_dim_params: tuple[str | None, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    for axis, dim in enumerate(input_shape):
        dim_param = input_dim_params[axis] if axis < len(input_dim_params) else None
        # Unknown symbolic dims are represented with placeholder size 1 in import.
        # Without explicit axes we may only drop dimensions known to be concrete 1.
        if dim == 1 and not dim_param:
            continue
        result.append(dim)
    return tuple(result)


def _validate_output_shape_for_unknown_axes(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], node: Node
) -> None:
    output_index = 0
    for dim in input_shape:
        if output_index < len(output_shape) and dim == output_shape[output_index]:
            output_index += 1
        else:
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
    input_dim_params = value_dim_params(graph, node.inputs[0])
    has_symbolic_input_dims = any(input_dim_params)
    _validate_shape(input_shape, node, "input")
    _validate_shape(output_shape, node, "output")
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Squeeze expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    axes = _resolve_axes(graph, node)
    if axes is None:
        if len(node.inputs) == 2:
            axes_dtype = _value_dtype(graph, node.inputs[1], node)
            if axes_dtype not in {ScalarType.I64, ScalarType.I32}:
                raise UnsupportedOpError(
                    "Squeeze axes input must be int64 or int32, "
                    f"got {axes_dtype.onnx_name}"
                )
            _validate_output_shape_for_unknown_axes(input_shape, output_shape, node)
        else:
            expected_shape = _expected_output_shape_without_axes(
                input_shape, input_dim_params
            )
            reconciled = reconcile_shape_with_dim_params(
                expected_shape,
                output_shape,
                tuple(
                    dim_param
                    for axis, dim_param in enumerate(input_dim_params)
                    if not (input_shape[axis] == 1 and not dim_param)
                ),
            )
            if (
                output_shape != expected_shape
                and not (
                    output_shape == () and expected_shape and has_symbolic_input_dims
                )
                and reconciled is None
            ):
                raise ShapeInferenceError(
                    f"Squeeze output shape must be {expected_shape}, got {output_shape}"
                )
            output_shape = expected_shape
    else:
        normalized_axes = _normalize_axes(list(axes), len(input_shape), node)
        for axis in normalized_axes:
            dim_param = input_dim_params[axis] if axis < len(input_dim_params) else None
            if input_shape[axis] != 1 and not dim_param:
                raise ShapeInferenceError(
                    "Squeeze axes must target dimensions of size 1"
                )
        expected_shape = _expected_output_shape(input_shape, normalized_axes)
        reduced_dim_params = tuple(
            dim_param
            for axis, dim_param in enumerate(input_dim_params)
            if axis not in set(normalized_axes)
        )
        reconciled = reconcile_shape_with_dim_params(
            expected_shape,
            output_shape,
            reduced_dim_params,
        )
        if (
            output_shape != expected_shape
            and not (output_shape == () and expected_shape and has_symbolic_input_dims)
            and reconciled is None
        ):
            raise ShapeInferenceError(
                f"Squeeze output shape must be {expected_shape}, got {output_shape}"
            )
        output_shape = expected_shape
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )
