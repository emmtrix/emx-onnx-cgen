from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import ReshapeOp
from ..dtypes import scalar_type_from_onnx
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


def _value_dtype(graph: Graph, name: str, node: Node) -> ScalarType:
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
        product *= dim
    return product


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _find_node_by_output(graph: Graph, name: str) -> Node | None:
    for node in graph.nodes:
        if name in node.outputs:
            return node
    return None


def _shape_values_from_shape_node(
    graph: Graph, name: str, node: Node
) -> list[int] | None:
    shape_node = _find_node_by_output(graph, name)
    if shape_node is None or shape_node.op_type != "Shape":
        return None
    if len(shape_node.inputs) != 1 or len(shape_node.outputs) != 1:
        raise UnsupportedOpError("Shape must have 1 input and 1 output")
    source_shape = _value_shape(graph, shape_node.inputs[0], node)
    return list(source_shape)


def _resolve_shape_values_from_graph(
    graph: Graph, name: str, node: Node
) -> list[int] | None:
    node_by_output = {
        output: candidate
        for candidate in graph.nodes
        for output in candidate.outputs
        if output
    }
    initializer_by_name = {
        initializer.name: initializer for initializer in graph.initializers
    }
    input_names = {value.name for value in graph.inputs}
    cache: dict[str, np.ndarray | None] = {}

    def resolve_axes(target: Node) -> list[int] | None:
        if len(target.inputs) > 1 and target.inputs[1]:
            axes_value = resolve_value(target.inputs[1])
            if axes_value is None:
                return None
            return [int(axis) for axis in axes_value.reshape(-1)]
        axes_attr = target.attrs.get("axes")
        if axes_attr is None:
            return None
        return [int(axis) for axis in axes_attr]

    def resolve_value(value_name: str) -> np.ndarray | None:
        if not value_name:
            return None
        if value_name in cache:
            return cache[value_name]
        if value_name in initializer_by_name:
            value = initializer_by_name[value_name].data
            cache[value_name] = value
            return value
        if value_name in input_names:
            cache[value_name] = None
            return None
        producer = node_by_output.get(value_name)
        if producer is None:
            cache[value_name] = None
            return None
        resolved = resolve_node(producer)
        cache[value_name] = resolved
        return resolved

    def resolve_node(target: Node) -> np.ndarray | None:
        if target.op_type == "Shape":
            return np.array(
                _value_shape(graph, target.inputs[0], node), dtype=np.int64
            )
        if target.op_type == "Identity":
            return resolve_value(target.inputs[0])
        if target.op_type == "Cast":
            input_value = resolve_value(target.inputs[0])
            if input_value is None:
                return None
            target_dtype = scalar_type_from_onnx(int(target.attrs.get("to", 0)))
            if target_dtype is None:
                return None
            return input_value.astype(target_dtype.np_dtype, copy=False)
        if target.op_type in {"Add", "Sub", "Mul", "Div"}:
            left = resolve_value(target.inputs[0])
            right = resolve_value(target.inputs[1])
            if left is None or right is None:
                return None
            if target.op_type == "Add":
                return left + right
            if target.op_type == "Sub":
                return left - right
            if target.op_type == "Mul":
                return left * right
            return left / right
        if target.op_type == "Concat":
            axis = int(target.attrs.get("axis", 0))
            parts = [resolve_value(name) for name in target.inputs if name]
            if any(part is None for part in parts):
                return None
            return np.concatenate(parts, axis=axis)
        if target.op_type == "Unsqueeze":
            axes = resolve_axes(target)
            value = resolve_value(target.inputs[0])
            if axes is None or value is None:
                return None
            for axis in sorted(axes):
                value = np.expand_dims(value, axis=axis)
            return value
        if target.op_type == "Squeeze":
            axes = resolve_axes(target)
            value = resolve_value(target.inputs[0])
            if value is None:
                return None
            if axes is None:
                return np.squeeze(value)
            return np.squeeze(value, axis=tuple(axes))
        if target.op_type == "Gather":
            data = resolve_value(target.inputs[0])
            indices = resolve_value(target.inputs[1])
            if data is None or indices is None:
                return None
            axis = int(target.attrs.get("axis", 0))
            return np.take(data, indices.astype(np.int64), axis=axis)
        if target.op_type == "Slice":
            data = resolve_value(target.inputs[0])
            starts = resolve_value(target.inputs[1])
            ends = resolve_value(target.inputs[2])
            if data is None or starts is None or ends is None:
                return None
            axes_value = (
                resolve_value(target.inputs[3]) if len(target.inputs) > 3 else None
            )
            steps_value = (
                resolve_value(target.inputs[4]) if len(target.inputs) > 4 else None
            )
            axes = (
                [int(axis) for axis in axes_value.reshape(-1)]
                if axes_value is not None
                else list(range(starts.size))
            )
            steps = (
                [int(step) for step in steps_value.reshape(-1)]
                if steps_value is not None
                else [1] * len(axes)
            )
            slices = [slice(None)] * data.ndim
            for axis, start, end, step in zip(
                axes,
                starts.reshape(-1),
                ends.reshape(-1),
                steps,
            ):
                slices[axis] = slice(int(start), int(end), int(step))
            return data[tuple(slices)]
        return None

    resolved = resolve_value(name)
    if resolved is None:
        return None
    return [int(value) for value in resolved.reshape(-1)]


def resolve_reshape_output_shape(
    input_shape: tuple[int, ...],
    shape_values: list[int],
    *,
    allowzero: int,
    node: Node,
) -> tuple[int, ...]:
    if allowzero not in (0, 1):
        raise UnsupportedOpError("Reshape allowzero must be 0 or 1")
    output_dims: list[int] = []
    unknown_index: int | None = None
    known_product = 1
    contains_zero = False
    for index, dim in enumerate(shape_values):
        if dim == -1:
            if unknown_index is not None:
                raise ShapeInferenceError("Reshape allows only one -1 dimension")
            unknown_index = index
            output_dims.append(-1)
            continue
        if dim == 0:
            contains_zero = True
            if allowzero == 0:
                if index >= len(input_shape):
                    raise ShapeInferenceError(
                        "Reshape zero dim must index into input shape"
                    )
                dim = input_shape[index]
        if dim < 0:
            raise ShapeInferenceError("Reshape dims must be >= -1")
        output_dims.append(dim)
        known_product *= dim
    if allowzero == 1 and contains_zero and unknown_index is not None:
        raise ShapeInferenceError(
            "Reshape allowzero cannot combine zero and -1 dimensions"
        )
    input_product = _shape_product(input_shape)
    if unknown_index is not None:
        if known_product == 0:
            if input_product != 0:
                raise ShapeInferenceError(
                    "Reshape cannot infer dimension from input shape"
                )
            output_dims[unknown_index] = 0
        else:
            if input_product % known_product != 0:
                raise ShapeInferenceError(
                    "Reshape cannot infer dimension from input shape"
                )
            output_dims[unknown_index] = input_product // known_product
    output_shape = tuple(output_dims)
    if _shape_product(output_shape) != input_product:
        raise ShapeInferenceError(
            "Reshape input and output element counts must match"
        )
    return output_shape


@register_lowering("Reshape")
def lower_reshape(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Reshape must have 2 inputs and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Reshape expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    allowzero = int(node.attrs.get("allowzero", 0))
    shape_initializer = _find_initializer(graph, node.inputs[1])
    resolved_shape: tuple[int, ...] | None = None
    if shape_initializer is None:
        shape_values = _shape_values_from_shape_node(
            graph, node.inputs[1], node
        )
        if shape_values is not None:
            resolved_shape = resolve_reshape_output_shape(
                input_shape,
                shape_values,
                allowzero=allowzero,
                node=node,
            )
        else:
            shape_values = _resolve_shape_values_from_graph(
                graph, node.inputs[1], node
            )
            if shape_values is not None:
                try:
                    resolved_shape = resolve_reshape_output_shape(
                        input_shape,
                        shape_values,
                        allowzero=allowzero,
                        node=node,
                    )
                except ShapeInferenceError:
                    resolved_shape = None
            elif _shape_product(output_shape) != _shape_product(input_shape):
                raise ShapeInferenceError(
                    "Reshape input and output element counts must match"
                )
    else:
        if shape_initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                "Reshape expects int64 or int32 shape input, "
                f"got {shape_initializer.type.dtype.onnx_name}"
            )
        if len(shape_initializer.type.shape) != 1:
            raise UnsupportedOpError("Reshape expects a 1D shape input")
        shape_values = [int(value) for value in shape_initializer.data.reshape(-1)]
        resolved_shape = resolve_reshape_output_shape(
            input_shape,
            shape_values,
            allowzero=allowzero,
            node=node,
        )
        if output_shape and resolved_shape != output_shape:
            raise ShapeInferenceError(
                "Reshape output shape must be "
                f"{resolved_shape}, got {output_shape}"
            )
    if resolved_shape is not None:
        output_shape = resolved_shape
    for dim in output_shape:
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=input_dtype,
        input_dtype=input_dtype,
    )
