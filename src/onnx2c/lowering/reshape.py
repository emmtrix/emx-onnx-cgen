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


def _resolve_target_shape(
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
    for index, dim in enumerate(shape_values):
        if dim == -1:
            if unknown_index is not None:
                raise ShapeInferenceError("Reshape allows only one -1 dimension")
            unknown_index = index
            output_dims.append(-1)
            continue
        if dim == 0:
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
    input_product = _shape_product(input_shape)
    if unknown_index is not None:
        if known_product == 0 or input_product % known_product != 0:
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
            f"got {input_dtype} and {output_dtype}"
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
            resolved_shape = _resolve_target_shape(
                input_shape,
                shape_values,
                allowzero=allowzero,
                node=node,
            )
        else:
            if _shape_product(output_shape) != _shape_product(input_shape):
                raise ShapeInferenceError(
                    "Reshape input and output element counts must match"
                )
    else:
        if shape_initializer.type.dtype not in {"int64", "int32"}:
            raise UnsupportedOpError(
                "Reshape expects int64 or int32 shape input, "
                f"got {shape_initializer.type.dtype}"
            )
        if len(shape_initializer.type.shape) != 1:
            raise UnsupportedOpError("Reshape expects a 1D shape input")
        shape_values = [int(value) for value in shape_initializer.data.reshape(-1)]
        resolved_shape = _resolve_target_shape(
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
    )
