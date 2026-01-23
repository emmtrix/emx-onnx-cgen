from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from .common import value_shape as resolved_value_shape
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


def _reshape_mismatch_error(
    node: Node,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> ShapeInferenceError:
    node_name = node.name or "<unnamed>"
    return ShapeInferenceError(
        "Reshape input/output element counts must match for op "
        f"{node.op_type} (node '{node_name}'): input shape {input_shape}, "
        f"output shape {output_shape}. "
        "Hint: ensure the reshape target has the same number of elements as "
        "the input."
    )


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
    graph: Graph, shape_node: Node, node: Node
) -> list[int]:
    if len(shape_node.inputs) != 1 or len(shape_node.outputs) != 1:
        raise UnsupportedOpError("Shape must have 1 input and 1 output")
    source_shape = _value_shape(graph, shape_node.inputs[0], node)
    start = int(shape_node.attrs.get("start", 0))
    end = int(shape_node.attrs.get("end", len(source_shape)))
    if start < 0:
        start += len(source_shape)
    if end < 0:
        end += len(source_shape)
    start = max(start, 0)
    end = min(end, len(source_shape))
    if start > end:
        return []
    return list(source_shape[start:end])


def _shape_values_from_initializer(
    graph: Graph,
    name: str,
) -> list[int] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "Reshape expects int64 or int32 shape input, "
            f"got {initializer.type.dtype.onnx_name}"
        )
    return [int(value) for value in initializer.data.reshape(-1)]


def _shape_values_from_input(
    graph: Graph,
    name: str,
    node: Node,
    *,
    _visited: set[str] | None = None,
) -> list[int] | None:
    if _visited is None:
        _visited = set()
    if name in _visited:
        return None
    _visited.add(name)
    try:
        shape_values = _shape_values_from_initializer(graph, name)
        if shape_values is not None:
            return shape_values
        source_node = _find_node_by_output(graph, name)
        if source_node is None:
            return None
        if source_node.op_type == "Shape":
            return _shape_values_from_shape_node(graph, source_node, node)
        if source_node.op_type == "Concat":
            axis = int(source_node.attrs.get("axis", 0))
            if axis != 0:
                raise UnsupportedOpError("Reshape shape concat must use axis 0")
            values: list[int] = []
            for input_name in source_node.inputs:
                input_values = _shape_values_from_input(
                    graph,
                    input_name,
                    node,
                    _visited=_visited,
                )
                if input_values is None:
                    return None
                values.extend(input_values)
            return values
        if source_node.op_type == "Cast":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Cast must have 1 input and 1 output")
            return _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type == "Unsqueeze":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Unsqueeze must have 1 input and 1 output")
            return _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type == "Identity":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Identity must have 1 input and 1 output")
            return _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type in {"Equal", "And", "Or", "Div", "Mod"}:
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{source_node.op_type} must have 2 inputs and 1 output"
                )
            left = _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            right = _shape_values_from_input(
                graph,
                source_node.inputs[1],
                node,
                _visited=_visited,
            )
            if left is None or right is None:
                return None
            if len(left) == 1 and len(right) != 1:
                left = left * len(right)
            if len(right) == 1 and len(left) != 1:
                right = right * len(left)
            if len(left) != len(right):
                return None
            if source_node.op_type == "Equal":
                return [1 if l == r else 0 for l, r in zip(left, right)]
            if source_node.op_type == "And":
                return [1 if (l and r) else 0 for l, r in zip(left, right)]
            if source_node.op_type == "Or":
                return [1 if (l or r) else 0 for l, r in zip(left, right)]
            if source_node.op_type == "Div":
                return [int(l / r) if r != 0 else 0 for l, r in zip(left, right)]
            if source_node.op_type == "Mod":
                return [l % r if r != 0 else 0 for l, r in zip(left, right)]
        if source_node.op_type == "Not":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Not must have 1 input and 1 output")
            values = _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if values is None:
                return None
            return [0 if value else 1 for value in values]
        if source_node.op_type == "Where":
            if len(source_node.inputs) != 3 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Where must have 3 inputs and 1 output")
            condition = _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if condition is None:
                return None
            on_true = _shape_values_from_input(
                graph,
                source_node.inputs[1],
                node,
                _visited=_visited,
            )
            on_false = _shape_values_from_input(
                graph,
                source_node.inputs[2],
                node,
                _visited=_visited,
            )
            if on_true is None or on_false is None:
                return None
            if len(condition) == 1:
                condition = condition * max(len(on_true), len(on_false))
            if len(on_true) == 1 and len(condition) != 1:
                on_true = on_true * len(condition)
            if len(on_false) == 1 and len(condition) != 1:
                on_false = on_false * len(condition)
            if not (len(condition) == len(on_true) == len(on_false)):
                return None
            return [
                t if cond else f
                for cond, t, f in zip(condition, on_true, on_false)
            ]
        return None
    finally:
        _visited.remove(name)


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
    contains_zero = False
    for index, dim in enumerate(shape_values):
        if dim == -1:
            if unknown_index is not None:
                raise ShapeInferenceError("Reshape allows only one -1 dimension")
            unknown_index = index
            output_dims.append(-1)
        else:
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
        raise _reshape_mismatch_error(node, input_shape, output_shape)
    return output_shape


@register_lowering("Reshape")
def lower_reshape(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Reshape must have 2 inputs and 1 output")
    input_shape = resolved_value_shape(graph, node.inputs[0], node)
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Reshape expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    output_value = graph.find_value(node.outputs[0])
    output_shape = resolved_value_shape(graph, node.outputs[0], node)
    output_dim_params = output_value.type.dim_params
    allowzero = int(node.attrs.get("allowzero", 0))
    resolved_shape: tuple[int, ...] | None = None
    shape_values = _shape_values_from_input(graph, node.inputs[1], node)
    if shape_values is not None:
        resolved_shape = _resolve_target_shape(
            input_shape,
            shape_values,
            allowzero=allowzero,
            node=node,
        )
        if output_shape and resolved_shape != output_shape and not any(
            output_dim_params
        ):
            raise ShapeInferenceError(
                "Reshape output shape must be "
                f"{resolved_shape}, got {output_shape}"
            )
    else:
        if _shape_product(output_shape) != _shape_product(input_shape):
            raise _reshape_mismatch_error(node, input_shape, output_shape)
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
