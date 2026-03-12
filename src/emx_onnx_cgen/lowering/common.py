from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Initializer, Node, TensorType

LiteralValue = float | int | bool


def ensure_supported_dtype(dtype: ScalarType) -> ScalarType:
    if not isinstance(dtype, ScalarType):
        raise UnsupportedOpError(f"Unsupported dtype {dtype}")
    return dtype


def onnx_opset_version(graph: Graph | GraphContext, domain: str = "") -> int | None:
    if isinstance(graph, GraphContext):
        return graph.opset_version(domain)
    if domain in {"", "ai.onnx"}:
        domains = {"", "ai.onnx"}
    else:
        domains = {domain}
    for opset_domain, version in graph.opset_imports:
        if opset_domain in domains:
            return int(version)
    return None


def value_dtype(
    graph: Graph | GraphContext, name: str, node: Node | None = None
) -> ScalarType:
    if isinstance(graph, GraphContext):
        return graph.dtype(name, node)
    try:
        value = graph.find_value(name)
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc
    if not isinstance(value.type, TensorType):
        op_type = node.op_type if node is not None else "unknown"
        raise UnsupportedOpError(
            f"Unsupported non-tensor value '{name}' in op {op_type}."
        )
    return ensure_supported_dtype(value.type.dtype)


def value_shape(
    graph: Graph | GraphContext, name: str, node: Node | None = None
) -> tuple[int, ...]:
    if isinstance(graph, GraphContext):
        if graph.has_shape(name):
            return graph.shape(name, node)
        shape = graph.shape(name, node)
        value = graph.find_value(name)
    else:
        try:
            value = graph.find_value(name)
        except KeyError as exc:
            op_type = node.op_type if node is not None else "unknown"
            raise ShapeInferenceError(
                f"Missing shape for value '{name}' in op {op_type}. "
                "Hint: run ONNX shape inference or export with static shapes."
            ) from exc
        if not isinstance(value.type, TensorType):
            op_type = node.op_type if node is not None else "unknown"
            raise UnsupportedOpError(
                f"Unsupported non-tensor value '{name}' in op {op_type}."
            )
        shape = value.type.shape
    if isinstance(value.type, TensorType) and any(value.type.dim_params):
        resolved = _resolve_value_shape(graph, name, node)
        if resolved is not None:
            if isinstance(graph, GraphContext):
                graph.set_shape(name, resolved)
            return resolved
        return value.type.shape
    return shape


def _find_initializer(graph: Graph | GraphContext, name: str) -> Initializer | None:
    if isinstance(graph, GraphContext):
        return graph.initializer(name)
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _find_node_by_output(graph: Graph | GraphContext, name: str) -> Node | None:
    if isinstance(graph, GraphContext):
        return graph.producer(name)
    for node in graph.nodes:
        if name in node.outputs:
            return node
    return None


def _find_consumers(graph: Graph | GraphContext, name: str) -> tuple[Node, ...]:
    return tuple(node for node in graph.nodes if name in node.inputs)


def _shape_values_from_shape_node(
    graph: Graph | GraphContext, shape_node: Node, node: Node | None
) -> list[int]:
    if len(shape_node.inputs) != 1 or len(shape_node.outputs) != 1:
        raise UnsupportedOpError("Shape must have 1 input and 1 output")
    source_shape = value_shape(graph, shape_node.inputs[0], node)
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
    graph: Graph | GraphContext,
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


def _numeric_values_from_initializer(
    graph: Graph | GraphContext,
    name: str,
) -> list[LiteralValue] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype == ScalarType.STRING:
        return None
    return [value.item() for value in initializer.data.reshape(-1)]


def _broadcast_values(
    left: list[LiteralValue],
    right: list[LiteralValue],
) -> tuple[list[LiteralValue], list[LiteralValue]] | None:
    if len(left) == 1 and len(right) != 1:
        left = left * len(right)
    if len(right) == 1 and len(left) != 1:
        right = right * len(left)
    if len(left) != len(right):
        return None
    return left, right


def _gather_literal_values(
    graph: Graph | GraphContext,
    source_node: Node,
    node: Node | None,
    *,
    value_resolver: Callable[
        [Graph | GraphContext, str, Node | None],
        list[LiteralValue] | None,
    ],
    _visited: set[str],
) -> list[LiteralValue] | None:
    if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
        raise UnsupportedOpError("Gather must have 2 inputs and 1 output")
    axis = int(source_node.attrs.get("axis", 0))
    data_shape = value_shape(graph, source_node.inputs[0], node)
    if len(data_shape) != 1:
        return None
    if axis < 0:
        axis += len(data_shape)
    if axis != 0:
        return None
    data = value_resolver(
        graph,
        source_node.inputs[0],
        node,
        _visited=_visited,
    )
    indices = value_resolver(
        graph,
        source_node.inputs[1],
        node,
        _visited=_visited,
    )
    if data is None or indices is None:
        return None
    axis_size = data_shape[0]
    if axis_size < 0 or len(data) != axis_size:
        return None
    gathered: list[LiteralValue] = []
    for index_value in indices:
        if isinstance(index_value, bool) or not isinstance(index_value, int):
            return None
        index = index_value
        if index < 0:
            index += axis_size
        if index < 0 or index >= axis_size:
            return None
        gathered.append(data[index])
    return gathered


def _shape_values_from_input(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None,
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
        if source_node.op_type == "Size":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Size must have 1 input and 1 output")
            input_shape = value_shape(graph, source_node.inputs[0], node)
            if any(dim < 0 for dim in input_shape):
                return None
            return [shape_product(input_shape)]
        if source_node.op_type == "Concat":
            axis = int(source_node.attrs.get("axis", 0))
            if axis not in {0, -1}:
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
            if len(source_node.outputs) != 1:
                raise UnsupportedOpError("Unsqueeze must have 1 output")
            return _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type == "Squeeze":
            if len(source_node.outputs) != 1:
                raise UnsupportedOpError("Squeeze must have 1 output")
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
        if source_node.op_type == "Neg":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Neg must have 1 input and 1 output")
            values = _shape_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if values is None:
                return None
            return [-value for value in values]
        if source_node.op_type == "Gather":
            gathered = _gather_literal_values(
                graph,
                source_node,
                node,
                value_resolver=_shape_values_from_input,
                _visited=_visited,
            )
            if gathered is None:
                return None
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in gathered
            ):
                return None
            return [int(value) for value in gathered]
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
                return [
                    1 if left_value == right_value else 0
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "And":
                return [
                    1 if (left_value and right_value) else 0
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Or":
                return [
                    1 if (left_value or right_value) else 0
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Div":
                return [
                    int(left_value / right_value) if right_value != 0 else 0
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Mod":
                return [
                    left_value % right_value if right_value != 0 else 0
                    for left_value, right_value in zip(left, right)
                ]
        if source_node.op_type in {"Add", "Sub", "Mul", "Max", "Min"}:
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
            if source_node.op_type == "Add":
                return [
                    left_value + right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Sub":
                return [
                    left_value - right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Mul":
                return [
                    left_value * right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Max":
                return [
                    max(left_value, right_value)
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Min":
                return [
                    min(left_value, right_value)
                    for left_value, right_value in zip(left, right)
                ]
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
                t if cond else f for cond, t, f in zip(condition, on_true, on_false)
            ]
        return None
    finally:
        _visited.remove(name)


def _numeric_values_from_input(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None,
    *,
    _visited: set[str] | None = None,
) -> list[LiteralValue] | None:
    if _visited is None:
        _visited = set()
    if name in _visited:
        return None
    _visited.add(name)
    try:
        numeric_values = _numeric_values_from_initializer(graph, name)
        if numeric_values is not None:
            return numeric_values
        source_node = _find_node_by_output(graph, name)
        if source_node is None:
            return None
        if source_node.op_type == "Shape":
            return _shape_values_from_shape_node(graph, source_node, node)
        if source_node.op_type == "Size":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Size must have 1 input and 1 output")
            input_shape = value_shape(graph, source_node.inputs[0], node)
            if any(dim < 0 for dim in input_shape):
                return None
            return [shape_product(input_shape)]
        if source_node.op_type == "Concat":
            axis = int(source_node.attrs.get("axis", 0))
            rank = len(value_shape(graph, source_node.outputs[0], node))
            if axis < 0:
                axis += rank
            if axis != 0:
                return None
            values: list[LiteralValue] = []
            for input_name in source_node.inputs:
                input_values = _numeric_values_from_input(
                    graph,
                    input_name,
                    node,
                    _visited=_visited,
                )
                if input_values is None:
                    return None
                values.extend(input_values)
            return values
        if source_node.op_type in {"Cast", "Identity"}:
            if len(source_node.outputs) != 1 or len(source_node.inputs) != 1:
                raise UnsupportedOpError(
                    f"{source_node.op_type} must have 1 input and 1 output"
                )
            return _numeric_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type in {"Unsqueeze", "Squeeze"}:
            if len(source_node.outputs) != 1 or not source_node.inputs:
                raise UnsupportedOpError(
                    f"{source_node.op_type} must have at least 1 input and 1 output"
                )
            return _numeric_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
        if source_node.op_type == "Neg":
            if len(source_node.inputs) != 1 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Neg must have 1 input and 1 output")
            values = _numeric_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if values is None:
                return None
            return [-value for value in values]
        if source_node.op_type == "Gather":
            return _gather_literal_values(
                graph,
                source_node,
                node,
                value_resolver=_numeric_values_from_input,
                _visited=_visited,
            )
        if source_node.op_type in {"Add", "Sub", "Mul", "Div", "Mod", "Max", "Min"}:
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{source_node.op_type} must have 2 inputs and 1 output"
                )
            left = _numeric_values_from_input(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            right = _numeric_values_from_input(
                graph,
                source_node.inputs[1],
                node,
                _visited=_visited,
            )
            if left is None or right is None:
                return None
            broadcast = _broadcast_values(left, right)
            if broadcast is None:
                return None
            left, right = broadcast
            if source_node.op_type == "Add":
                return [
                    left_value + right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Sub":
                return [
                    left_value - right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Mul":
                return [
                    left_value * right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Div":
                if any(right_value == 0 for right_value in right):
                    return None
                return [
                    left_value / right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Mod":
                if any(right_value == 0 for right_value in right):
                    return None
                return [
                    left_value % right_value
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Max":
                return [
                    max(left_value, right_value)
                    for left_value, right_value in zip(left, right)
                ]
            if source_node.op_type == "Min":
                return [
                    min(left_value, right_value)
                    for left_value, right_value in zip(left, right)
                ]
        return None
    finally:
        _visited.remove(name)


def _broadcast_shapes(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...] | None:
    result = []
    left_rev = list(reversed(left))
    right_rev = list(reversed(right))
    for index in range(max(len(left_rev), len(right_rev))):
        left_dim = left_rev[index] if index < len(left_rev) else 1
        right_dim = right_rev[index] if index < len(right_rev) else 1
        if left_dim == right_dim:
            result.append(left_dim)
        elif left_dim < 0 and right_dim == 1:
            result.append(left_dim)
        elif right_dim < 0 and left_dim == 1:
            result.append(right_dim)
        elif left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1:
            result.append(left_dim)
        else:
            return None
    return tuple(reversed(result))


_SHAPE_PRESERVING_CONSUMER_UNARY_OPS = frozenset(
    {
        "Identity",
        "Cast",
        "CastLike",
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Ceil",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Log",
        "Neg",
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
    }
)

_SHAPE_PRESERVING_CONSUMER_BINARY_OPS = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Max",
        "Min",
    }
)


def _shape_is_scalar_like(shape: tuple[int, ...]) -> bool:
    if shape == () or shape == (1,):
        return True
    if not shape:
        return True
    return all(dim == 1 for dim in shape)


def _static_shape_if_known(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None,
    *,
    _visited: set[str],
) -> tuple[int, ...] | None:
    value = graph.find_value(name)
    if not isinstance(value.type, TensorType):
        return None
    resolved = _resolve_value_shape(graph, name, node, _visited=_visited)
    if resolved is not None:
        return resolved
    if any(value.type.dim_params):
        return None
    return value.type.shape


def _resolve_value_shape_from_consumers(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None,
    *,
    _visited: set[str],
) -> tuple[int, ...] | None:
    for consumer in _find_consumers(graph, name):
        if len(consumer.outputs) != 1 or not consumer.outputs[0]:
            continue
        output_name = consumer.outputs[0]
        output_shape = _static_shape_if_known(
            graph,
            output_name,
            node,
            _visited=_visited,
        )
        if output_shape is None or any(dim < 0 for dim in output_shape):
            continue
        if consumer.op_type in _SHAPE_PRESERVING_CONSUMER_UNARY_OPS:
            if len(consumer.inputs) != 1:
                continue
            if consumer.op_type == "CastLike" and consumer.inputs[0] != name:
                continue
            return output_shape
        if consumer.op_type in _SHAPE_PRESERVING_CONSUMER_BINARY_OPS:
            if len(consumer.inputs) != 2:
                continue
            input_index = consumer.inputs.index(name)
            other_input = consumer.inputs[1 - input_index]
            other_shape = _static_shape_if_known(
                graph,
                other_input,
                node,
                _visited=_visited,
            )
            if other_shape is None:
                return output_shape
            if _shape_is_scalar_like(other_shape) or other_shape == output_shape:
                return output_shape
    return None


def _resolve_value_shape(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None,
    *,
    _visited: set[str] | None = None,
) -> tuple[int, ...] | None:
    if _visited is None:
        _visited = set()
    if name in _visited:
        return None
    _visited.add(name)
    try:
        value = graph.find_value(name)
        if not isinstance(value.type, TensorType):
            op_type = node.op_type if node is not None else "unknown"
            raise UnsupportedOpError(
                f"Unsupported non-tensor value '{name}' in op {op_type}."
            )
        shape = value.type.shape
        if not any(value.type.dim_params):
            return shape
        source_node = _find_node_by_output(graph, name)
        if source_node is None:
            return None
        if source_node.op_type in {"Identity", "Cast", "CastLike"}:
            if not source_node.inputs or len(source_node.outputs) != 1:
                consumer_shape = _resolve_value_shape_from_consumers(
                    graph,
                    name,
                    node,
                    _visited=_visited,
                )
                if consumer_shape is not None:
                    return consumer_shape
                return None
            passthrough_shape = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if passthrough_shape is not None:
                return passthrough_shape
        if source_node.op_type == "Unsqueeze":
            if len(source_node.outputs) != 1:
                return None
            input_shape = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if input_shape is None:
                return None
            axes_values: list[int] | None = None
            if len(source_node.inputs) > 1 and source_node.inputs[1]:
                axes_values = _shape_values_from_input(
                    graph, source_node.inputs[1], node
                )
            elif "axes" in source_node.attrs:
                axes_values = [int(a) for a in source_node.attrs["axes"]]
            if axes_values is None:
                return None
            out_rank = len(input_shape) + len(axes_values)
            normalized_axes = sorted(a if a >= 0 else a + out_rank for a in axes_values)
            result = list(input_shape)
            for ax in normalized_axes:
                result.insert(ax, 1)
            return tuple(result)
        if source_node.op_type == "Squeeze":
            if len(source_node.outputs) != 1:
                return None
            input_shape = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if input_shape is None:
                return None
            squeeze_axes: list[int] | None = None
            if len(source_node.inputs) > 1 and source_node.inputs[1]:
                squeeze_axes = _shape_values_from_input(
                    graph, source_node.inputs[1], node
                )
            elif "axes" in source_node.attrs:
                squeeze_axes = [int(a) for a in source_node.attrs["axes"]]
            if squeeze_axes is not None:
                normalized_sq = sorted(
                    (a if a >= 0 else a + len(input_shape) for a in squeeze_axes),
                    reverse=True,
                )
                result = list(input_shape)
                for ax in normalized_sq:
                    if 0 <= ax < len(result) and result[ax] == 1:
                        result.pop(ax)
                return tuple(result)
            return tuple(d for d in input_shape if d != 1)
        if source_node.op_type == "Range":
            if len(source_node.inputs) != 3 or len(source_node.outputs) != 1:
                consumer_shape = _resolve_value_shape_from_consumers(
                    graph,
                    name,
                    node,
                    _visited=_visited,
                )
                if consumer_shape is not None:
                    return consumer_shape
                return None
            start_vals = _numeric_values_from_input(graph, source_node.inputs[0], node)
            limit_vals = _numeric_values_from_input(graph, source_node.inputs[1], node)
            step_vals = _numeric_values_from_input(graph, source_node.inputs[2], node)
            if (
                start_vals is not None
                and limit_vals is not None
                and step_vals is not None
                and len(start_vals) == 1
                and len(limit_vals) == 1
                and len(step_vals) == 1
                and step_vals[0] != 0
            ):
                length = max(
                    0, math.ceil((limit_vals[0] - start_vals[0]) / step_vals[0])
                )
                return (length,)
        if source_node.op_type == "Expand":
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Expand must have 2 inputs and 1 output")
            shape_values = _shape_values_from_input(graph, source_node.inputs[1], node)
            if shape_values is not None and all(dim >= 0 for dim in shape_values):
                return tuple(shape_values)
            return None
        if source_node.op_type == "Reshape":
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Reshape must have 2 inputs and 1 output")
            shape_values = _shape_values_from_input(graph, source_node.inputs[1], node)
            if shape_values is None:
                return None
            allowzero = int(source_node.attrs.get("allowzero", 0))
            input_shape = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if input_shape is None:
                return None
            output_dims: list[int] = []
            unknown_index: int | None = None
            known_product = 1
            contains_zero = False
            for index, dim in enumerate(shape_values):
                if dim == -1:
                    if unknown_index is not None:
                        return None
                    unknown_index = len(output_dims)
                    output_dims.append(-1)
                else:
                    if dim == 0:
                        contains_zero = True
                        if allowzero == 0:
                            if index >= len(input_shape):
                                return None
                            dim = input_shape[index]
                    if dim < 0:
                        return None
                    output_dims.append(dim)
                    known_product *= dim
            if allowzero == 1 and contains_zero and unknown_index is not None:
                return None
            input_product = shape_product(input_shape)
            if unknown_index is not None:
                if known_product == 0:
                    if input_product != 0:
                        return None
                    output_dims[unknown_index] = 0
                else:
                    if input_product % known_product != 0:
                        return None
                    output_dims[unknown_index] = input_product // known_product
            return tuple(output_dims)
        if source_node.op_type in {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Mod",
            "And",
            "Or",
            "Xor",
            "Equal",
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
        }:
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{source_node.op_type} must have 2 inputs and 1 output"
                )
            left = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            right = _resolve_value_shape(
                graph,
                source_node.inputs[1],
                node,
                _visited=_visited,
            )
            if left is not None and right is not None:
                broadcast_shape = _broadcast_shapes(left, right)
                if broadcast_shape is not None:
                    return broadcast_shape
        if source_node.op_type == "Where":
            if len(source_node.inputs) != 3 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Where must have 3 inputs and 1 output")
            condition = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            on_true = _resolve_value_shape(
                graph,
                source_node.inputs[1],
                node,
                _visited=_visited,
            )
            on_false = _resolve_value_shape(
                graph,
                source_node.inputs[2],
                node,
                _visited=_visited,
            )
            if condition is not None and on_true is not None and on_false is not None:
                cond_xy = _broadcast_shapes(condition, on_true)
                if cond_xy is not None:
                    where_shape = _broadcast_shapes(cond_xy, on_false)
                    if where_shape is not None:
                        return where_shape
        if source_node.op_type == "Pad":
            if not source_node.inputs or len(source_node.outputs) != 1:
                consumer_shape = _resolve_value_shape_from_consumers(
                    graph,
                    name,
                    node,
                    _visited=_visited,
                )
                if consumer_shape is not None:
                    return consumer_shape
                return None
            input_shape = _resolve_value_shape(
                graph,
                source_node.inputs[0],
                node,
                _visited=_visited,
            )
            if input_shape is not None:
                rank = len(input_shape)
                pads_name = (
                    source_node.inputs[1]
                    if len(source_node.inputs) > 1 and source_node.inputs[1]
                    else None
                )
                if pads_name:
                    pads = _shape_values_from_input(graph, pads_name, node)
                else:
                    pads_attr = source_node.attrs.get("pads")
                    if pads_attr is not None:
                        pads = [int(v) for v in pads_attr]
                    else:
                        pads = [0] * (2 * rank)
                if pads is not None:
                    axes_name = (
                        source_node.inputs[3]
                        if len(source_node.inputs) > 3 and source_node.inputs[3]
                        else None
                    )
                    if axes_name:
                        axes_values = _shape_values_from_input(graph, axes_name, node)
                        if axes_values is not None:
                            axes = [a if a >= 0 else a + rank for a in axes_values]
                            if len(pads) == 2 * len(axes):
                                output = list(input_shape)
                                for i, axis in enumerate(axes):
                                    if axis < 0 or axis >= rank:
                                        break
                                    output[axis] += pads[i] + pads[i + len(axes)]
                                else:
                                    return tuple(output)
                    else:
                        if len(pads) == 2 * rank:
                            output = list(input_shape)
                            for i in range(rank):
                                output[i] += pads[i] + pads[i + rank]
                            return tuple(output)
        consumer_shape = _resolve_value_shape_from_consumers(
            graph,
            name,
            node,
            _visited=_visited,
        )
        if consumer_shape is not None:
            return consumer_shape
        return None
    finally:
        _visited.remove(name)


def node_dtype(graph: Graph | GraphContext, node: Node, *names: str) -> ScalarType:
    filtered = [name for name in names if name]
    if not filtered:
        raise UnsupportedOpError(
            f"{node.op_type} expects at least one typed input or output"
        )
    dtypes = {value_dtype(graph, name, node) for name in filtered}
    if len(dtypes) != 1:
        dtype_names = ", ".join(dtype.onnx_name for dtype in sorted(dtypes, key=str))
        raise UnsupportedOpError(
            f"{node.op_type} expects matching dtypes, got {dtype_names}"
        )
    return next(iter(dtypes))


def shape_product(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    product = 1
    for dim in shape:
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
        if dim == 0:
            return 0
        product *= dim
    return product


def optional_name(names: Sequence[str], index: int) -> str | None:
    if index >= len(names):
        return None
    name = names[index]
    return name or None


def resolve_int_list_from_value(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None = None,
) -> list[int] | None:
    return _shape_values_from_input(graph, name, node)


def resolve_numeric_list_from_value(
    graph: Graph | GraphContext,
    name: str,
    node: Node | None = None,
) -> list[LiteralValue] | None:
    return _numeric_values_from_input(graph, name, node)


def value_has_dim_params(
    graph: Graph | GraphContext,
    name: str,
) -> bool:
    return any(graph.find_value(name).type.dim_params)


def value_dim_params(
    graph: Graph | GraphContext,
    name: str,
) -> tuple[str | None, ...]:
    value = graph.find_value(name)
    if not isinstance(value.type, TensorType):
        raise UnsupportedOpError(f"Unsupported non-tensor value '{name}'.")
    return value.type.dim_params


def reconcile_shape_with_dim_params(
    expected_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
    dim_params: tuple[str | None, ...],
) -> tuple[int, ...] | None:
    if not expected_shape or not actual_shape:
        return None
    if len(expected_shape) != len(actual_shape):
        return None
    for axis, (expected_dim, actual_dim) in enumerate(
        zip(expected_shape, actual_shape)
    ):
        dim_param = dim_params[axis] if axis < len(dim_params) else None
        if expected_dim != actual_dim and not dim_param:
            return None
    return actual_shape
