from __future__ import annotations

from collections.abc import Sequence

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node


def ensure_supported_dtype(dtype: ScalarType) -> ScalarType:
    if not isinstance(dtype, ScalarType):
        raise UnsupportedOpError(f"Unsupported dtype {dtype}")
    return dtype


def value_dtype(graph: Graph, name: str, node: Node | None = None) -> ScalarType:
    try:
        value = graph.find_value(name)
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc
    return ensure_supported_dtype(value.type.dtype)


def value_shape(graph: Graph, name: str, node: Node | None = None) -> tuple[int, ...]:
    try:
        value = graph.find_value(name)
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc
    if any(value.type.dim_params):
        resolved = _resolve_value_shape(graph, name, node)
        if resolved is not None:
            return resolved
        return value.type.shape
    return value.type.shape


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
    graph: Graph, shape_node: Node, node: Node | None
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
        elif left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1:
            result.append(left_dim)
        else:
            return None
    return tuple(reversed(result))


def _resolve_value_shape(
    graph: Graph,
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
        shape = value.type.shape
        if not any(value.type.dim_params):
            return shape
        source_node = _find_node_by_output(graph, name)
        if source_node is None:
            return None
        if source_node.op_type == "Expand":
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Expand must have 2 inputs and 1 output")
            shape_values = _shape_values_from_input(
                graph, source_node.inputs[1], node
            )
            if shape_values is not None and all(dim >= 0 for dim in shape_values):
                return tuple(shape_values)
            return None
        if source_node.op_type == "Reshape":
            if len(source_node.inputs) != 2 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Reshape must have 2 inputs and 1 output")
            shape_values = _shape_values_from_input(
                graph, source_node.inputs[1], node
            )
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
                    continue
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
            if left is None or right is None:
                return None
            return _broadcast_shapes(left, right)
        if source_node.op_type == "Where":
            if len(source_node.inputs) != 3 or len(source_node.outputs) != 1:
                raise UnsupportedOpError("Where must have 3 inputs and 1 output")
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
            if on_true is None or on_false is None:
                return None
            return _broadcast_shapes(on_true, on_false)
        return None
    finally:
        _visited.remove(name)


def node_dtype(graph: Graph, node: Node, *names: str) -> ScalarType:
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
