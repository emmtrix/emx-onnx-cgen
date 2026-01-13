from __future__ import annotations

from collections.abc import Sequence

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node


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
        return graph.find_value(name).type.shape
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


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
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        product *= dim
    return product


def optional_name(names: Sequence[str], index: int) -> str | None:
    if index >= len(names):
        return None
    name = names[index]
    return name or None
