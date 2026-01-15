from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from ..ir.model import Graph, Node
from ..errors import UnsupportedOpError

LoweredOp = TypeVar("LoweredOp")
Handler = TypeVar("Handler")

_LOWERING_REGISTRY: dict[str, Callable[[Graph, Node], object]] = {}


def register_lowering(
    op_type: str,
) -> Callable[[Callable[[Graph, Node], LoweredOp]], Callable[[Graph, Node], LoweredOp]]:
    def decorator(
        func: Callable[[Graph, Node], LoweredOp],
    ) -> Callable[[Graph, Node], LoweredOp]:
        _LOWERING_REGISTRY[op_type] = func
        return func

    return decorator


def get_lowering(op_type: str) -> Callable[[Graph, Node], object] | None:
    return _LOWERING_REGISTRY.get(op_type)


def get_lowering_registry() -> Mapping[str, Callable[[Graph, Node], object]]:
    return _LOWERING_REGISTRY


def resolve_dispatch(
    op_type: str,
    registry: Mapping[str, Handler],
    *,
    binary_types: set[str],
    unary_types: set[str],
    binary_fallback: Callable[[], Handler],
    unary_fallback: Callable[[], Handler],
) -> Handler:
    handler = registry.get(op_type)
    if handler is not None:
        return handler
    if op_type in binary_types:
        return binary_fallback()
    if op_type in unary_types:
        return unary_fallback()
    raise UnsupportedOpError(f"Unsupported op {op_type}")
