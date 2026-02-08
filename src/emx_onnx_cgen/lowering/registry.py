from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.op_base import OpBase
from ..errors import UnsupportedOpError

LoweredOp = TypeVar("LoweredOp")
Handler = TypeVar("Handler")

_LOWERING_REGISTRY: dict[str, Callable[[Graph | GraphContext, Node], OpBase]] = {}


def register_lowering(
    op_type: str,
) -> Callable[[Callable[[Graph, Node], LoweredOp]], Callable[[Graph, Node], LoweredOp]]:
    def decorator(
        func: Callable[[Graph | GraphContext, Node], LoweredOp],
    ) -> Callable[[Graph | GraphContext, Node], LoweredOp]:
        _LOWERING_REGISTRY[op_type] = func
        return func

    return decorator


def register_lowering_if_missing(
    op_type: str,
) -> Callable[
    [Callable[[Graph | GraphContext, Node], LoweredOp]],
    Callable[[Graph | GraphContext, Node], LoweredOp],
]:
    def decorator(
        func: Callable[[Graph | GraphContext, Node], LoweredOp],
    ) -> Callable[[Graph | GraphContext, Node], LoweredOp]:
        if op_type not in _LOWERING_REGISTRY:
            _LOWERING_REGISTRY[op_type] = func
        return func

    return decorator


def get_lowering(
    op_type: str,
) -> Callable[[Graph | GraphContext, Node], OpBase] | None:
    lowering = _LOWERING_REGISTRY.get(op_type)
    if lowering is not None:
        return lowering
    from . import load_lowering_registry

    load_lowering_registry()
    return _LOWERING_REGISTRY.get(op_type)


def get_lowering_registry() -> (
    Mapping[str, Callable[[Graph | GraphContext, Node], OpBase]]
):
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
