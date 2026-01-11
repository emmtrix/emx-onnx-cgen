from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ..ir.model import Graph, Node

LoweredOp = TypeVar("LoweredOp")

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
