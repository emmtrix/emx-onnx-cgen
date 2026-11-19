from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.scalar_types import ScalarType

if TYPE_CHECKING:
    from .op_base import OpBase
    from .op_context import OpContext


@dataclass(frozen=True)
class StatePair:
    """A graph input and a graph output that share one persistent buffer."""

    input_name: str
    output_name: str

    def format(self) -> str:
        return f"{self.input_name}={self.output_name}"


@dataclass(frozen=True)
class StateBinding:
    """A validated :class:`StatePair` with the tensor type of its buffer.

    After state aliasing, every op refers to the buffer by ``input_name``;
    ``output_name`` only remains as the name of the graph output.
    """

    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: ScalarType


def index_input_bounds(
    ops: Iterable["OpBase"], ctx: "OpContext", input_names: Sequence[str]
) -> dict[str, int]:
    """Exclusive upper bounds of graph inputs that ops use as indices.

    Test drivers draw such inputs from ``[0, bound)`` so that a random value
    never addresses memory outside the buffer it indexes.
    """
    graph_inputs = set(input_names)
    bounds: dict[str, int] = {}
    for op in ops:
        for name, bound in op.index_input_bounds(ctx).items():
            if name in graph_inputs:
                bounds[name] = min(bound, bounds.get(name, bound))
    return bounds
