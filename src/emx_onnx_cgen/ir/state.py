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

    A *growing* pair (a KV cache appended with ``Concat``) keeps both names:
    the input is the first ``n`` rows of the buffer along ``growth_axis`` and
    the output the first ``n + growth`` rows. ``shape`` is then the storage
    shape with the capacity on ``growth_axis``.
    """

    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: ScalarType
    growth_axis: int | None = None
    growth: int = 0

    @property
    def is_growing(self) -> bool:
        return self.growth_axis is not None


def index_input_bounds(
    ops: Iterable["OpBase"], ctx: "OpContext", input_names: Sequence[str]
) -> dict[str, int]:
    """Exclusive upper bounds of graph inputs that ops use as indices.

    Test drivers draw such inputs from ``[0, bound)`` so that a random value
    never addresses memory outside the buffer it indexes. A bound also holds
    for the values an index is copied from (``Cast``, ``Reshape``, ...; see
    :meth:`OpBase.value_source_inputs`), so it reaches the graph input behind
    e.g. ``pos -> Cast -> write_indices``.
    """
    bounds: dict[str, int] = {}

    def tighten(name: str, bound: int) -> None:
        bounds[name] = min(bound, bounds.get(name, bound))

    # Consumers come after producers, so one reverse pass propagates bounds
    # from every index use back to its sources.
    for op in reversed(list(ops)):
        for name, bound in op.index_input_bounds(ctx).items():
            tighten(name, bound)
        output_bounds = [bounds[name] for name in op.output_names if name in bounds]
        if output_bounds:
            for source in op.value_source_inputs():
                tighten(source, min(output_bounds))
    graph_inputs = set(input_names)
    return {name: bound for name, bound in bounds.items() if name in graph_inputs}
