from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType


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
