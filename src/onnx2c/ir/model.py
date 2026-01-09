from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Value:
    name: str
    type: TensorType


@dataclass(frozen=True)
class Node:
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Graph:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    nodes: tuple[Node, ...]

    def find_value(self, name: str) -> Value:
        for value in self.inputs + self.outputs:
            if value.name == name:
                return value
        raise KeyError(name)
