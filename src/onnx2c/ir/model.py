from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from shared.scalar_types import ScalarType


@dataclass(frozen=True)
class TensorType:
    dtype: ScalarType
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
    attrs: Mapping[str, object]


@dataclass(frozen=True)
class Initializer:
    name: str
    type: TensorType
    data: np.ndarray


@dataclass(frozen=True)
class Graph:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    nodes: tuple[Node, ...]
    initializers: tuple[Initializer, ...]
    values: tuple[Value, ...] = ()

    def find_value(self, name: str) -> Value:
        for value in self.inputs + self.outputs + self.values:
            if value.name == name:
                return value
        for initializer in self.initializers:
            if initializer.name == name:
                return Value(name=initializer.name, type=initializer.type)
        raise KeyError(name)
