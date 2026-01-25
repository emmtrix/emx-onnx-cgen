from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ShapeInferenceError, UnsupportedOpError
from .model import Graph, Initializer, Node, Value
from shared.scalar_types import ScalarType


@dataclass
class GraphContext:
    graph: Graph
    _dtype_cache: dict[str, ScalarType] = field(default_factory=dict)
    _shape_cache: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _initializer_cache: dict[str, Initializer] = field(default_factory=dict)
    _producer_cache: dict[str, Node] = field(default_factory=dict)
    _value_cache: dict[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in self.graph.inputs + self.graph.outputs + self.graph.values:
            self._value_cache[value.name] = value
        for initializer in self.graph.initializers:
            if initializer.name not in self._value_cache:
                self._value_cache[initializer.name] = Value(
                    name=initializer.name,
                    type=initializer.type,
                )
            self._initializer_cache[initializer.name] = initializer
        for node in self.graph.nodes:
            for output in node.outputs:
                if output and output not in self._producer_cache:
                    self._producer_cache[output] = node

    def find_value(self, name: str) -> Value:
        value = self._value_cache.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def dtype(self, name: str, node: Node | None = None) -> ScalarType:
        if name in self._dtype_cache:
            return self._dtype_cache[name]
        try:
            value = self.graph.find_value(name)
        except KeyError as exc:
            op_type = node.op_type if node is not None else "unknown"
            raise ShapeInferenceError(
                f"Missing dtype for value '{name}' in op {op_type}. "
                "Hint: run ONNX shape inference or export with static shapes."
            ) from exc
        dtype = value.type.dtype
        if not isinstance(dtype, ScalarType):
            raise UnsupportedOpError(f"Unsupported dtype {dtype}")
        self._dtype_cache[name] = dtype
        return dtype

    def set_dtype(self, name: str, dtype: ScalarType) -> None:
        self._dtype_cache[name] = dtype

    def shape(self, name: str, node: Node | None = None) -> tuple[int, ...]:
        if name in self._shape_cache:
            return self._shape_cache[name]
        try:
            value = self.graph.find_value(name)
        except KeyError as exc:
            op_type = node.op_type if node is not None else "unknown"
            raise ShapeInferenceError(
                f"Missing shape for value '{name}' in op {op_type}. "
                "Hint: run ONNX shape inference or export with static shapes."
            ) from exc
        self._shape_cache[name] = value.type.shape
        return value.type.shape

    def set_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self._shape_cache[name] = shape

    def initializer(self, name: str) -> Initializer | None:
        return self._initializer_cache.get(name)

    def producer(self, output_name: str) -> Node | None:
        return self._producer_cache.get(output_name)

    def opset_version(self, domain: str = "") -> int | None:
        if domain in {"", "ai.onnx"}:
            domains = {"", "ai.onnx"}
        else:
            domains = {domain}
        for opset_domain, version in self.graph.opset_imports:
            if opset_domain in domains:
                return int(version)
        return None

    def __getattr__(self, name: str):
        return getattr(self.graph, name)
