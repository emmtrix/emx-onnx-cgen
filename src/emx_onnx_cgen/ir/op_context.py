from __future__ import annotations

from dataclasses import dataclass, field

from shared.scalar_types import ScalarType

from .context import GraphContext


_MISSING = object()


@dataclass
class OpContext:
    graph: GraphContext
    _dtype_overrides: dict[str, ScalarType] = field(default_factory=dict)
    _shape_overrides: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _derived: dict[int, dict[str, object]] = field(default_factory=dict)

    def dtype(self, name: str) -> ScalarType:
        if name in self._dtype_overrides:
            return self._dtype_overrides[name]
        return self.graph.dtype(name)

    def shape(self, name: str) -> tuple[int, ...]:
        if name in self._shape_overrides:
            return self._shape_overrides[name]
        return self.graph.shape(name)

    def set_dtype(self, name: str, dtype: ScalarType) -> None:
        self._dtype_overrides[name] = dtype
        self.graph.set_dtype(name, dtype)

    def set_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self._shape_overrides[name] = shape
        self.graph.set_shape(name, shape)

    def set_derived(self, op: object, key: str, value: object) -> None:
        self._derived.setdefault(id(op), {})[key] = value

    def get_derived(
        self, op: object, key: str, default: object = _MISSING
    ) -> object:
        derived = self._derived.get(id(op), {})
        if key in derived:
            return derived[key]
        if default is _MISSING:
            return _MISSING
        return default

    def require_derived(self, op: object, key: str) -> object:
        derived = self._derived.get(id(op), {})
        if key in derived:
            return derived[key]
        raise KeyError(
            f"Missing derived value '{key}' for op {op.__class__.__name__}"
        )

    def copy_derived(self, source_op: object, target_op: object) -> None:
        derived = self._derived.get(id(source_op))
        if derived:
            self._derived[id(target_op)] = dict(derived)

    def __getattr__(self, name: str):
        return getattr(self.graph, name)
