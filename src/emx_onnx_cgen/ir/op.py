"""Phase 1 wrapper polymorphism for lowered ops.

This module introduces OpBase and WrapperOp while preserving legacy op objects
for the emitter. Behavior must remain identical to pre-refactor outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping


class OpBase(ABC):
    """Abstract base for lowered ops (Phase 1 wrapper polymorphism)."""

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def inputs(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def outputs(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def debug_name(self) -> str | None:
        raise NotImplementedError

    def as_inner(self) -> object:
        raise NotImplementedError(
            "OpBase subclasses must provide a legacy inner op via as_inner()."
        )


@dataclass(frozen=True)
class WrapperOp(OpBase):
    """Wrapper for legacy lowered op objects produced by existing lowerings."""

    inner: object

    @property
    def kind(self) -> str:
        kind = _first_attr_value(
            self.inner, ("kind", "op_type", "operator_kind")
        )
        if kind is None:
            return type(self.inner).__name__
        return str(kind)

    @property
    def inputs(self) -> tuple[str, ...]:
        return _extract_inputs(self.inner)

    @property
    def outputs(self) -> tuple[str, ...]:
        return _extract_outputs(self.inner)

    @property
    def debug_name(self) -> str | None:
        debug_name = _first_attr_value(
            self.inner, ("debug_name", "name", "node_name")
        )
        if debug_name is None:
            return None
        return str(debug_name)

    def as_inner(self) -> object:
        return self.inner


def _first_attr_value(obj: object, names: Iterable[str]) -> object | None:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _extract_inputs(obj: object) -> tuple[str, ...]:
    return _extract_names(
        obj,
        primary_fields=("inputs", "inps", "input_names"),
        numbered_prefix="input",
        fallback_fields=("input", "input0", "input1", "input_x", "input_y", "condition"),
    )


def _extract_outputs(obj: object) -> tuple[str, ...]:
    return _extract_names(
        obj,
        primary_fields=("outputs", "outs", "output_names"),
        numbered_prefix="output",
        fallback_fields=("output", "output0", "output1"),
    )


def _extract_names(
    obj: object,
    *,
    primary_fields: tuple[str, ...],
    numbered_prefix: str,
    fallback_fields: tuple[str, ...],
) -> tuple[str, ...]:
    for field in primary_fields:
        if hasattr(obj, field):
            value = getattr(obj, field)
            return _normalize_names(value)
    numbered = _extract_numbered(obj, numbered_prefix)
    if numbered:
        return numbered
    collected: list[str] = []
    for field in fallback_fields:
        if hasattr(obj, field):
            value = getattr(obj, field)
            collected.extend(_normalize_names(value))
    return tuple(collected)


def _extract_numbered(obj: object, prefix: str) -> tuple[str, ...]:
    candidates: dict[int, str] = {}
    for attr in dir(obj):
        if not attr.startswith(prefix):
            continue
        suffix = attr[len(prefix) :]
        if not suffix.isdigit():
            continue
        value = getattr(obj, attr)
        if isinstance(value, str):
            candidates[int(suffix)] = value
    if not candidates:
        return ()
    return tuple(value for _, value in sorted(candidates.items()))


def _normalize_names(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value.keys())
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return ()
