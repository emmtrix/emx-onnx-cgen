from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


_SEQUENCE_SHAPE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<shape>\[.*\])$")
_DYNAMIC_DIM_RE = re.compile(r"^<=(?P<value>[1-9][0-9]*)$")
_STATIC_DIM_RE = re.compile(r"^(?P<value>[1-9][0-9]*)$")


@dataclass(frozen=True)
class SequenceElementDimHint:
    max_size: int
    is_static: bool

    def format(self) -> str:
        if self.is_static:
            return str(self.max_size)
        return f"<={self.max_size}"


@dataclass(frozen=True)
class SequenceElementShapeHint:
    input_name: str
    dims: tuple[SequenceElementDimHint, ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def max_shape(self) -> tuple[int, ...]:
        return tuple(dim.max_size for dim in self.dims)

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        return tuple(index for index, dim in enumerate(self.dims) if not dim.is_static)

    def format(self) -> str:
        return "[" + ",".join(dim.format() for dim in self.dims) + "]"


def parse_sequence_element_shape_hint(value: str) -> SequenceElementShapeHint:
    match = _SEQUENCE_SHAPE_RE.match(value.strip())
    if match is None:
        raise ValueError(
            "Sequence element shape must be formatted as NAME=[dim,...], "
            "for example sequence=[<=8] or boxes=[<=100,4]."
        )
    name = match.group("name")
    shape_text = match.group("shape")
    inner = shape_text[1:-1].strip()
    if not inner:
        dims: tuple[SequenceElementDimHint, ...] = ()
    else:
        parsed_dims: list[SequenceElementDimHint] = []
        for token in inner.split(","):
            item = token.strip()
            dynamic_match = _DYNAMIC_DIM_RE.match(item)
            if dynamic_match is not None:
                parsed_dims.append(
                    SequenceElementDimHint(
                        max_size=int(dynamic_match.group("value")),
                        is_static=False,
                    )
                )
                continue
            static_match = _STATIC_DIM_RE.match(item)
            if static_match is not None:
                parsed_dims.append(
                    SequenceElementDimHint(
                        max_size=int(static_match.group("value")),
                        is_static=True,
                    )
                )
                continue
            raise ValueError(
                f"Invalid sequence element dimension {item!r} in {value!r}. "
                "Use either N or <=N."
            )
        dims = tuple(parsed_dims)
    return SequenceElementShapeHint(input_name=name, dims=dims)


def parse_sequence_element_shape_hints(
    values: Sequence[str] | None,
) -> dict[str, SequenceElementShapeHint]:
    if not values:
        return {}
    hints: dict[str, SequenceElementShapeHint] = {}
    for raw_value in values:
        hint = parse_sequence_element_shape_hint(raw_value)
        existing = hints.get(hint.input_name)
        if existing is not None:
            raise ValueError(
                "Duplicate sequence element shape hint for "
                f"{hint.input_name!r}: {existing.format()} and {hint.format()}."
            )
        hints[hint.input_name] = hint
    return hints


def format_runtime_shape(shape: Sequence[int]) -> str:
    return "[" + ",".join(str(int(dim)) for dim in shape) + "]"


def validate_runtime_shape(
    hint: SequenceElementShapeHint,
    actual_shape: Sequence[int],
) -> str | None:
    if len(actual_shape) != hint.rank:
        return (
            f"shape {format_runtime_shape(actual_shape)} has rank {len(actual_shape)}, "
            f"expected {hint.format()}"
        )
    for axis, (actual_dim, dim_hint) in enumerate(zip(actual_shape, hint.dims)):
        if actual_dim < 0:
            return (
                f"shape {format_runtime_shape(actual_shape)} has invalid negative dim "
                f"at axis {axis}"
            )
        if dim_hint.is_static and int(actual_dim) != dim_hint.max_size:
            return (
                f"shape {format_runtime_shape(actual_shape)} does not match expected "
                f"{hint.format()} at axis {axis}"
            )
        if not dim_hint.is_static and int(actual_dim) > dim_hint.max_size:
            return (
                f"shape {format_runtime_shape(actual_shape)} exceeds expected "
                f"{hint.format()} at axis {axis}"
            )
    return None


def hint_max_sizes_by_symbol(
    *,
    input_hints: Mapping[str, SequenceElementShapeHint],
    input_dim_params: Mapping[str, tuple[str | None, ...]],
) -> dict[str, int]:
    symbol_sizes: dict[str, int] = {}
    for input_name, hint in input_hints.items():
        dim_params = input_dim_params.get(input_name, ())
        for axis, dim_hint in enumerate(hint.dims):
            if axis >= len(dim_params):
                continue
            dim_param = dim_params[axis]
            if not dim_param:
                continue
            current = symbol_sizes.get(dim_param)
            if current is None or dim_hint.max_size > current:
                symbol_sizes[dim_param] = dim_hint.max_size
    return symbol_sizes
