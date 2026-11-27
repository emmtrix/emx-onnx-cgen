"""C spelling of the named dynamic dimensions (``dim_param``) of a model.

ONNX allows any string as a ``dim_param``. Exporters write expressions such as
``past_sequence_length + 1`` for a dimension that depends on another one. The
generated code needs a C spelling for every name:

* a C identifier (``batch``) is a runtime parameter of the model function;
* an expression whose symbols are all such parameters (``past_sequence_length
  + 1``) is *derived*: it is written as the parenthesized C expression and has
  no parameter of its own, so the caller cannot pass an inconsistent value;
* anything else becomes a runtime parameter with a sanitized identifier.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import sympy

from ..dim_params import parse_dim_param

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class DimSpelling:
    """How one ``dim_param`` appears in C."""

    text: str
    expression: sympy.Expr | None = None

    @property
    def is_derived(self) -> bool:
        return self.expression is not None

    def expected_size(self, parameter_sizes: Mapping[str, int], default: int) -> int:
        """Size of a derived dimension for the given parameter sizes."""

        if self.expression is None:
            return default
        substituted = self.expression.subs(
            {
                symbol: parameter_sizes.get(symbol.name, default)
                for symbol in self.expression.free_symbols
            }
        )
        return max(int(substituted), 0) if substituted.is_Integer else default


def is_derived_dim_text(text: str) -> bool:
    """Whether a C dim spelling is a derived expression, not a parameter."""

    return text.startswith("(")


def derived_dim_size(text: str, sizes: Mapping[str, int], default: int) -> int:
    """Size of the derived C dim ``text`` (e.g. ``(p + 1)``) for given sizes."""

    expression = parse_dim_param(text)
    if expression is None:
        return default
    return DimSpelling(text, expression=expression).expected_size(sizes, default)


def spell_dim_params(dim_params: Iterable[str]) -> dict[str, DimSpelling]:
    """Map every ``dim_param`` to its C spelling (see module docstring)."""

    names = sorted(set(dim_params))
    identifiers = {name for name in names if _IDENTIFIER.match(name)}
    spellings: dict[str, DimSpelling] = {}
    for name in names:
        if name in identifiers:
            spellings[name] = DimSpelling(name)
            continue
        expression = parse_dim_param(name)
        if (
            expression is not None
            and expression.free_symbols
            and all(symbol.name in identifiers for symbol in expression.free_symbols)
        ):
            spellings[name] = DimSpelling(
                f"({sympy.ccode(expression)})", expression=expression
            )
            continue
        spellings[name] = DimSpelling(_sanitize(name))
    return spellings


def _sanitize(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_") or "dim"
    if sanitized[0].isdigit():
        sanitized = f"dim_{sanitized}"
    return sanitized
