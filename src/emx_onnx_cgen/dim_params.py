"""Parse ONNX ``dim_param`` strings such as ``past_sequence_length + 1``."""

from __future__ import annotations

import re

import sympy

_DIM_PARAM_TOKEN = re.compile(
    r"\s*(?:(?P<int>\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*()]))"
)


def parse_dim_param(text: str) -> sympy.Expr | None:
    """Parse a dim_param such as ``past_sequence_length + 1`` into sympy.

    Only integers, identifiers, ``+``, ``-``, ``*`` and parentheses are accepted
    (no ``eval``); anything else returns ``None``. Identifiers become
    non-negative integer symbols, since they name tensor dimensions.
    """

    tokens: list[tuple[str, str]] = []
    position = 0
    text = text.rstrip()
    while position < len(text):
        match = _DIM_PARAM_TOKEN.match(text, position)
        if match is None or match.end() == position:
            return None
        kind = match.lastgroup
        assert kind is not None
        tokens.append((kind, match.group(kind)))
        position = match.end()
    if not tokens:
        return None
    parser = _DimParamParser(tokens)
    try:
        expression = parser.parse_sum()
    except ValueError:
        return None
    return expression if parser.at_end() else None


class _DimParamParser:
    """Recursive-descent parser for ``sum := product (('+'|'-') product)*``."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._index = 0

    def at_end(self) -> bool:
        return self._index == len(self._tokens)

    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._index] if self._index < len(self._tokens) else None

    def _take(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of dim_param")
        self._index += 1
        return token

    def parse_sum(self) -> sympy.Expr:
        result = self._parse_product()
        while self._peek() in {("op", "+"), ("op", "-")}:
            _, op = self._take()
            operand = self._parse_product()
            result = result + operand if op == "+" else result - operand
        return result

    def _parse_product(self) -> sympy.Expr:
        result = self._parse_atom()
        while self._peek() == ("op", "*"):
            self._take()
            result = result * self._parse_atom()
        return result

    def _parse_atom(self) -> sympy.Expr:
        kind, text = self._take()
        if kind == "int":
            return sympy.Integer(int(text))
        if kind == "name":
            return sympy.Symbol(text, integer=True, nonnegative=True)
        if text == "(":
            inner = self.parse_sum()
            if self._take() != ("op", ")"):
                raise ValueError("unbalanced parenthesis in dim_param")
            return inner
        raise ValueError(f"unexpected token {text!r} in dim_param")
