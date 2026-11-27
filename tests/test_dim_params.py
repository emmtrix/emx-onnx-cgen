from __future__ import annotations

import pytest
import sympy

from emx_onnx_cgen.dim_params import parse_dim_param


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n", "n"),
        ("past_sequence_length + 1", "past_sequence_length + 1"),
        ("2*(a + b) - 3", "2*a + 2*b - 3"),
    ],
)
def test_parse_dim_param(text: str, expected: str) -> None:
    assert str(sympy.expand(parse_dim_param(text))) == expected


@pytest.mark.parametrize("text", ["n/2", "1e3", "n+", "(n", "unk__3 ?"])
def test_parse_dim_param_rejects_other_syntax(text: str) -> None:
    assert parse_dim_param(text) is None
