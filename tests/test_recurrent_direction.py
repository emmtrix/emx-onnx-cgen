from __future__ import annotations

import pytest

from emx_onnx_cgen.errors import UnsupportedOpError
from emx_onnx_cgen.ir.model import Node
from emx_onnx_cgen.lowering.recurrent_direction import resolve_direction


def _node(op_type: str, **attrs: object) -> Node:
    return Node(
        op_type=op_type, name=None, inputs=("X", "W", "R"), outputs=("Y",), attrs=attrs
    )


@pytest.mark.parametrize(
    ("direction", "num_directions"),
    [(b"forward", 1), (b"reverse", 1), (b"bidirectional", 2), ("reverse", 1)],
)
def test_resolve_direction_decodes_onnx_string_attribute(
    direction: bytes | str, num_directions: int
) -> None:
    expected = direction.decode() if isinstance(direction, bytes) else direction
    node = _node("GRU", direction=direction)
    assert resolve_direction(node, num_directions) == expected


def test_resolve_direction_defaults_to_forward() -> None:
    assert resolve_direction(_node("RNN"), 1) == "forward"


@pytest.mark.parametrize(
    ("direction", "num_directions", "message"),
    [
        (b"sideways", 1, "Unsupported LSTM direction sideways"),
        (b"bidirectional", 1, "expects num_directions=2, got 1"),
        (b"reverse", 2, "expects num_directions=1, got 2"),
    ],
)
def test_resolve_direction_rejects_invalid_direction(
    direction: bytes, num_directions: int, message: str
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        resolve_direction(_node("LSTM", direction=direction), num_directions)
