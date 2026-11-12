from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Node

_NUM_DIRECTIONS_BY_DIRECTION = {"forward": 1, "reverse": 1, "bidirectional": 2}


def resolve_direction(node: Node, num_directions: int) -> str:
    """The ``direction`` attribute of a recurrent op, checked against W's num_directions.

    ONNX stores string attributes as bytes.
    """
    direction = node.attrs.get("direction", "forward")
    if isinstance(direction, bytes):
        direction = direction.decode("utf-8")
    if not isinstance(direction, str):
        raise UnsupportedOpError(f"{node.op_type} direction must be a string")
    expected = _NUM_DIRECTIONS_BY_DIRECTION.get(direction)
    if expected is None:
        raise UnsupportedOpError(f"Unsupported {node.op_type} direction {direction}")
    if num_directions != expected:
        raise UnsupportedOpError(
            f"{node.op_type} with direction={direction} expects "
            f"num_directions={expected}, got {num_directions}"
        )
    return direction
