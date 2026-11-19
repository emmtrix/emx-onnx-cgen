from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StateTestbenchData:
    """Values for the multi-step testbench of a model with state pairs.

    The testbench resets the state, loads ``initial_state`` (per state input;
    missing entries stay zero) and calls the model ``steps`` times. Every other
    graph input takes its value of each call from ``step_inputs``, whose
    arrays carry a leading axis of length ``steps``; inputs without an entry
    are filled with random values.
    """

    steps: int
    initial_state: Mapping[str, np.ndarray] = field(default_factory=dict)
    step_inputs: Mapping[str, np.ndarray] = field(default_factory=dict)


def _convert_hex_floats(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert_hex_floats(item) for item in value]
    if isinstance(value, str):
        return float.fromhex(value)
    return value


def _is_float_like_dtype(dtype: np.dtype) -> bool:
    resolved = np.dtype(dtype)
    return np.issubdtype(resolved, np.floating) or resolved.name == "bfloat16"


def _is_float8_dtype(dtype: np.dtype) -> bool:
    resolved = np.dtype(dtype)
    return resolved.name.startswith("float8_") or resolved.name.startswith("float4_")


def decode_testbench_array(data: object, dtype: np.dtype) -> np.ndarray:
    """Decode testbench JSON data into a numpy array.

    Floating-point values are expected to be hex strings (C99 %a formatting).
    Float8 values are emitted as semantic float hex strings as well.
    """
    if _is_float8_dtype(dtype):
        return np.array(_convert_hex_floats(data), dtype=dtype)
    if _is_float_like_dtype(dtype):
        data = _convert_hex_floats(data)
    return np.array(data, dtype=dtype)
