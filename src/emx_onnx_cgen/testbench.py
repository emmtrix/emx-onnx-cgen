from __future__ import annotations

from typing import Any

import numpy as np


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
