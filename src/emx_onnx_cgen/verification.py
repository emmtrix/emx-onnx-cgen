from __future__ import annotations

import numpy as np

from shared.ulp import ulp_intdiff_float


def max_ulp_diff(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for ULP calculation: {actual.shape} vs {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.floating):
        return 0
    dtype = expected.dtype
    if dtype not in (np.float16, np.float32, np.float64):
        raise ValueError(f"Unsupported floating dtype for ULP calculation: {dtype}")
    actual_cast = actual.astype(dtype, copy=False)
    expected_cast = expected.astype(dtype, copy=False)
    max_diff = 0
    for actual_value, expected_value in np.nditer(
        [actual_cast, expected_cast], flags=["refs_ok"]
    ):
        diff = ulp_intdiff_float(actual_value[()], expected_value[()])
        if diff > max_diff:
            max_diff = diff
    return max_diff


def format_success_message(max_ulp: int) -> str:
    return f"OK (max ULP {max_ulp})"
