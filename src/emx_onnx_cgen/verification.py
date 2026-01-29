from __future__ import annotations

from typing import TypeAlias

import numpy as np

from shared.ulp import ulp_intdiff_float

WorstUlpDiff: TypeAlias = tuple[tuple[int, ...], float, float]


def _validate_ulp_inputs(
    actual: np.ndarray, expected: np.ndarray
) -> np.dtype | None:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for ULP calculation: {actual.shape} vs {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.floating):
        return None
    dtype = expected.dtype
    if dtype not in (np.float16, np.float32, np.float64):
        raise ValueError(f"Unsupported floating dtype for ULP calculation: {dtype}")
    return dtype


def worst_ulp_diff(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol_eps: float = 1.0,
) -> tuple[int, WorstUlpDiff | None]:
    dtype = _validate_ulp_inputs(actual, expected)
    if dtype is None:
        return 0, None
    if actual.size == 0:
        return 0, None
    actual_cast = actual.astype(dtype, copy=False)
    expected_cast = expected.astype(dtype, copy=False)
    abs_tol = np.finfo(dtype).eps * atol_eps
    max_diff = 0
    worst: WorstUlpDiff | None = None
    iterator = np.nditer(
        [actual_cast, expected_cast], flags=["refs_ok", "multi_index"]
    )
    for actual_value, expected_value in iterator:
        if (
            abs(float(actual_value[()]) - float(expected_value[()]))
            <= abs_tol
        ):
            continue
        diff = ulp_intdiff_float(actual_value[()], expected_value[()])
        if diff > max_diff:
            max_diff = diff
            worst = (
                iterator.multi_index,
                float(actual_value[()]),
                float(expected_value[()]),
            )
    return max_diff, worst


def max_ulp_diff(
    actual: np.ndarray, expected: np.ndarray, *, atol_eps: float = 1.0
) -> int:
    max_diff, _ = worst_ulp_diff(actual, expected, atol_eps=atol_eps)
    return max_diff


def format_success_message(max_ulp: int) -> str:
    return f"OK (max ULP {max_ulp})"
