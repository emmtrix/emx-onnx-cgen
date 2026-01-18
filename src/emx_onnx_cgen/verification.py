from __future__ import annotations

import numpy as np


def _float_uint_dtype(values: np.ndarray) -> type[np.unsignedinteger]:
    if values.dtype == np.float16:
        return np.uint16
    if values.dtype == np.float32:
        return np.uint32
    if values.dtype == np.float64:
        return np.uint64
    raise ValueError(f"Unsupported floating dtype for ULP calculation: {values.dtype}")


def _float_to_ordered_int(values: np.ndarray) -> np.ndarray:
    uint_dtype = _float_uint_dtype(values)
    bits = np.dtype(uint_dtype).itemsize * 8
    sign_mask = np.array(1 << (bits - 1), dtype=uint_dtype)
    as_uint = values.view(uint_dtype)
    ordered = np.where(as_uint & sign_mask, ~as_uint, as_uint | sign_mask)
    return ordered.astype(np.uint64, copy=False)


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
    nan_mask = np.isnan(actual_cast) | np.isnan(expected_cast)
    if nan_mask.any():
        both_nan = np.isnan(actual_cast) & np.isnan(expected_cast)
        if not np.all(both_nan):
            uint_dtype = _float_uint_dtype(expected_cast)
            return int(np.iinfo(uint_dtype).max)
        actual_cast = actual_cast[~nan_mask]
        expected_cast = expected_cast[~nan_mask]
        if actual_cast.size == 0:
            return 0
    ordered_actual = _float_to_ordered_int(actual_cast)
    ordered_expected = _float_to_ordered_int(expected_cast)
    deltas = ordered_actual.astype(np.int64) - ordered_expected.astype(np.int64)
    return int(np.max(np.abs(deltas)))


def format_success_message(max_ulp: int) -> str:
    return f"OK (max ULP {max_ulp})"
