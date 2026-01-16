from __future__ import annotations

import numpy as np
import pytest

from shared.scalar_types import ScalarFunctionError
from shared.ulp import ulp_intdiff_float


def _expected_same_sign_diff(f1: np.ndarray, f2: np.ndarray) -> int:
    uint_dtype = {
        np.dtype("float16"): np.dtype("uint16"),
        np.dtype("float32"): np.dtype("uint32"),
        np.dtype("float64"): np.dtype("uint64"),
    }[f1.dtype]
    i1 = f1.view(uint_dtype).item()
    i2 = f2.view(uint_dtype).item()
    return int(i1 - i2) if i1 > i2 else int(i2 - i1)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_ulp_intdiff_same_sign_neighbors(dtype: np.dtype) -> None:
    base = np.array(1.0, dtype=dtype)
    next_val = np.nextafter(base, np.array(2.0, dtype=dtype))
    assert ulp_intdiff_float(base, next_val) == 1
    assert (
        ulp_intdiff_float(base, base)
        == _expected_same_sign_diff(base, base)
        == 0
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_ulp_intdiff_cross_sign(dtype: np.dtype) -> None:
    pos = np.array(1.0, dtype=dtype)
    neg = np.array(-1.0, dtype=dtype)
    zero = np.array(0.0, dtype=dtype)
    expected = (
        _expected_same_sign_diff(zero, np.abs(pos))
        + _expected_same_sign_diff(zero, np.abs(neg))
        + 1
    )
    assert ulp_intdiff_float(pos, neg) == expected


def test_ulp_intdiff_rejects_non_float() -> None:
    with pytest.raises(ScalarFunctionError):
        ulp_intdiff_float(np.array(1, dtype=np.int32), np.array(2, dtype=np.int32))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_ulp_intdiff_examples(dtype: np.dtype) -> None:
    dtype = np.dtype(dtype)
    finfo = np.finfo(dtype)
    eps = finfo.eps
    min_sub = np.nextafter(dtype.type(0.0), dtype.type(1.0))
    max_val = finfo.max
    if dtype == np.dtype("float16"):
        cases = [
            (np.float16(1.0), np.float16(1.0) + eps, 1),
            (np.float16(1.0), np.float16(1.0) - eps, 2),
            (min_sub, np.float16(0.0), 1),
            (-min_sub, min_sub, 3),
            (np.float16(-1.0), np.float16(-1.0) - eps, 1),
            (np.float16(-1.0), np.float16(-1.0) + eps, 2),
            (np.float16(-0.0), np.float16(0.0), 1),
            (np.float16(0.0), np.float16(0.0), 0),
            (np.float16(max_val), np.float16(np.inf), 1),
            (np.float16(-max_val), np.float16(-np.inf), 1),
            (np.float16(1.0), np.float16(0.5), 1024),
            (np.float16(1.0), np.float16(2.0), 1024),
        ]
    elif dtype == np.dtype("float32"):
        cases = [
            (np.float32(1.0), np.float32(1.0) + eps, 1),
            (np.float32(1.0), np.float32(1.0) - eps, 2),
            (min_sub, np.float32(0.0), 1),
            (-min_sub, min_sub, 3),
            (np.float32(-1.0), np.float32(-1.0) - eps, 1),
            (np.float32(-1.0), np.float32(-1.0) + eps, 2),
            (np.float32(-0.0), np.float32(0.0), 1),
            (np.float32(0.0), np.float32(0.0), 0),
            (np.float32(max_val), np.float32(np.inf), 1),
            (np.float32(-max_val), np.float32(-np.inf), 1),
            (np.float32(1.0), np.float32(0.5), 8388608),
            (np.float32(1.0), np.float32(2.0), 8388608),
        ]
    elif dtype == np.dtype("float64"):
        cases = [
            (np.float64(1.0), np.float64(1.0) + eps, 1),
            (np.float64(1.0), np.float64(1.0) - eps, 2),
            (min_sub, np.float64(0.0), 1),
            (-min_sub, min_sub, 3),
            (np.float64(-1.0), np.float64(-1.0) - eps, 1),
            (np.float64(-1.0), np.float64(-1.0) + eps, 2),
            (np.float64(-0.0), np.float64(0.0), 1),
            (np.float64(0.0), np.float64(0.0), 0),
            (np.float64(max_val), np.float64(np.inf), 1),
            (np.float64(-max_val), np.float64(-np.inf), 1),
            (np.float64(1.0), np.float64(0.5), 4503599627370496),
            (np.float64(1.0), np.float64(2.0), 4503599627370496),
        ]
    else:
        raise AssertionError(f"unexpected dtype: {dtype}")

    for value1, value2, expected in cases:
        assert ulp_intdiff_float(value1, value2) == expected
