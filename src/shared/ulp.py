from __future__ import annotations

from typing import Dict

import numpy as np

from shared.scalar_types import ScalarFunctionError

_FLOAT_TO_UINT: Dict[np.dtype, np.dtype] = {
    np.dtype("float16"): np.dtype("uint16"),
    np.dtype("float32"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("uint64"),
}


def _coerce_float_scalar(value: object, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(())


def _ulp_intdiff_same_sign(
    f1: np.ndarray, f2: np.ndarray, uint_dtype: np.dtype
) -> int:
    i1 = f1.view(uint_dtype).item()
    i2 = f2.view(uint_dtype).item()
    return int(i1 - i2) if i1 > i2 else int(i2 - i1)


def ulp_intdiff_float(f1: object, f2: object) -> int:
    dtype = np.result_type(f1, f2)
    try:
        uint_dtype = _FLOAT_TO_UINT[dtype]
    except KeyError as exc:
        raise ScalarFunctionError(
            f"unsupported dtype for ULP diff: {dtype}"
        ) from exc

    f1_scalar = _coerce_float_scalar(f1, dtype)
    f2_scalar = _coerce_float_scalar(f2, dtype)

    if np.signbit(f1_scalar) != np.signbit(f2_scalar):
        zero = _coerce_float_scalar(0.0, dtype)
        return (
            ulp_intdiff_float(zero, np.abs(f1_scalar))
            + ulp_intdiff_float(zero, np.abs(f2_scalar))
            + 1
        )

    return _ulp_intdiff_same_sign(f1_scalar, f2_scalar, uint_dtype)
