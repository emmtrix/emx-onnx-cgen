from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from emx_onnx_cgen.testbench import decode_testbench_array


def test_decode_testbench_array_parses_hex_for_bfloat16() -> None:
    dtype = ScalarType.BF16.np_dtype
    values = ["0x1.0p+0", "-0x1.8p+1", "0x1.2p+2"]

    decoded = decode_testbench_array(values, dtype)

    assert decoded.dtype == dtype
    np.testing.assert_allclose(
        decoded.astype(np.float32),
        np.array([1.0, -3.0, 4.5], dtype=np.float32),
        rtol=0.0,
        atol=1e-3,
    )
