"""Backend tests for 8-bit floating point data types.

Float8 types (FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ,
FLOAT8E8M0) are stored as ``uint8_t`` in C with typedef aliases and use
manual conversion functions to/from ``float``.

Test coverage philosophy
========================
One backend test is provided per data type to verify the C ABI interfacing
and the correctness of the conversion functions.  Each test exercises the
data type in all three roles:

  - input  : a runtime-provided tensor flowing into the model
  - weight : a constant initializer (graph weight) embedded in the model
  - output : a result tensor produced by the model (via Cast)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.testbench import decode_testbench_array

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def _compile_and_run_testbench(
    model: onnx.ModelProto,
    *,
    testbench_inputs: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], str]:
    """Compile and run a testbench for a model that uses float8 types."""
    cc = os.environ.get("CC") or "cc"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        options = CompilerOptions(emit_testbench=True)
        compiler = Compiler(options)
        generated = compiler.compile(model)
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [cc, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
            check=True,
            capture_output=True,
            text=True,
        )
        run_cmd = [str(exe_path)]
        if testbench_inputs:
            input_path = temp_path / "inputs.bin"
            initializer_names = {init.name for init in model.graph.initializer}
            with input_path.open("wb") as handle:
                for value_info in model.graph.input:
                    if value_info.name in initializer_names:
                        continue
                    array = testbench_inputs.get(value_info.name)
                    if array is None:
                        raise AssertionError(
                            f"Missing testbench input {value_info.name}"
                        )
                    handle.write(np.ascontiguousarray(array).tobytes(order="C"))
            run_cmd.append(str(input_path))
        result = subprocess.run(
            run_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        generated = c_path.read_text(encoding="utf-8")
    return json.loads(result.stdout), generated


def _make_cast_model(
    from_dtype: int,
    to_dtype: int,
    shape: list[int],
    weight_values: list[float] | None = None,
    opset: int = 25,
) -> onnx.ModelProto:
    """Build a model that casts from_dtype to to_dtype.

    The graph has one runtime input ``x`` of ``from_dtype`` cast to ``to_dtype``.
    If ``weight_values`` is provided, an additional weight initializer ``w`` of
    ``from_dtype`` is also cast to ``to_dtype``.
    """
    x_info = helper.make_tensor_value_info("x", from_dtype, shape)
    y_info = helper.make_tensor_value_info("y", to_dtype, shape)
    nodes = [helper.make_node("Cast", ["x"], ["y"], to=to_dtype)]
    outputs = [y_info]
    initializers = []

    if weight_values is not None:
        w_info = helper.make_tensor_value_info("w", from_dtype, shape)
        yw_info = helper.make_tensor_value_info("y_from_weight", to_dtype, shape)
        weight_tensor = helper.make_tensor("w", from_dtype, shape, weight_values)
        nodes.append(helper.make_node("Cast", ["w"], ["y_from_weight"], to=to_dtype))
        outputs.append(yw_info)
        initializers.append(weight_tensor)

    graph = helper.make_graph(
        nodes,
        "test_graph",
        [x_info],
        outputs,
        initializer=initializers,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )


# -- FLOAT8E4M3FN backend test -----------------------------------------------


@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes required")
def test_float8e4m3fn_backend() -> None:
    """FLOAT8E4M3FN as input, weight, and output (via Cast from float)."""
    shape = [2, 3]
    f32_input = np.array([[1.0, 0.5, -0.5], [3.25, 0.0, -1.0]], dtype=np.float32)
    f32_weights = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5]
    model = _make_cast_model(
        TensorProto.FLOAT, TensorProto.FLOAT8E4M3FN, shape,
        weight_values=f32_weights,
    )
    payload, generated = _compile_and_run_testbench(
        model, testbench_inputs={"x": f32_input}
    )
    assert "emx_float8e4m3fn_t" in generated
    assert "ref_scalar_f8e4m3fn_from_f32" in generated

    # Decode output and compare with ml_dtypes reference
    out_data = decode_testbench_array(
        payload["outputs"]["y"]["data"], np.dtype(np.uint8)
    )
    expected = f32_input.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    np.testing.assert_array_equal(out_data, expected)

    out_weight = decode_testbench_array(
        payload["outputs"]["y_from_weight"]["data"], np.dtype(np.uint8)
    )
    expected_w = (
        np.array(f32_weights, dtype=np.float32)
        .reshape(shape)
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
    )
    np.testing.assert_array_equal(out_weight, expected_w)


# -- FLOAT8E4M3FNUZ backend test ---------------------------------------------


@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes required")
def test_float8e4m3fnuz_backend() -> None:
    """FLOAT8E4M3FNUZ as input, weight, and output (via Cast from float)."""
    shape = [2, 3]
    f32_input = np.array([[1.0, 0.5, -0.5], [3.25, 0.0, -1.0]], dtype=np.float32)
    f32_weights = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5]
    model = _make_cast_model(
        TensorProto.FLOAT, TensorProto.FLOAT8E4M3FNUZ, shape,
        weight_values=f32_weights,
    )
    payload, generated = _compile_and_run_testbench(
        model, testbench_inputs={"x": f32_input}
    )
    assert "emx_float8e4m3fnuz_t" in generated
    assert "ref_scalar_f8e4m3fnuz_from_f32" in generated

    out_data = decode_testbench_array(
        payload["outputs"]["y"]["data"], np.dtype(np.uint8)
    )
    expected = f32_input.astype(ml_dtypes.float8_e4m3fnuz).view(np.uint8)
    np.testing.assert_array_equal(out_data, expected)


# -- FLOAT8E5M2 backend test -------------------------------------------------


@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes required")
def test_float8e5m2_backend() -> None:
    """FLOAT8E5M2 as input, weight, and output (via Cast from float)."""
    shape = [2, 3]
    f32_input = np.array([[1.0, 0.5, -0.5], [3.0, 0.0, -1.0]], dtype=np.float32)
    f32_weights = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5]
    model = _make_cast_model(
        TensorProto.FLOAT, TensorProto.FLOAT8E5M2, shape,
        weight_values=f32_weights,
    )
    payload, generated = _compile_and_run_testbench(
        model, testbench_inputs={"x": f32_input}
    )
    assert "emx_float8e5m2_t" in generated
    assert "ref_scalar_f8e5m2_from_f32" in generated

    out_data = decode_testbench_array(
        payload["outputs"]["y"]["data"], np.dtype(np.uint8)
    )
    expected = f32_input.astype(ml_dtypes.float8_e5m2).view(np.uint8)
    np.testing.assert_array_equal(out_data, expected)


# -- FLOAT8E5M2FNUZ backend test ---------------------------------------------


@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes required")
def test_float8e5m2fnuz_backend() -> None:
    """FLOAT8E5M2FNUZ as input, weight, and output (via Cast from float)."""
    shape = [2, 3]
    f32_input = np.array([[1.0, 0.5, -0.5], [3.0, 0.0, -1.0]], dtype=np.float32)
    f32_weights = [1.0, -1.0, 0.0, 2.0, -2.0, 0.5]
    model = _make_cast_model(
        TensorProto.FLOAT, TensorProto.FLOAT8E5M2FNUZ, shape,
        weight_values=f32_weights,
    )
    payload, generated = _compile_and_run_testbench(
        model, testbench_inputs={"x": f32_input}
    )
    assert "emx_float8e5m2fnuz_t" in generated
    assert "ref_scalar_f8e5m2fnuz_from_f32" in generated

    out_data = decode_testbench_array(
        payload["outputs"]["y"]["data"], np.dtype(np.uint8)
    )
    expected = f32_input.astype(ml_dtypes.float8_e5m2fnuz).view(np.uint8)
    np.testing.assert_array_equal(out_data, expected)


# -- FLOAT8E8M0FNU backend test ----------------------------------------------


@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes required")
def test_float8e8m0fnu_backend() -> None:
    """FLOAT8E8M0FNU as input, weight, and output (via Cast from float)."""
    shape = [2, 3]
    # E8M0 only stores powers of 2 (unsigned)
    f32_input = np.array([[1.0, 2.0, 4.0], [0.5, 8.0, 16.0]], dtype=np.float32)
    f32_weights = [1.0, 2.0, 4.0, 0.5, 0.25, 8.0]
    model = _make_cast_model(
        TensorProto.FLOAT, TensorProto.FLOAT8E8M0, shape,
        weight_values=f32_weights,
    )
    payload, generated = _compile_and_run_testbench(
        model, testbench_inputs={"x": f32_input}
    )
    assert "emx_float8e8m0fnu_t" in generated
    assert "ref_scalar_f8e8m0fnu_from_f32" in generated

    out_data = decode_testbench_array(
        payload["outputs"]["y"]["data"], np.dtype(np.uint8)
    )
    expected = f32_input.astype(ml_dtypes.float8_e8m0fnu).view(np.uint8)
    np.testing.assert_array_equal(out_data, expected)
