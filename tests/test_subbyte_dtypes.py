"""Backend tests for sub-byte integer data types (INT2, UINT2, INT4, UINT4).

These types use C23 _BitInt and require a compiler with _BitInt support
(e.g., Clang 15+).

Test coverage philosophy
========================
One backend test is provided per data type to verify the C ABI interfacing.
Each test must exercise the data type in all three roles:

  - input  : a runtime-provided tensor flowing into the model
  - weight : a constant initializer (graph weight) embedded in the model
  - output : a result tensor produced by the model

Additional tests per data type are added only when a single model cannot
exercise all three roles simultaneously.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.testbench import decode_testbench_array


def _find_bitint_compiler() -> str | None:
    """Return a C compiler command that supports _BitInt, or None."""
    for cmd in (
        os.environ.get("CC"),
        shutil.which("clang"),
        shutil.which("cc"),
        shutil.which("gcc"),
    ):
        if cmd is None:
            continue
        try:
            result = subprocess.run(
                [cmd, "-std=c23", "-x", "c", "-", "-o", "/dev/null"],
                input="int main(void){ _BitInt(4) x = 1; return (int)x; }\n",
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


_BITINT_CC = _find_bitint_compiler()


def _compile_and_run_bitint_testbench(
    model: onnx.ModelProto,
    *,
    testbench_inputs: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], str]:
    """Compile and run a testbench for a model that uses _BitInt types."""
    if _BITINT_CC is None:
        pytest.skip("No C compiler with _BitInt support available")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        options = CompilerOptions(emit_testbench=True)
        compiler = Compiler(options)
        generated = compiler.compile(model)
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [_BITINT_CC, "-std=c23", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
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


def _make_identity_with_weight_model(
    dtype: int,
    shape: list[int],
    weight_values: list[int],
    opset: int = 21,
) -> onnx.ModelProto:
    """Build a model that exercises ``dtype`` as input, weight, and output.

    The graph has one runtime input ``x`` of ``dtype``, one constant
    initializer ``w`` of ``dtype``, and two outputs produced by passing
    each through Identity.  This covers the data type in all three roles
    that the interfacing test requires.
    """
    x_info = helper.make_tensor_value_info("x", dtype, shape)
    y_input_info = helper.make_tensor_value_info("y_from_input", dtype, shape)
    y_weight_info = helper.make_tensor_value_info("y_from_weight", dtype, shape)
    weight_tensor = helper.make_tensor("w", dtype, shape, weight_values)
    node_input = helper.make_node("Identity", ["x"], ["y_from_input"])
    node_weight = helper.make_node("Identity", ["w"], ["y_from_weight"])
    graph = helper.make_graph(
        [node_input, node_weight],
        "test_graph",
        [x_info],
        [y_input_info, y_weight_info],
        initializer=[weight_tensor],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )


# -- INT4 backend test --------------------------------------------------------


def test_int4_backend() -> None:
    """INT4 as input, weight (initializer), and output."""
    shape = [2, 3]
    input_data = np.array([[-7, 3, 0], [1, -8, 7]], dtype=np.int8)
    weight_values = [1, -1, 0, 3, -3, 7]
    model = _make_identity_with_weight_model(
        TensorProto.INT4, shape, weight_values
    )
    payload, generated = _compile_and_run_bitint_testbench(
        model, testbench_inputs={"x": input_data}
    )
    assert "_BitInt(4)" in generated
    out_from_input = decode_testbench_array(
        payload["outputs"]["y_from_input"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(out_from_input, input_data.astype(np.int32))
    out_from_weight = decode_testbench_array(
        payload["outputs"]["y_from_weight"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(
        out_from_weight,
        np.array(weight_values, dtype=np.int32).reshape(shape),
    )


# -- UINT4 backend test -------------------------------------------------------


def test_uint4_backend() -> None:
    """UINT4 as input, weight (initializer), and output."""
    shape = [2, 3]
    input_data = np.array([[0, 5, 10], [15, 1, 8]], dtype=np.uint8)
    weight_values = [0, 3, 7, 15, 2, 9]
    model = _make_identity_with_weight_model(
        TensorProto.UINT4, shape, weight_values
    )
    payload, generated = _compile_and_run_bitint_testbench(
        model, testbench_inputs={"x": input_data}
    )
    assert "unsigned _BitInt(4)" in generated
    out_from_input = decode_testbench_array(
        payload["outputs"]["y_from_input"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(out_from_input, input_data.astype(np.int32))
    out_from_weight = decode_testbench_array(
        payload["outputs"]["y_from_weight"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(
        out_from_weight,
        np.array(weight_values, dtype=np.int32).reshape(shape),
    )


# -- INT2 backend test --------------------------------------------------------


def test_int2_backend() -> None:
    """INT2 as input, weight (initializer), and output."""
    shape = [2, 3]
    input_data = np.array([[-2, 1, 0], [-1, 0, 1]], dtype=np.int8)
    weight_values = [1, 0, -1, -2, 1, 0]
    model = _make_identity_with_weight_model(
        TensorProto.INT2, shape, weight_values
    )
    payload, generated = _compile_and_run_bitint_testbench(
        model, testbench_inputs={"x": input_data}
    )
    assert "_BitInt(2)" in generated
    out_from_input = decode_testbench_array(
        payload["outputs"]["y_from_input"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(out_from_input, input_data.astype(np.int32))
    out_from_weight = decode_testbench_array(
        payload["outputs"]["y_from_weight"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(
        out_from_weight,
        np.array(weight_values, dtype=np.int32).reshape(shape),
    )


# -- UINT2 backend test -------------------------------------------------------


def test_uint2_backend() -> None:
    """UINT2 as input, weight (initializer), and output."""
    shape = [2, 3]
    input_data = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
    weight_values = [3, 2, 1, 0, 3, 2]
    model = _make_identity_with_weight_model(
        TensorProto.UINT2, shape, weight_values
    )
    payload, generated = _compile_and_run_bitint_testbench(
        model, testbench_inputs={"x": input_data}
    )
    assert "unsigned _BitInt(2)" in generated
    out_from_input = decode_testbench_array(
        payload["outputs"]["y_from_input"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(out_from_input, input_data.astype(np.int32))
    out_from_weight = decode_testbench_array(
        payload["outputs"]["y_from_weight"]["data"], np.dtype(np.int32)
    )
    np.testing.assert_array_equal(
        out_from_weight,
        np.array(weight_values, dtype=np.int32).reshape(shape),
    )
