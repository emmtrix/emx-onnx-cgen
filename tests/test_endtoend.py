from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnx2c import Compiler
from onnx2c.compiler import CompilerOptions
from test_golden import _make_add_model


def _compile_and_run_testbench(model: onnx.ModelProto) -> dict[str, object]:
    options = CompilerOptions(template_dir=Path("templates"), emit_testbench=True)
    compiler = Compiler(options)
    generated = compiler.compile(model)
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [compiler_cmd, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [str(exe_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(result.stdout)


def test_add_c_testbench_matches_onnxruntime() -> None:
    model = _make_add_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "add.onnx"
        onnx.save_model(model, model_path)
        loaded_model = onnx.load_model(model_path)
    payload = _compile_and_run_testbench(loaded_model)
    inputs = {
        name: np.array(value["data"], dtype=np.float32)
        for name, value in payload["inputs"].items()
    }
    sess = ort.InferenceSession(
        loaded_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = sess.run(None, inputs)
    output_data = np.array(payload["output"]["data"], dtype=np.float32)
    np.testing.assert_allclose(output_data, ort_out, rtol=1e-4, atol=1e-5)
