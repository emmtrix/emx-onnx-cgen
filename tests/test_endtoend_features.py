from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.testbench import decode_testbench_array

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _make_add_initializer_model() -> tuple[onnx.ModelProto, np.ndarray]:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shape)
    weight_values = np.linspace(0.1, 0.6, num=6, dtype=np.float32).reshape(input_shape)
    weight_initializer = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        dims=input_shape,
        vals=weight_values.flatten().tolist(),
    )
    weight_info = helper.make_tensor_value_info(
        "weight", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, input_shape)
    node = helper.make_node("Add", inputs=["in0", "weight"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "add_init_graph",
        [input_info, weight_info],
        [output],
        initializer=[weight_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model, weight_values


def _compile_and_run_testbench(
    model: onnx.ModelProto,
    *,
    compiler_options: CompilerOptions | None = None,
    testbench_inputs: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], str]:
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        if compiler_options is None:
            compiler_options = CompilerOptions(emit_testbench=True)
        compiler = Compiler(compiler_options)
        generated = compiler.compile(model)
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [compiler_cmd, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
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


def _run_cli_verify(model_path: Path) -> None:
    env = os.environ.copy()
    python_path = str(SRC_ROOT)
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    subprocess.run(
        [
            sys.executable,
            "-m",
            "emx_onnx_cgen",
            "verify",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


def test_initializer_weights_emitted_as_static_arrays() -> None:
    model, weights = _make_add_initializer_model()
    payload, generated = _compile_and_run_testbench(model)
    assert "extern const float weight" in generated
    assert "const EMX_UNUSED float weight" in generated
    output_data = decode_testbench_array(
        payload["outputs"]["out"]["data"], np.dtype(np.float32)
    )
    assert output_data.shape == weights.shape
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "add_init.onnx"
        onnx.save_model(model, model_path)
        _run_cli_verify(model_path)


def test_testbench_accepts_constant_inputs() -> None:
    model, weights = _make_add_initializer_model()
    input_values = np.linspace(1.0, 6.0, num=6, dtype=np.float32).reshape(weights.shape)
    options = CompilerOptions(emit_testbench=True)
    payload, generated = _compile_and_run_testbench(
        model, compiler_options=options, testbench_inputs={"in0": input_values}
    )
    assert "static const float in0_testbench_data" not in generated
    output_data = decode_testbench_array(
        payload["outputs"]["out"]["data"], np.dtype(np.float32)
    )
    np.testing.assert_allclose(output_data, input_values + weights)


def _encode_ppm(pixels: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as a binary PPM (P6) image."""
    height, width, _ = pixels.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + pixels.tobytes()


def _encode_bmp(pixels: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as a 24-bit (BGR, bottom-up) BMP image."""
    import struct

    height, width, _ = pixels.shape
    row_stride = (width * 3 + 3) & ~3
    body = bytearray()
    for y in range(height - 1, -1, -1):
        row = bytearray()
        for x in range(width):
            r, g, b = pixels[y, x]
            row += bytes((int(b), int(g), int(r)))
        row += b"\x00" * (row_stride - len(row))
        body += row
    file_header = b"BM" + struct.pack("<IHHI", 54 + len(body), 0, 0, 54)
    info_header = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(body), 2835, 2835, 0, 0
    )
    return bytes(file_header + info_header + body)


def _write_image_decoder_case(
    directory: Path,
    *,
    pixel_format: str,
    expected: np.ndarray,
    encoded: bytes,
) -> None:
    from onnx import numpy_helper

    height, width, channels = expected.shape
    encoded_input = helper.make_tensor_value_info(
        "encoded", TensorProto.UINT8, [len(encoded)]
    )
    image_output = helper.make_tensor_value_info(
        "image", TensorProto.UINT8, [height, width, channels]
    )
    node = helper.make_node(
        "ImageDecoder", ["encoded"], ["image"], pixel_format=pixel_format
    )
    model = helper.make_model(
        helper.make_graph([node], "image_decoder", [encoded_input], [image_output]),
        opset_imports=[helper.make_operatorsetid("", 20)],
    )
    model.ir_version = 9
    onnx.save_model(model, str(directory / "model.onnx"))
    data_dir = directory / "test_data_set_0"
    data_dir.mkdir()
    (data_dir / "input_0.pb").write_bytes(
        numpy_helper.from_array(
            np.frombuffer(encoded, dtype=np.uint8).copy(), name="encoded"
        ).SerializeToString()
    )
    (data_dir / "output_0.pb").write_bytes(
        numpy_helper.from_array(expected, name="image").SerializeToString()
    )


def _verify_image_decoder_case(directory: Path) -> "object":
    from emx_onnx_cgen import cli

    return cli.run_cli_command(
        [
            "emx-onnx-cgen",
            "verify",
            "--model-base-dir",
            str(directory),
            "model.onnx",
            "--test-data-dir",
            "test_data_set_0",
        ]
    )


@pytest.mark.parametrize("encoder", [_encode_ppm, _encode_bmp])
@pytest.mark.parametrize("pixel_format", ["RGB", "BGR"])
def test_image_decoder_lossless_roundtrip(encoder, pixel_format: str) -> None:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("C compiler not available")
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    expected = pixels if pixel_format == "RGB" else pixels[:, :, ::-1].copy()
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        _write_image_decoder_case(
            directory,
            pixel_format=pixel_format,
            expected=expected,
            encoded=encoder(pixels),
        )
        result = _verify_image_decoder_case(directory)
    assert result.exit_code == 0, result.result
    assert result.result == "OK (max abs diff 0)"


def test_image_decoder_grayscale() -> None:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("C compiler not available")
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(3, 4, 1), dtype=np.uint8)
    encoded = _encode_ppm(np.repeat(gray, 3, axis=2))
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        _write_image_decoder_case(
            directory, pixel_format="Grayscale", expected=gray, encoded=encoded
        )
        result = _verify_image_decoder_case(directory)
    assert result.exit_code == 0, result.result
    assert result.result == "OK (max abs diff 0)"


def test_image_decoder_grayscale_uses_itu_r_luma_weights() -> None:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("C compiler not available")
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    # Same ITU-R 601-2 luma transform the generated C applies, matching the
    # reference (Pillow) decoder: L = (R*19595 + G*38470 + B*7471) >> 16.
    weights = np.array([19595, 38470, 7471], dtype=np.uint32)
    luma = (pixels.astype(np.uint32) * weights).sum(axis=2) >> 16
    expected = luma.astype(np.uint8)[:, :, np.newaxis]
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        _write_image_decoder_case(
            directory,
            pixel_format="Grayscale",
            expected=expected,
            encoded=_encode_ppm(pixels),
        )
        result = _verify_image_decoder_case(directory)
    assert result.exit_code == 0, result.result
    assert result.result == "OK (max abs diff 0)"


def test_image_decoder_rejects_undecodable_stream() -> None:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("C compiler not available")
    rng = np.random.default_rng(2)
    expected = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    junk = bytes(rng.integers(0, 256, size=128, dtype=np.uint8))
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        _write_image_decoder_case(
            directory, pixel_format="RGB", expected=expected, encoded=junk
        )
        result = _verify_image_decoder_case(directory)
    # stb_image cannot decode arbitrary bytes; the generated C reports the
    # failure at runtime instead of silently producing garbage.
    assert result.exit_code != 0
