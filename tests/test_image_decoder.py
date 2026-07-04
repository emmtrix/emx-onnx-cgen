from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from emx_onnx_cgen import cli
from emx_onnx_cgen.codegen.image_decoder_libs import (
    parse_image_decoder_libs,
    resolve_image_decoder_plan,
)
from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.errors import UnsupportedOpError


def _make_image_decoder_model(
    encoded_size: int,
    output_shape: tuple[int, int, int],
    pixel_format: str = "RGB",
) -> onnx.ModelProto:
    node = helper.make_node(
        "ImageDecoder",
        inputs=["data"],
        outputs=["output"],
        pixel_format=pixel_format,
    )
    graph = helper.make_graph(
        [node],
        "image_decoder_test",
        [helper.make_tensor_value_info("data", TensorProto.UINT8, [encoded_size])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.UINT8, list(output_shape)
            )
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )


def _encode_test_image(image_format: str) -> tuple[bytes, np.ndarray]:
    PIL_Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(1234)
    pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    PIL_Image.fromarray(pixels).save(buffer, format=image_format)
    encoded = buffer.getvalue()
    decoded = np.array(PIL_Image.open(io.BytesIO(encoded)))
    return encoded, decoded


def test_parse_image_decoder_libs_priority_and_validation() -> None:
    assert parse_image_decoder_libs("stb") == ("stb",)
    assert parse_image_decoder_libs("libjpeg-turbo, stb") == (
        "libjpeg-turbo",
        "stb",
    )
    with pytest.raises(ValueError):
        parse_image_decoder_libs("")
    with pytest.raises(ValueError):
        parse_image_decoder_libs("nosuchlib")


def test_resolve_plan_first_library_wins_per_format() -> None:
    plan = resolve_image_decoder_plan(("libjpeg-turbo", "stb"))
    assignments = dict(plan.format_to_library)
    assert assignments["jpeg"] == "libjpeg-turbo"
    assert assignments["png"] == "stb"
    assert "tiff" not in assignments
    assert plan.link_libraries == ("jpeg",)

    stb_first = resolve_image_decoder_plan(("stb", "libjpeg-turbo"))
    assert dict(stb_first.format_to_library)["jpeg"] == "stb"
    assert stb_first.link_libraries == ()


def test_lowering_rejects_unknown_pixel_format() -> None:
    model = _make_image_decoder_model(64, (8, 8, 3), pixel_format="YUV")
    with pytest.raises(UnsupportedOpError, match="pixel_format"):
        Compiler(CompilerOptions()).compile(model)


def test_lowering_rejects_channel_mismatch() -> None:
    model = _make_image_decoder_model(64, (8, 8, 3), pixel_format="Grayscale")
    with pytest.raises(UnsupportedOpError, match="channel count"):
        Compiler(CompilerOptions()).compile(model)


def test_generated_code_detects_format_at_runtime() -> None:
    model = _make_image_decoder_model(64, (8, 8, 3))
    generated = Compiler(CompilerOptions()).compile(model)
    # Runtime magic-byte dispatch with the stb decoder compiled in.
    assert "emx_image_decode_rgb" in generated
    assert "emx_image_decode_stb" in generated
    assert "0xFFu && data[1] == 0xD8u" in generated  # JPEG magic
    assert "STBI_ONLY_PNG" in generated
    # No decoder configured for webp/tiff/jpeg2000 with the stb default.
    assert "emx_image_decode_libwebp" not in generated


def test_generated_code_prefers_configured_library() -> None:
    model = _make_image_decoder_model(64, (8, 8, 3))
    generated = Compiler(
        CompilerOptions(image_decoder_libs=("libjpeg-turbo", "stb"))
    ).compile(model)
    assert "emx_image_decode_libjpeg_turbo" in generated
    # JPEG is assigned to libjpeg-turbo, so stb must not compile its decoder.
    assert "STBI_ONLY_JPEG" not in generated
    assert "STBI_ONLY_PNG" in generated


def _write_test_data_dir(directory: Path, encoded: bytes, expected: np.ndarray) -> None:
    directory.mkdir(parents=True)
    input_tensor = numpy_helper.from_array(
        np.frombuffer(encoded, dtype=np.uint8), name="data"
    )
    (directory / "input_0.pb").write_bytes(input_tensor.SerializeToString())
    output_tensor = numpy_helper.from_array(expected, name="output")
    (directory / "output_0.pb").write_bytes(output_tensor.SerializeToString())


@pytest.mark.parametrize(
    ("image_format", "libs"),
    [
        ("PNG", "stb"),
        ("BMP", "stb"),
        ("JPEG", "libjpeg-turbo"),
    ],
)
def test_verify_decodes_image_end_to_end(
    tmp_path: Path, image_format: str, libs: str
) -> None:
    encoded, expected = _encode_test_image(image_format)
    model = _make_image_decoder_model(len(encoded), expected.shape)
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    _write_test_data_dir(tmp_path / "test_data_set_0", encoded, expected)

    result = cli.run_cli_command(
        [
            "emx-onnx-cgen",
            "verify",
            "--model-base-dir",
            str(tmp_path),
            "model.onnx",
            "--test-data-dir",
            "test_data_set_0",
            "--image-decoder-libs",
            libs,
        ]
    )
    assert result.exit_code == 0, result.result
    assert result.result == "OK (max abs diff 0)"


def test_verify_zero_fills_when_format_has_no_decoder(tmp_path: Path) -> None:
    # A JPEG input with only stb's PNG decoder configured cannot be decoded;
    # the generated code must zero-fill deterministically instead of failing.
    encoded, expected = _encode_test_image("JPEG")
    model = _make_image_decoder_model(len(encoded), expected.shape)
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    _write_test_data_dir(tmp_path / "test_data_set_0", encoded, np.zeros_like(expected))

    result = cli.run_cli_command(
        [
            "emx-onnx-cgen",
            "verify",
            "--model-base-dir",
            str(tmp_path),
            "model.onnx",
            "--test-data-dir",
            "test_data_set_0",
            "--image-decoder-libs",
            "libwebp",
        ]
    )
    assert result.exit_code == 0, result.result
    assert result.result == "OK (max abs diff 0)"
