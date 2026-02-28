from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
import pytest

from onnx import TensorProto

from test_ops import (
    _make_operator_model,
    _make_reduce_model,
    _reduce_output_shape,
)
from emx_onnx_cgen import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _run_cli_verify(model: onnx.ModelProto) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx.save_model(model, model_path)
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
                "--temp-dir-root",
                str(temp_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )


def test_cli_verify_operator_model() -> None:
    model = _make_operator_model(
        op_type="Add",
        input_shapes=[[2, 3], [2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        attrs={},
    )
    _run_cli_verify(model)


def test_cli_verify_reduce_model() -> None:
    output_shape = _reduce_output_shape([2, 3, 4], [1], 1)
    model = _make_reduce_model(
        op_type="ReduceSum",
        input_shape=[2, 3, 4],
        output_shape=output_shape,
        axes=[1],
        keepdims=1,
        dtype=TensorProto.FLOAT,
    )
    _run_cli_verify(model)


def test_cli_testbench_filename_and_include() -> None:
    output_path = Path("out.c")
    testbench_path = output_path.with_name(
        f"{output_path.stem}_testbench{output_path.suffix}"
    )
    assert testbench_path.name == "out_testbench.c"

    from emx_onnx_cgen.cli import (
        _resolve_testbench_output_path,
        _wrap_separate_testbench_source,
    )

    rendered = _wrap_separate_testbench_source(
        "void model(int x);",
        "int main(void) { return 0; }\n",
    )
    assert '#include "out.c"' not in rendered
    assert "void model(int x);" in rendered

    resolved = _resolve_testbench_output_path(output_path, "tb")
    assert resolved.name == "tb.c"
    resolved = _resolve_testbench_output_path(output_path, "tb_custom.c")
    assert resolved.name == "tb_custom.c"


def test_cli_model_base_dir_resolves_relative_paths(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    parser = cli._build_parser()
    args = parser.parse_args(
        ["compile", "model.onnx", "out.c", "--model-base-dir", str(base_dir)]
    )
    cli._apply_base_dir(args, parser)
    assert args.model == base_dir / "model.onnx"
    assert args.output == Path("out.c")


def test_cli_model_base_dir_absolute_passthrough(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "out.c"
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "compile",
            str(model_path),
            str(output_path),
            "--model-base-dir",
            str(base_dir),
        ]
    )
    cli._apply_base_dir(args, parser)
    assert args.model == model_path
    assert args.output == output_path


def test_cli_model_base_dir_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        ["compile", "model.onnx", "--model-base-dir", str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit):
        cli._apply_base_dir(args, parser)
    err = capsys.readouterr().err
    assert "--model-base-dir" in err
    assert "does not exist or is not a directory" in err


def test_cli_model_base_dir_resolves_test_data(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "verify",
            "model.onnx",
            "--test-data-dir",
            "inputs",
            "--model-base-dir",
            str(base_dir),
        ]
    )
    cli._apply_base_dir(args, parser)
    assert args.model == base_dir / "model.onnx"
    assert args.test_data_dir == base_dir / "inputs"


def test_cli_verify_rejects_model_name_flag() -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify", "model.onnx", "--model-name", "x"])


def test_cli_verify_sanitize_flag_defaults_to_false() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx"])
    assert args.sanitize is False


def test_cli_verify_sanitize_flag_can_be_enabled() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx", "--sanitize"])
    assert args.sanitize is True


def test_cli_verify_per_node_accuracy_flag_defaults_to_false() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx"])
    assert args.per_node_accuracy is False


def test_cli_verify_per_node_accuracy_flag_can_be_enabled() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx", "--per-node-accuracy"])
    assert args.per_node_accuracy is True


def test_cli_verify_runtime_defaults_to_auto() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx"])
    assert args.runtime == "auto"


def test_select_runtime_backend_auto_uses_reference_without_custom_domains() -> None:
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "g", [], []),
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
    )
    runtime_name, runtime_note = cli._select_runtime_backend("auto", model)
    assert runtime_name == "onnx-reference"
    assert runtime_note == "Runtime: auto selected onnx-reference"


def test_select_runtime_backend_auto_uses_onnxruntime_for_custom_domains() -> None:
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "g", [], []),
        opset_imports=[
            onnx.helper.make_operatorsetid("", 13),
            onnx.helper.make_operatorsetid("com.example", 1),
        ],
    )
    runtime_name, runtime_note = cli._select_runtime_backend("auto", model)
    assert runtime_name == "onnxruntime"
    assert runtime_note == "Runtime: auto selected onnxruntime for custom domains com.example"


def test_select_runtime_backend_explicit_backend_is_fixed() -> None:
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "g", [], []),
        opset_imports=[
            onnx.helper.make_operatorsetid("", 13),
            onnx.helper.make_operatorsetid("com.example", 1),
        ],
    )
    runtime_name, runtime_note = cli._select_runtime_backend("onnx-reference", model)
    assert runtime_name == "onnx-reference"
    assert runtime_note is None
