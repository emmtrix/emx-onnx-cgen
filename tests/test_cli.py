from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import onnx
import pytest
import numpy as np

from onnx import TensorProto, helper, numpy_helper

from test_ops import (
    _make_operator_model,
    _make_reduce_model,
    _reduce_output_shape,
)
from emx_onnx_cgen import cli
from emx_onnx_cgen.ir.model import Graph, Node, TensorType, Value
from shared.scalar_types import ScalarType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _run_cli_verify(model: onnx.ModelProto, *, runtime: str = "ort") -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx.save_model(model, model_path)
        env = os.environ.copy()
        python_path = str(SRC_ROOT)
        if env.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
        verify_cmd = [
            sys.executable,
            "-m",
            "emx_onnx_cgen",
            "verify",
            str(model_path),
            "--temp-dir-root",
            str(temp_dir),
        ]
        if runtime != "ort":
            verify_cmd.extend(["--runtime", runtime])
        subprocess.run(
            verify_cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )


def _make_constant_only_model() -> onnx.ModelProto:
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    constant_value = helper.make_tensor(
        name="value",
        data_type=TensorProto.FLOAT,
        dims=[2],
        vals=[1.5, -2.0],
    )
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["out"],
        value=constant_value,
    )
    graph = helper.make_graph(
        [constant_node],
        "constant_only_graph",
        [],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 25)],
    )
    model.ir_version = 13
    onnx.checker.check_model(model)
    return model


def _make_dynamic_identity_chain_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N"])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N"])
    mid_info = helper.make_tensor_value_info("mid", TensorProto.FLOAT, ["N"])
    nodes = [
        helper.make_node("Identity", inputs=["x"], outputs=["mid"]),
        helper.make_node("Identity", inputs=["mid"], outputs=["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dynamic_identity_chain",
        [input_info],
        [output_info],
        value_info=[mid_info],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _write_input_pb(data_dir: Path, name: str, values: np.ndarray) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tensor = numpy_helper.from_array(values, name=name)
    (data_dir / "input_0.pb").write_bytes(tensor.SerializeToString())


def test_cli_verify_operator_model() -> None:
    model = _make_operator_model(
        op_type="Add",
        input_shapes=[[2, 3], [2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        attrs={},
    )
    _run_cli_verify(model)


def test_cli_verify_constant_only_model() -> None:
    model = _make_constant_only_model()
    _run_cli_verify(model, runtime="onnx-reference")


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


def test_cli_parse_shape_inference_shapes_flag() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "compile",
            "model.onnx",
            "--shape-inference-shapes",
            "x=2x3;size=[2,3,5,6];seq=seq[4]:8x16",
        ]
    )
    assert args.shape_inference_shapes == "x=2x3;size=[2,3,5,6];seq=seq[4]:8x16"


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


def test_cli_resolve_sanitize_enabled_uses_cli_flag_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cli._ENABLE_SANITIZE_ENV, raising=False)
    enabled, override = cli._resolve_sanitize_enabled(cli_requested=False)
    assert enabled is False
    assert override is None

    enabled, override = cli._resolve_sanitize_enabled(cli_requested=True)
    assert enabled is True
    assert override is None


@pytest.mark.parametrize(
    ("env_value", "cli_requested", "expected_enabled"),
    [
        ("1", False, True),
        ("0", True, False),
    ],
)
def test_cli_resolve_sanitize_enabled_env_overrides_cli_flag(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    cli_requested: bool,
    expected_enabled: bool,
) -> None:
    monkeypatch.setenv(cli._ENABLE_SANITIZE_ENV, env_value)
    enabled, override = cli._resolve_sanitize_enabled(cli_requested=cli_requested)
    assert enabled is expected_enabled
    assert override == f"{cli._ENABLE_SANITIZE_ENV}={env_value!r}"


def test_cli_verify_per_node_accuracy_flag_defaults_to_false() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx"])
    assert args.per_node_accuracy is False


def test_cli_verify_per_node_accuracy_flag_can_be_enabled() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["verify", "model.onnx", "--per-node-accuracy"])
    assert args.per_node_accuracy is True


def test_cli_compile_accepts_io_bound_dynamic_intermediates(
    tmp_path: Path,
) -> None:
    model = _make_dynamic_identity_chain_model()
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.c"
    onnx.save_model(model, model_path)

    result = cli.run_cli_command(["compile", str(model_path), str(output_path)])

    assert result.exit_code == 0
    assert result.result == ""
    assert result.generated is not None
    generated = result.generated
    assert "int N" in generated


def test_cli_compile_accepts_explicit_shape_inference_shapes(tmp_path: Path) -> None:
    model = _make_dynamic_identity_chain_model()
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.c"
    onnx.save_model(model, model_path)

    result = cli.run_cli_command(
        [
            "compile",
            str(model_path),
            str(output_path),
            "--shape-inference-shapes",
            "x=2",
        ]
    )

    assert result.exit_code == 0
    assert result.result == ""


def test_cli_compile_accepts_dynamic_maxpool_reference_model(tmp_path: Path) -> None:
    output_path = tmp_path / "model.c"
    result = cli.run_cli_command(
        [
            "compile",
            "--model-base-dir",
            str(PROJECT_ROOT / "onnx2c-org" / "test" / "simple_networks"),
            "maxpool_k2.onnx",
            str(output_path),
        ]
    )

    assert result.exit_code == 0
    assert result.generated is not None
    generated = result.generated
    assert "int N" in generated


def test_cli_verify_notes_shape_inference_shapes_hint_from_test_data_dir(
    tmp_path: Path,
) -> None:
    model = _make_dynamic_identity_chain_model()
    model_path = tmp_path / "model.onnx"
    test_data_dir = tmp_path / "test_data_set_0"
    onnx.save_model(model, model_path)
    _write_input_pb(test_data_dir, "x", np.array([1.0, 2.0], dtype=np.float32))

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "verify",
            str(model_path),
            "--test-data-dir",
            str(test_data_dir),
        ]
    )
    stream = StringIO()
    reporter = cli._VerifyReporter(stream=stream, color_mode="never")
    error = (
        "Code generation needs explicit shape concretization, but no "
        "--shape-inference-shapes were provided. Reason: tensor 'mid' has "
        "dynamic dimensions ('N',)."
    )

    assert "--shape-inference-shapes" in error
    cli._maybe_note_shape_inference_shapes_hint(reporter, args=args, error=error)
    reporter.flush_deferred()
    assert '--shape-inference-shapes "x=[1.0,2.0]"' in stream.getvalue()


def test_augment_model_with_tensor_node_outputs_skips_non_top_level_outputs() -> None:
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    identity = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    model = helper.make_model(
        helper.make_graph(
            [identity],
            "top_level_graph",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [output],
        ),
        opset_imports=[helper.make_operatorsetid("", 20)],
    )

    tensor_type = TensorType(dtype=ScalarType.F32, shape=(1,), dim_params=(None,))
    graph = Graph(
        inputs=(Value(name="x", type=tensor_type),),
        outputs=(Value(name="y", type=tensor_type),),
        nodes=(
            Node(
                op_type="FakeInner",
                name="inner",
                inputs=("x",),
                outputs=("inner_only",),
                attrs={},
            ),
            Node(
                op_type="Identity",
                name="top",
                inputs=("x",),
                outputs=("y",),
                attrs={},
            ),
        ),
        initializers=(),
        values=(
            Value(name="inner_only", type=tensor_type),
            Value(name="y", type=tensor_type),
        ),
    )

    augmented = cli._augment_model_with_tensor_node_outputs(model, graph)

    output_names = [value.name for value in augmented.graph.output]
    assert "y" in output_names
    assert "inner_only" not in output_names


def test_cli_compile_accepts_testbench_output_format_txt_emmtrix() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "compile",
            "model.onnx",
            "out.c",
            "--testbench-output-format",
            "txt-emmtrix",
        ]
    )
    assert args.testbench_output_format == "txt-emmtrix"


def test_cli_compile_accepts_testbench_output_format_txt_emmtrix_with_ulp() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "compile",
            "model.onnx",
            "out.c",
            "--testbench-output-format",
            "txt-emmtrix:123.5",
        ]
    )
    assert args.testbench_output_format == "txt-emmtrix:123.5"


def test_cli_compile_rejects_testbench_output_format_txt_emmtrix_invalid_ulp() -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compile",
                "model.onnx",
                "out.c",
                "--testbench-output-format",
                "txt-emmtrix:abc",
            ]
        )
