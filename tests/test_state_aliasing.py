from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from emx_onnx_cgen import cli
from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.errors import CodegenError
from emx_onnx_cgen.ir.state import StatePair
from emx_onnx_cgen.state_aliasing import (
    STATE_PAIRS_METADATA_KEY,
    StateAliasingError,
    parse_state_pair_args,
    parse_state_pairs,
    resolve_state_pairs,
)
from emx_onnx_cgen.testbench import StateTestbenchData
from golden_utils import assert_golden

GOLDEN_DIR = Path(__file__).parent / "golden"
BATCH, SEQ, DIM = 2, 8, 3


def _make_model(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes, "state_graph", inputs, outputs, initializer=initializers or []
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 24)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


def _cache_info(name: str, dtype: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, [BATCH, SEQ, DIM])


def _make_kv_cache_model(
    *, seq: int = SEQ, update: int = 1, pos_name: str = "pos"
) -> onnx.ModelProto:
    """One decoder step: write the new key row, then attend over the cache."""
    return _make_model(
        [
            helper.make_node(
                "TensorScatter",
                ["past_k", "k_row", pos_name],
                ["new_k"],
                axis=1,
                name="write_cache",
            ),
            helper.make_node("Transpose", ["new_k"], ["new_k_t"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["q", "new_k_t"], ["scores"], name="attend"),
        ],
        [
            helper.make_tensor_value_info(
                "past_k", TensorProto.FLOAT, [BATCH, seq, DIM]
            ),
            helper.make_tensor_value_info(
                "k_row", TensorProto.FLOAT, [BATCH, update, DIM]
            ),
            helper.make_tensor_value_info(pos_name, TensorProto.INT64, [BATCH]),
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [BATCH, 1, DIM]),
        ],
        [
            helper.make_tensor_value_info(
                "new_k", TensorProto.FLOAT, [BATCH, seq, DIM]
            ),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [BATCH, 1, seq]),
        ],
    )


def _make_masked_update_model() -> onnx.ModelProto:
    """``past * (1 - onehot) + fresh * onehot`` (the current Tiny export)."""
    return _make_model(
        [
            helper.make_node("Sub", ["one", "onehot"], ["keep_mask"]),
            helper.make_node("Mul", ["past", "keep_mask"], ["kept"]),
            helper.make_node("Mul", ["fresh", "onehot"], ["written"]),
            helper.make_node("Add", ["kept", "written"], ["present"]),
        ],
        [
            _cache_info("past"),
            helper.make_tensor_value_info("fresh", TensorProto.FLOAT, [BATCH, 1, DIM]),
            helper.make_tensor_value_info("onehot", TensorProto.FLOAT, [1, SEQ, 1]),
        ],
        [_cache_info("present")],
        [helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])],
    )


def _make_accumulate_model() -> onnx.ModelProto:
    """The node writing the state reads it at the same index (in place)."""
    return _make_model(
        [
            helper.make_node("Add", ["acc", "delta"], ["acc_next"], name="accumulate"),
            helper.make_node("ReduceSum", ["acc_next"], ["total"], keepdims=0),
        ],
        [_cache_info("acc"), _cache_info("delta")],
        [_cache_info("acc_next"), helper.make_tensor_value_info("total", 1, [])],
    )


def _make_read_after_write_model() -> onnx.ModelProto:
    return _make_model(
        [
            helper.make_node(
                "TensorScatter", ["past", "row", "pos"], ["present"], axis=1
            ),
            helper.make_node("Sub", ["present", "past"], ["delta"], name="late_read"),
        ],
        [
            _cache_info("past"),
            helper.make_tensor_value_info("row", TensorProto.FLOAT, [BATCH, 1, DIM]),
            helper.make_tensor_value_info("pos", TensorProto.INT64, [BATCH]),
        ],
        [_cache_info("present"), _cache_info("delta")],
    )


def _make_transpose_state_model() -> onnx.ModelProto:
    square = helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 4, 4])
    square_out = helper.make_tensor_value_info(
        "state_out", TensorProto.FLOAT, [1, 4, 4]
    )
    return _make_model(
        [
            helper.make_node(
                "Transpose", ["state"], ["state_out"], perm=[0, 2, 1], name="flip"
            )
        ],
        [square],
        [square_out],
    )


def _make_cast_state_model() -> onnx.ModelProto:
    return _make_model(
        [
            helper.make_node(
                "Cast", ["past"], ["present"], to=TensorProto.DOUBLE, name="widen"
            )
        ],
        [_cache_info("past")],
        [_cache_info("present", TensorProto.DOUBLE)],
    )


def _compile(model: onnx.ModelProto, *pairs: str, **options: object) -> str:
    compiler = Compiler(
        CompilerOptions(state_pairs=parse_state_pairs(list(pairs)), **options)
    )
    return compiler.compile(model)


def _run_verify(model: onnx.ModelProto, tmp_path: Path, *args: str) -> cli.CliResult:
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    return cli.run_cli_command(
        ["verify", str(model_path), "--temp-dir-root", str(tmp_path), *args]
    )


def test_parse_state_pairs() -> None:
    assert parse_state_pairs(["a=b", " c = d ,e=f"]) == (
        StatePair("a", "b"),
        StatePair("c", "d"),
        StatePair("e", "f"),
    )
    with pytest.raises(ValueError, match="expected INPUT=OUTPUT"):
        parse_state_pairs(["a"])
    with pytest.raises(ValueError, match="more than one pair"):
        parse_state_pairs(["a=b", "a=c"])
    with pytest.raises(ValueError, match="more than one pair"):
        parse_state_pairs(["a=c", "b=c"])


def test_state_pairs_from_metadata_and_overrides() -> None:
    model = _make_kv_cache_model()
    onnx.helper.set_model_props(model, {STATE_PAIRS_METADATA_KEY: "past_k=new_k"})
    assert resolve_state_pairs(None, model) == (StatePair("past_k", "new_k"),)
    assert resolve_state_pairs((StatePair("x", "y"),), model) == (StatePair("x", "y"),)
    assert resolve_state_pairs((), model) == ()
    assert parse_state_pair_args(None) is None
    assert parse_state_pair_args(["a=b"], disabled=True) == ()
    # The compiler itself reads the metadata; () switches it off.
    assert "float model_past_k[2][8][3];" in Compiler().compile(model)
    plain = Compiler(CompilerOptions(state_pairs=())).compile(model)
    assert "model_past_k" not in plain


def test_cli_no_state_pairs_ignores_metadata(tmp_path: Path) -> None:
    model = _make_kv_cache_model()
    onnx.helper.set_model_props(model, {STATE_PAIRS_METADATA_KEY: "past_k=new_k"})
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    compiled = cli.run_cli_command(["compile", str(model_path), str(tmp_path / "a.c")])
    assert compiled.exit_code == 0, compiled.result
    assert "float model_past_k[2][8][3];" in compiled.generated
    plain = cli.run_cli_command(
        ["compile", str(model_path), str(tmp_path / "b.c"), "--no-state-pairs"]
    )
    assert plain.exit_code == 0, plain.result
    assert "model_past_k" not in plain.generated


def test_tensor_scatter_state_drops_cache_copy() -> None:
    model = _make_kv_cache_model()
    baseline = _compile(model)
    generated = _compile(model, "past_k=new_k")
    assert "output[i0][i1][i2] = past_cache[i0][i1][i2];" in baseline
    assert "past_cache" not in generated
    assert "void model(const float k_row[restrict 2][1][3], " in generated
    assert "past_k[restrict" not in generated
    assert "new_k[restrict" not in generated
    assert "node0_write_cache(k_row, pos, model_past_k);" in generated
    assert "void model_state_reset(void) {" in generated


def test_codegen_golden_tensor_scatter_state() -> None:
    generated = _compile(
        _make_kv_cache_model(), "past_k=new_k", model_name="kv_cache_state_model"
    )
    assert_golden(generated, GOLDEN_DIR / "kv_cache_state_model.c")


def test_codegen_golden_tensor_scatter_state_testbench() -> None:
    rng = np.random.default_rng(0)
    steps = 2
    generated = _compile(
        _make_kv_cache_model(),
        "past_k=new_k",
        model_name="kv_cache_state_model",
        emit_testbench=True,
        state_testbench=StateTestbenchData(
            steps=steps,
            initial_state={"past_k": np.zeros((BATCH, SEQ, DIM), dtype=np.float32)},
            step_inputs={
                "k_row": rng.uniform(-1, 1, (steps, BATCH, 1, DIM)).astype(np.float32),
                "pos": np.array([[0, 1], [1, 2]], dtype=np.int64),
                "q": rng.uniform(-1, 1, (steps, BATCH, 1, DIM)).astype(np.float32),
            },
        ),
    )
    assert_golden(generated, GOLDEN_DIR / "kv_cache_state_model_testbench.c")


@pytest.mark.parametrize("runtime", ["onnxruntime", "onnx-reference"])
def test_tensor_scatter_state_matches_reference_over_steps(
    tmp_path: Path, runtime: str
) -> None:
    result = _run_verify(
        _make_kv_cache_model(),
        tmp_path,
        "--state-pair",
        "past_k=new_k",
        "--state-steps",
        "4",
        "--runtime",
        runtime,
    )
    assert result.exit_code == 0, result.result
    assert result.result.startswith("OK"), result.result


@pytest.mark.parametrize(
    ("model_factory", "pair"),
    [
        (_make_masked_update_model, "past=present"),
        (_make_accumulate_model, "acc=acc_next"),
    ],
)
def test_elementwise_state_update_matches_onnxruntime(
    tmp_path: Path, model_factory, pair: str
) -> None:
    result = _run_verify(model_factory(), tmp_path, "--state-pair", pair)
    assert result.exit_code == 0, result.result
    assert result.result.startswith("OK"), result.result


def test_in_place_elementwise_writer_shares_buffer() -> None:
    generated = _compile(_make_accumulate_model(), "acc=acc_next")
    assert "node0_accumulate(model_acc, delta, model_acc);" in generated
    assert "node1_reducesum(model_acc, total);" in generated


def test_state_buffer_persists_and_resets_in_c(tmp_path: Path) -> None:
    compiler_cmd = cli._resolve_compiler(None)
    if compiler_cmd is None:
        pytest.skip("No C compiler available")
    source = _compile(_make_kv_cache_model(), "past_k=new_k")
    harness = """
#include <stdio.h>
int main(void) {
    for (idx_t b = 0; b < 2; ++b)
        for (idx_t s = 0; s < 8; ++s)
            for (idx_t d = 0; d < 3; ++d)
                model_past_k[b][s][d] = 7.0f;
    model_state_reset();
    for (idx_t b = 0; b < 2; ++b)
        for (idx_t s = 0; s < 8; ++s)
            for (idx_t d = 0; d < 3; ++d)
                if (model_past_k[b][s][d] != 0.0f) return 1;
    float k_row[2][1][3] = {{{1, 2, 3}}, {{4, 5, 6}}};
    float q[2][1][3] = {{{1, 0, 0}}, {{0, 1, 0}}};
    float scores[2][1][8];
    for (int64_t step = 0; step < 3; ++step) {
        int64_t pos[2] = {step, 7 - step};
        model(k_row, pos, q, scores);
        k_row[0][0][0] += 1.0f;
        k_row[1][0][1] += 1.0f;
    }
    /* Rows written in earlier calls survive the later calls. */
    if (model_past_k[0][0][0] != 1.0f || model_past_k[0][2][0] != 3.0f) return 2;
    if (model_past_k[1][7][1] != 5.0f || model_past_k[1][5][1] != 7.0f) return 3;
    if (model_past_k[0][3][0] != 0.0f || model_past_k[1][4][1] != 0.0f) return 4;
    if (scores[0][0][0] != 1.0f || scores[1][0][5] != 7.0f) return 5;
    return 0;
}
"""
    c_path = tmp_path / "state.c"
    c_path.write_text(source + harness, encoding="utf-8")
    exe_path = tmp_path / "state"
    subprocess.run(
        [
            *compiler_cmd,
            "-std=c99",
            "-Wall",
            "-Werror",
            str(c_path),
            "-o",
            str(exe_path),
            "-lm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(exe_path)], capture_output=True, text=True)
    assert result.returncode == 0, result


def test_testbench_declarations_include_state_api() -> None:
    compiler = Compiler(CompilerOptions(state_pairs=(StatePair("past_k", "new_k"),)))
    declarations = compiler.compile_testbench_declarations(_make_kv_cache_model())
    assert "extern float model_past_k[2][8][3];" in declarations
    assert "void model_state_reset(void);" in declarations
    assert "void model(const float k_row[restrict 2][1][3], " in declarations


@pytest.mark.parametrize(
    ("model_factory", "pair", "message"),
    [
        (
            _make_read_after_write_model,
            "past=present",
            r"node 1 \(Sub 'late_read'\) reads 'past' after node 0 "
            r"\(TensorScatter\) has written 'present'",
        ),
        (
            _make_transpose_state_model,
            "state=state_out",
            r"node 0 \(Transpose 'flip'\) reads 'state' while writing "
            r"'state_out'.*cannot update that buffer in place",
        ),
        (
            _make_cast_state_model,
            "past=present",
            r"dtype mismatch, input 'past' is float but output 'present' is double",
        ),
        (
            _make_kv_cache_model,
            "past_k=scores",
            r"shape mismatch, input 'past_k' has shape \[2, 8, 3\] but output "
            r"'scores' has shape \[2, 1, 8\]",
        ),
        (_make_kv_cache_model, "k=new_k", r"'k' is not a graph input"),
        (_make_kv_cache_model, "past_k=new", r"'new' is not a graph output"),
    ],
)
def test_illegal_state_pairs_raise(model_factory, pair: str, message: str) -> None:
    with pytest.raises(StateAliasingError, match=message):
        _compile(model_factory(), pair)


def test_cli_reports_illegal_state_pair(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    onnx.save_model(_make_read_after_write_model(), model_path)
    result = cli.run_cli_command(
        [
            "compile",
            str(model_path),
            str(tmp_path / "model.c"),
            "--state-pair",
            "past=present",
        ]
    )
    assert result.exit_code == 1
    assert "reads 'past' after node 0" in result.result
    result = _run_verify(
        _make_read_after_write_model(), tmp_path, "--state-pair", "past=present"
    )
    assert result.exit_code == 1
    assert "reads 'past' after node 0" in result.result


def test_subgraph_references_count_as_reads() -> None:
    from emx_onnx_cgen.state_aliasing import _subgraph_references

    inner = helper.make_graph(
        [helper.make_node("Identity", ["past"], ["inner_out"])],
        "inner",
        [],
        [helper.make_tensor_value_info("inner_out", TensorProto.FLOAT, None)],
    )
    outer = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["branch_out"], then_branch=inner, else_branch=inner
            )
        ],
        "outer",
        [],
        [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, None)],
    )
    assert {"past", "cond"} <= _subgraph_references([outer])
    assert _subgraph_references([1, "x", (outer,)]) >= {"past"}


def _build_and_run(source: str, tmp_path: Path) -> subprocess.CompletedProcess:
    compiler_cmd = cli._resolve_compiler(None)
    if compiler_cmd is None:
        pytest.skip("No C compiler available")
    c_path = tmp_path / "testbench.c"
    c_path.write_text(source, encoding="utf-8")
    exe_path = tmp_path / "testbench"
    subprocess.run(
        [
            *compiler_cmd,
            "-std=c99",
            "-Wall",
            "-Werror",
            str(c_path),
            "-o",
            str(exe_path),
            "-lm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=tmp_path)


def test_state_testbench_avoids_model_value_names(tmp_path: Path) -> None:
    # A step model with an input named like the testbench's own identifiers.
    model = _make_kv_cache_model(pos_name="step")
    source = _compile(
        model,
        "past_k=new_k",
        emit_testbench=True,
        state_testbench=StateTestbenchData(steps=3),
    )
    assert "for (idx_t step_1 = 0; step_1 < 3; ++step_1)" in source
    result = _build_and_run(source, tmp_path)
    assert result.returncode == 0, result


def test_state_testbench_random_positions_stay_in_bounds(tmp_path: Path) -> None:
    source = _compile(
        _make_kv_cache_model(seq=4, update=2),
        "past_k=new_k",
        emit_testbench=True,
        state_testbench=StateTestbenchData(steps=8),
    )
    # Linear TensorScatter: write_index + 2 rows must fit into 4 rows.
    assert "(int64_t)(rng_next_u64() % 3ull)" in source
    result = _build_and_run(source, tmp_path)
    assert result.returncode == 0, result


def test_short_cache_matches_onnxruntime(tmp_path: Path) -> None:
    result = _run_verify(
        _make_kv_cache_model(seq=4, update=2),
        tmp_path,
        "--state-pair",
        "past_k=new_k",
        "--state-steps",
        "6",
    )
    assert result.exit_code == 0, result.result


def test_tensor_scatter_write_index_bounds() -> None:
    model = _make_kv_cache_model(seq=4, update=2)
    assert Compiler().index_input_bounds(model) == {"pos": 3}
    model.graph.node[0].attribute.append(helper.make_attribute("mode", "circular"))
    assert Compiler().index_input_bounds(model) == {"pos": 4}


def test_state_api_names_are_reserved() -> None:
    model = _make_model(
        [
            helper.make_node("Add", ["model_acc", "model_state_reset"], ["delta"]),
            helper.make_node("Add", ["acc", "delta"], ["acc_next"]),
        ],
        [
            _cache_info("acc"),
            _cache_info("model_acc"),
            _cache_info("model_state_reset"),
        ],
        [_cache_info("acc_next")],
    )
    generated = _compile(model, "acc=acc_next")
    assert "float model_acc[2][8][3];" in generated
    assert "const float model_acc_1[restrict 2][8][3]" in generated
    assert "const float model_state_reset_1[restrict 2][8][3]" in generated
    assert "void model_state_reset(void) {" in generated


def test_update_reading_the_state_is_not_in_place() -> None:
    model = _make_model(
        [
            helper.make_node(
                "TensorScatter",
                ["past", "past", "pos"],
                ["present"],
                axis=1,
                mode="circular",
                name="self_scatter",
            )
        ],
        [
            _cache_info("past"),
            helper.make_tensor_value_info("pos", TensorProto.INT64, [BATCH]),
        ],
        [_cache_info("present")],
    )
    with pytest.raises(
        StateAliasingError,
        match=r"node 0 \(TensorScatter 'self_scatter'\) reads 'past' while writing",
    ):
        _compile(model, "past=present")


def test_state_testbench_rejects_testbench_inputs() -> None:
    with pytest.raises(Exception, match="state_testbench"):
        _compile(
            _make_kv_cache_model(),
            "past_k=new_k",
            emit_testbench=True,
            testbench_inputs={"q": np.zeros((BATCH, 1, DIM), dtype=np.float32)},
        )


def _make_cast_position_model(seq: int = 2) -> onnx.ModelProto:
    """``pos`` reaches TensorScatter through Cast and Reshape."""
    return _make_model(
        [
            helper.make_node("Cast", ["pos32"], ["pos64"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["pos64", "flat"], ["pos"]),
            helper.make_node(
                "TensorScatter", ["past", "row", "pos"], ["present"], axis=1
            ),
        ],
        [
            helper.make_tensor_value_info("past", TensorProto.FLOAT, [BATCH, seq, DIM]),
            helper.make_tensor_value_info("row", TensorProto.FLOAT, [BATCH, 1, DIM]),
            helper.make_tensor_value_info("pos32", TensorProto.INT32, [BATCH, 1]),
        ],
        [
            helper.make_tensor_value_info(
                "present", TensorProto.FLOAT, [BATCH, seq, DIM]
            )
        ],
        [helper.make_tensor("flat", TensorProto.INT64, [1], [BATCH])],
    )


def test_index_bounds_follow_cast_and_reshape(tmp_path: Path) -> None:
    model = _make_cast_position_model()
    assert Compiler().index_input_bounds(model) == {"pos32": 2}
    result = _run_verify(
        model, tmp_path, "--state-pair", "past=present", "--state-steps", "6"
    )
    assert result.exit_code == 0, result.result


@pytest.mark.parametrize(
    ("state_testbench", "message"),
    [
        (
            StateTestbenchData(
                steps=2, step_inputs={"q": np.zeros((2, 3, 1, DIM), np.float32)}
            ),
            r"Step input 'q' needs shape \[2, 2, 1, 3\]",
        ),
        (
            StateTestbenchData(
                steps=2, initial_state={"past_k": np.zeros((SEQ, DIM), np.float32)}
            ),
            r"Initial state 'past_k' needs shape \[2, 8, 3\]",
        ),
        (
            StateTestbenchData(
                steps=2, initial_state={"q": np.zeros((1,), np.float32)}
            ),
            r"Initial state 'q' is not the input of a state pair",
        ),
    ],
)
def test_state_testbench_data_is_validated(state_testbench, message: str) -> None:
    with pytest.raises(CodegenError, match=message):
        _compile(
            _make_kv_cache_model(),
            "past_k=new_k",
            emit_testbench=True,
            state_testbench=state_testbench,
        )


def test_pass_through_state_emits_no_copy() -> None:
    model = _make_model(
        [
            helper.make_node("Identity", ["past"], ["present"], name="keep"),
            helper.make_node("ReduceSum", ["present"], ["total"], keepdims=0),
        ],
        [_cache_info("past")],
        [_cache_info("present"), helper.make_tensor_value_info("total", 1, [])],
    )
    generated = _compile(model, "past=present")
    assert "OpType: Identity" not in generated
    assert "node0_reducesum(model_past, total);" in generated


def test_state_testbench_helpers_avoid_loop_var_names(tmp_path: Path) -> None:
    model = _make_kv_cache_model(pos_name="i0")
    source = _compile(
        model,
        "past_k=new_k",
        emit_testbench=True,
        state_testbench=StateTestbenchData(
            steps=2,
            step_inputs={"i0": np.array([[0, 1], [2, 3]], dtype=np.int64)},
        ),
    )
    result = _build_and_run(source, tmp_path)
    assert result.returncode == 0, result


def test_state_names_follow_model_name_remap() -> None:
    model = _make_model(
        [helper.make_node("Add", ["acc", "my_model"], ["acc_next"])],
        [_cache_info("acc"), _cache_info("my_model")],
        [_cache_info("acc_next")],
    )
    generated = _compile(model, "acc=acc_next", model_name="my-model")
    assert "void my_model(const float my_model_1[restrict 2][8][3]" in generated
    assert "float my_model_acc[2][8][3];" in generated
    assert "void my_model_state_reset(void) {" in generated


def test_state_input_named_like_the_model_is_rejected() -> None:
    model = _make_model(
        [helper.make_node("Add", ["acc", "acc"], ["acc_next"])],
        [_cache_info("acc")],
        [_cache_info("acc_next")],
    )
    with pytest.raises(CodegenError, match="--model-name"):
        _compile(model, "acc=acc_next", model_name="acc")
