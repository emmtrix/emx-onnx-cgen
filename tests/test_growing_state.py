"""Growing state pairs: a KV cache appended with Concat stays in one buffer.

The model is the smallest form of the Qwen2.5 decoder's cache update:
``present = Concat(past, new, axis=1)`` with ``past[2, p, 3]`` and
``present[2, p + 1, 3]``, followed by attention over ``present``. With
``--state-pair past=present`` and ``--input-dim p<=7`` the cache lives in a
state struct of capacity 8; a C driver runs several steps through that buffer
and every step is compared with ONNX Runtime fed the previous ``present``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.ir.state import StatePair
from emx_onnx_cgen.state_aliasing import StateAliasingError

_CAPACITY_BOUND = 7
_STEPS = 5


def _float(name: str, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _kv_cache_model(*, writer: str = "Concat") -> onnx.ModelProto:
    if writer == "Concat":
        update = helper.make_node("Concat", ["past", "new"], ["present"], axis=1)
    else:
        # Concat(new, past) prepends: the old rows move, so no shared buffer.
        update = helper.make_node("Concat", ["new", "past"], ["present"], axis=1)
    nodes = [
        helper.make_node("Relu", ["x"], ["new"]),
        update,
        helper.make_node("Transpose", ["present"], ["keys_t"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "keys_t"], ["scores"]),
        # repeat_kv as in Qwen: Unsqueeze + Expand reads the cache through a
        # reshape, whose generic kernel uses a flat (stride-less) view.
        helper.make_node("Unsqueeze", ["present", "axis1"], ["present_1"]),
        helper.make_node("Expand", ["present_1", "heads"], ["repeated"]),
        # MatMul and Expand reading the cache buffer directly.
        helper.make_node("MatMul", ["probs", "present"], ["context"]),
        helper.make_node("Expand", ["present", "heads"], ["broadcast"]),
    ]
    graph = helper.make_graph(
        nodes,
        "growing_kv_cache",
        [
            _float("past", [2, "p", 3]),
            _float("x", [2, 1, 3]),
            _float("q", [2, 1, 3]),
            _float("probs", [2, 1, "p + 1"]),
        ],
        [
            _float("present", [2, "p + 1", 3]),
            _float("scores", [2, 1, "p + 1"]),
            _float("repeated", [2, 2, "p + 1", 3]),
            _float("context", [2, 1, 3]),
            _float("broadcast", [2, 2, "p + 1", 3]),
        ],
        [
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
            numpy_helper.from_array(np.array([2, 2, 1, 1], dtype=np.int64), "heads"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _options(**overrides) -> CompilerOptions:
    values = {
        "state_pairs": (StatePair("past", "present"),),
        "dim_param_upper_bounds": {"p": _CAPACITY_BOUND},
    }
    values.update(overrides)
    return CompilerOptions(**values)


def test_growing_state_is_one_buffer_with_capacity() -> None:
    generated = Compiler(_options()).compile(_kv_cache_model())
    assert "    float past[2][8][3]; /* past -> present */" in generated
    # Concat only writes the new row; the old rows already are in place.
    assert "void node1_concat(int p, const float input_1[2][1][3], " in generated
    assert "output[i0][i1 + p][i2] = input_1[i0][i1][i2];" in generated
    # Readers declare the capacity and loop over the valid rows only.
    assert "const float input0[2][8][3], float output[2][3][(p + 1)]" in generated
    assert "for (idx_t i2 = 0; i2 < (p + 1); ++i2)" in generated
    assert "    if (p < 0 || p > 7) {" in generated
    assert "node1_concat(p, tmp0_new, state->past);" in generated


def test_growing_state_needs_a_capacity() -> None:
    with pytest.raises(StateAliasingError, match="needs a capacity"):
        Compiler(_options(dim_param_upper_bounds=None)).compile(_kv_cache_model())


def test_growing_state_must_append() -> None:
    with pytest.raises(StateAliasingError, match="only the new rows are written"):
        Compiler(_options()).compile(_kv_cache_model(writer="prepend"))


_DRIVER = r"""
#include <stdio.h>

int main(void) {
    model_state state;
    model_state_reset(&state);
    for (int step = 0; step < STEPS; ++step) {
        int p = step;
        float x[2][1][3];
        float q[2][1][3];
        float scores[2][1][p + 1];
        float repeated[2][2][p + 1][3];
        float probs[2][1][p + 1];
        float context[2][1][3];
        float broadcast[2][2][p + 1][3];
        if (fread(x, sizeof(x), 1, stdin) != 1 || fread(q, sizeof(q), 1, stdin) != 1
            || fread(probs, sizeof(probs), 1, stdin) != 1) {
            return 1;
        }
        model(p, &state, x, q, probs, scores, repeated, context, broadcast);
        printf("%s[[", step ? "," : "");
        for (int h = 0; h < 2; ++h)
            for (int t = 0; t <= p; ++t)
                printf("%s%.9g", (h || t) ? "," : "", scores[h][0][t]);
        printf("],[");
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int t = 0; t <= p; ++t)
                    for (int d = 0; d < 3; ++d)
                        printf("%s%.9g", (a || b || t || d) ? "," : "", repeated[a][b][t][d]);
        printf("],[");
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < 3; ++d)
                printf("%s%.9g", (h || d) ? "," : "", context[h][0][d]);
        printf("],[");
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int t = 0; t <= p; ++t)
                    for (int d = 0; d < 3; ++d)
                        printf("%s%.9g", (a || b || t || d) ? "," : "", broadcast[a][b][t][d]);
        printf("]]");
    }
    printf("\n");
    return 0;
}
"""


def test_growing_state_matches_onnxruntime_over_steps(tmp_path: Path) -> None:
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    model = _kv_cache_model()
    generated = Compiler(_options()).compile(model)
    source = tmp_path / "driver.c"
    source.write_text(
        generated + f"\n#define STEPS {_STEPS}\n" + _DRIVER, encoding="utf-8"
    )
    exe = tmp_path / "driver"
    subprocess.run(
        [compiler_cmd, "-std=c99", "-O1", str(source), "-o", str(exe), "-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    rng = np.random.default_rng(0)
    steps = [
        (
            rng.standard_normal((2, 1, 3)).astype(np.float32),
            rng.standard_normal((2, 1, 3)).astype(np.float32),
            rng.standard_normal((2, 1, step + 1)).astype(np.float32),
        )
        for step in range(_STEPS)
    ]
    stdin = b"".join(x.tobytes() + q.tobytes() + pr.tobytes() for x, q, pr in steps)
    result = subprocess.run([str(exe)], input=stdin, check=True, capture_output=True)
    got = json.loads("[" + result.stdout.decode() + "]")

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    past = np.zeros((2, 0, 3), dtype=np.float32)
    for step, (x, q, probs) in enumerate(steps):
        present, *expected = session.run(
            None, {"past": past, "x": x, "q": q, "probs": probs}
        )
        for got_values, want in zip(got[step], expected):
            np.testing.assert_allclose(
                np.array(got_values, dtype=np.float32), want.reshape(-1), rtol=1e-5
            )
        past = present


def test_testbench_rejects_growing_state_pairs() -> None:
    with pytest.raises(Exception, match="does not support growing state pairs"):
        Compiler(_options(emit_testbench=True)).compile(_kv_cache_model())


def test_growing_state_rejects_flat_readers() -> None:
    # Flatten would read the cache through the generic flat Reshape kernel.
    model = _kv_cache_model()
    model.graph.node.append(helper.make_node("Flatten", ["present"], ["flat"]))
    model.graph.output.append(_float("flat", [2, "3*p + 3"]))
    with pytest.raises(StateAliasingError, match="cannot read such a strided view"):
        Compiler(_options()).compile(model)
