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
from onnx import TensorProto, helper, numpy_helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.errors import UnsupportedOpError
from emx_onnx_cgen.onnx_import import import_onnx
from emx_onnx_cgen.sparse_inputs import coo_arrays, expand_sparse_inputs
from emx_onnx_cgen.testbench import decode_testbench_array


def _sparse_matmul_model(
    *,
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
    out_shape: tuple[int, int],
    elem_type: int = TensorProto.FLOAT,
    **attrs: object,
) -> onnx.ModelProto:
    node = helper.make_node(
        "SparseToDenseMatMul", ["A", "B"], ["Y"], domain="com.microsoft", **attrs
    )
    graph = helper.make_graph(
        [node],
        "sparse_matmul",
        [
            helper.make_sparse_tensor_value_info("A", elem_type, list(a_shape)),
            helper.make_tensor_value_info("B", elem_type, list(b_shape)),
        ],
        [helper.make_tensor_value_info("Y", elem_type, list(out_shape))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8
    return model


def test_sparse_graph_input_expands_to_coo_values_and_indices() -> None:
    model = _sparse_matmul_model(a_shape=(3, 4), b_shape=(4, 5), out_shape=(3, 5))
    graph = import_onnx(model)
    inputs = {value.name: value.type for value in graph.inputs}
    assert list(inputs) == ["A_values", "A_indices", "B"]
    assert inputs["A_values"].dim_params == ("A_nnz",)
    assert inputs["A_indices"].dim_params == ("A_nnz", None)
    assert inputs["A_indices"].shape[1] == 2
    assert graph.nodes[0].inputs == ("A_values", "A_indices", "B")


def test_sparse_graph_input_rejects_other_consumers() -> None:
    model = _sparse_matmul_model(a_shape=(3, 4), b_shape=(4, 5), out_shape=(3, 5))
    model.graph.node.append(helper.make_node("Identity", ["A"], ["A_copy"]))
    with pytest.raises(UnsupportedOpError, match="cannot take the sparse tensor"):
        expand_sparse_inputs(model)


def test_coo_arrays_unravel_linear_indices() -> None:
    sparse = helper.make_sparse_tensor(
        numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32)),
        numpy_helper.from_array(np.array([1, 6], dtype=np.int64)),
        [2, 4],
    )
    values, indices = coo_arrays(sparse)
    np.testing.assert_array_equal(values, [1.0, 2.0])
    np.testing.assert_array_equal(indices, [[0, 1], [1, 2]])


def _run_ort(
    model: onnx.ModelProto,
    dense_shape: tuple[int, int],
    values: np.ndarray,
    indices: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sparse = ort.SparseTensor.sparse_coo_from_numpy(
        list(dense_shape), values, indices, ort.OrtDevice.make("cpu", 0)
    )
    outputs = session.run_with_ort_values(
        ["Y"],
        {
            "A": ort.OrtValue.ort_value_from_sparse_tensor(sparse),
            "B": ort.OrtValue.ortvalue_from_numpy(b),
        },
    )
    return outputs[0].numpy()


def _run_testbench(
    model: onnx.ModelProto, testbench_inputs: dict[str, np.ndarray]
) -> dict[str, object]:
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    options = CompilerOptions(emit_testbench=True, testbench_inputs=testbench_inputs)
    generated = Compiler(options).compile(model)
    with tempfile.TemporaryDirectory() as temp_dir:
        c_path = Path(temp_dir) / "model.c"
        exe_path = Path(temp_dir) / "model"
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [compiler_cmd, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [str(exe_path)], check=True, capture_output=True, text=True
        )
    return json.loads(result.stdout)


_SPARSE_MATMUL_CASES = {
    # name: (A shape, B shape, attributes, dtype). ORT's shape inference for
    # SparseToDenseMatMul ignores transA/transB, so a transposed operand is
    # kept square for ORT to accept the model.
    "plain": ((3, 4), (4, 5), {}, np.float32),
    "trans_a": ((4, 4), (4, 5), {"transA": 1}, np.float32),
    "trans_b": ((3, 4), (4, 4), {"transB": 1}, np.float32),
    "trans_both_alpha": (
        (4, 4),
        (4, 4),
        {"transA": 1, "transB": 1, "alpha": 2.5},
        np.float32,
    ),
    "int64": ((3, 4), (4, 5), {}, np.int64),
}


@pytest.mark.parametrize("case", sorted(_SPARSE_MATMUL_CASES))
def test_sparse_to_dense_matmul_matches_onnxruntime(case: str) -> None:
    a_shape, b_shape, attrs, np_dtype = _SPARSE_MATMUL_CASES[case]
    trans_a = bool(attrs.get("transA", 0))
    trans_b = bool(attrs.get("transB", 0))
    m = a_shape[1] if trans_a else a_shape[0]
    n = b_shape[0] if trans_b else b_shape[1]
    elem_type = helper.np_dtype_to_tensor_dtype(np.dtype(np_dtype))
    model = _sparse_matmul_model(
        a_shape=a_shape,
        b_shape=b_shape,
        out_shape=(m, n),
        elem_type=elem_type,
        **attrs,
    )
    rng = np.random.default_rng(7)
    # Unsorted coordinates, including a duplicate that has to be accumulated.
    indices = np.array([[2, 1], [0, 3], [1, 0], [2, 1], [0, 0]], dtype=np.int64)
    indices = np.minimum(indices, np.array(a_shape, dtype=np.int64) - 1)
    values = rng.integers(-4, 5, size=len(indices)).astype(np_dtype)
    b = rng.integers(-4, 5, size=b_shape).astype(np_dtype)
    expected = _run_ort(model, a_shape, values, indices, b)
    payload = _run_testbench(model, {"A_values": values, "A_indices": indices, "B": b})
    output = payload["outputs"]["Y"]
    actual = decode_testbench_array(output["data"], expected.dtype).reshape(
        expected.shape
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "trans_a", "trans_b"),
    [((4, 3), (4, 5), 1, 0), ((3, 4), (5, 4), 0, 1), ((4, 3), (5, 4), 1, 1)],
)
def test_sparse_to_dense_matmul_non_square_transposes(
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
    trans_a: int,
    trans_b: int,
) -> None:
    # ORT rejects these models in its shape inference, so compare with numpy.
    m = a_shape[1] if trans_a else a_shape[0]
    n = b_shape[0] if trans_b else b_shape[1]
    model = _sparse_matmul_model(
        a_shape=a_shape,
        b_shape=b_shape,
        out_shape=(m, n),
        transA=trans_a,
        transB=trans_b,
    )
    indices = np.array([[3, 2], [0, 1], [3, 2], [2, 0]], dtype=np.int64)
    indices = np.minimum(indices, np.array(a_shape, dtype=np.int64) - 1)
    values = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    b = np.arange(np.prod(b_shape), dtype=np.float32).reshape(b_shape) - 7.0
    dense = np.zeros(a_shape, dtype=np.float32)
    np.add.at(dense, (indices[:, 0], indices[:, 1]), values)
    expected = (dense.T if trans_a else dense) @ (b.T if trans_b else b)
    payload = _run_testbench(model, {"A_values": values, "A_indices": indices, "B": b})
    actual = decode_testbench_array(
        payload["outputs"]["Y"]["data"], np.float32
    ).reshape(m, n)
    np.testing.assert_array_equal(actual, expected)
