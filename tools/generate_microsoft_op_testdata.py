"""Generate ORT-based test node data for Microsoft custom operators.

This script creates test cases for Microsoft (com.microsoft domain) operators
using ONNX Runtime as the reference implementation. The test data is stored in
``tests/onnx/microsoft/`` in the standard ONNX node test format:

    tests/onnx/microsoft/<test_name>/
        model.onnx
        test_data_set_0/
            input_0.pb   (one file per non-initializer model input)
            output_0.pb  (one file per model output)

The generated data is compatible with the existing test infrastructure in
``tests/test_official_onnx_files.py`` (``test_local_repo_onnx_expected_errors``).

Usage::

    python tools/generate_microsoft_op_testdata.py

Run ``UPDATE_REFS=1 pytest tests/test_official_onnx_files.py -k microsoft``
afterwards to regenerate the expected-error baselines.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = _REPO_ROOT / "tests" / "onnx" / "microsoft"

COM_MICROSOFT = "com.microsoft"


def _save_test_data(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    outputs: list[np.ndarray],
    test_dir: Path,
) -> None:
    """Persist a model and a single test data set under *test_dir*."""
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "model.onnx").write_bytes(model.SerializeToString())

    data_dir = test_dir / "test_data_set_0"
    data_dir.mkdir(exist_ok=True)

    initializer_names = {init.name for init in model.graph.initializer}
    non_init_inputs = [
        vi for vi in model.graph.input if vi.name not in initializer_names
    ]

    for idx, vi in enumerate(non_init_inputs):
        arr = inputs[vi.name]
        tensor = numpy_helper.from_array(arr, name=vi.name)
        (data_dir / f"input_{idx}.pb").write_bytes(tensor.SerializeToString())

    for idx, (vi, arr) in enumerate(zip(model.graph.output, outputs)):
        tensor = numpy_helper.from_array(arr, name=vi.name)
        (data_dir / f"output_{idx}.pb").write_bytes(tensor.SerializeToString())

    print(f"  Wrote {test_dir.relative_to(_REPO_ROOT)}")


def _run_ort(model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, inputs)


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------


def _generate_qlinear_add() -> None:
    """com.microsoft::QLinearAdd – element-wise quantized addition."""
    a_scale_val = np.float32(0.05)
    a_zp_val = np.uint8(128)
    b_scale_val = np.float32(0.05)
    b_zp_val = np.uint8(128)
    y_scale_val = np.float32(0.1)
    y_zp_val = np.uint8(128)

    a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [a_scale_val])
    a_zp = helper.make_tensor("a_zero_point", TensorProto.UINT8, [], [a_zp_val])
    b_scale = helper.make_tensor("b_scale", TensorProto.FLOAT, [], [b_scale_val])
    b_zp = helper.make_tensor("b_zero_point", TensorProto.UINT8, [], [b_zp_val])
    y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [y_scale_val])
    y_zp = helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [y_zp_val])

    input_a = helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 4])
    input_b = helper.make_tensor_value_info("b", TensorProto.UINT8, [1, 4])
    output = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 4])

    node = helper.make_node(
        "QLinearAdd",
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        domain=COM_MICROSOFT,
    )
    graph = helper.make_graph(
        [node],
        "qlinearadd_graph",
        [input_a, input_b],
        [output],
        initializer=[a_scale, a_zp, b_scale, b_zp, y_scale, y_zp],
    )
    model = helper.make_model(
        graph,
        producer_name="generate_microsoft_op_testdata",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid(COM_MICROSOFT, 1),
        ],
    )
    model.ir_version = 7

    rng = np.random.default_rng(0)
    inputs = {
        "a": rng.integers(100, 160, size=(1, 4), dtype=np.uint8),
        "b": rng.integers(100, 160, size=(1, 4), dtype=np.uint8),
    }
    outputs = _run_ort(model, inputs)
    _save_test_data(model, inputs, outputs, _OUTPUT_ROOT / "test_qlinear_add")


def _generate_qlinear_mul() -> None:
    """com.microsoft::QLinearMul – element-wise quantized multiplication."""
    a_scale_val = np.float32(0.05)
    a_zp_val = np.uint8(128)
    b_scale_val = np.float32(0.05)
    b_zp_val = np.uint8(128)
    y_scale_val = np.float32(0.01)
    y_zp_val = np.uint8(128)

    a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [a_scale_val])
    a_zp = helper.make_tensor("a_zero_point", TensorProto.UINT8, [], [a_zp_val])
    b_scale = helper.make_tensor("b_scale", TensorProto.FLOAT, [], [b_scale_val])
    b_zp = helper.make_tensor("b_zero_point", TensorProto.UINT8, [], [b_zp_val])
    y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [y_scale_val])
    y_zp = helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [y_zp_val])

    input_a = helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 4])
    input_b = helper.make_tensor_value_info("b", TensorProto.UINT8, [1, 4])
    output = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 4])

    node = helper.make_node(
        "QLinearMul",
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        domain=COM_MICROSOFT,
    )
    graph = helper.make_graph(
        [node],
        "qlinearmul_graph",
        [input_a, input_b],
        [output],
        initializer=[a_scale, a_zp, b_scale, b_zp, y_scale, y_zp],
    )
    model = helper.make_model(
        graph,
        producer_name="generate_microsoft_op_testdata",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid(COM_MICROSOFT, 1),
        ],
    )
    model.ir_version = 7

    rng = np.random.default_rng(0)
    inputs = {
        "a": rng.integers(100, 160, size=(1, 4), dtype=np.uint8),
        "b": rng.integers(100, 160, size=(1, 4), dtype=np.uint8),
    }
    outputs = _run_ort(model, inputs)
    _save_test_data(model, inputs, outputs, _OUTPUT_ROOT / "test_qlinear_mul")


def _generate_qlinear_softmax() -> None:
    """com.microsoft::QLinearSoftmax – quantized softmax."""
    x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [], [np.float32(0.125)])
    x_zp = helper.make_tensor("x_zero_point", TensorProto.UINT8, [], [np.uint8(3)])
    y_scale = helper.make_tensor(
        "y_scale", TensorProto.FLOAT, [], [np.float32(1.0 / 256.0)]
    )
    y_zp = helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [np.uint8(0)])

    input_x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 2, 4])
    output = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 2, 4])

    node = helper.make_node(
        "QLinearSoftmax",
        inputs=["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        outputs=["y"],
        domain=COM_MICROSOFT,
        opset=13,
    )
    graph = helper.make_graph(
        [node],
        "qlinearsoftmax_graph",
        [input_x],
        [output],
        initializer=[x_scale, x_zp, y_scale, y_zp],
    )
    model = helper.make_model(
        graph,
        producer_name="generate_microsoft_op_testdata",
        opset_imports=[
            helper.make_operatorsetid("", 15),
            helper.make_operatorsetid(COM_MICROSOFT, 1),
        ],
    )
    model.ir_version = 7

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.integers(10, 80, size=(1, 2, 4), dtype=np.uint8),
    }
    outputs = _run_ort(model, inputs)
    _save_test_data(model, inputs, outputs, _OUTPUT_ROOT / "test_qlinear_softmax")


def _generate_qlinear_average_pool() -> None:
    """com.microsoft::QLinearAveragePool – quantized average pooling (NCHW)."""
    x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [], [np.float32(0.1)])
    x_zp = helper.make_tensor("x_zero_point", TensorProto.UINT8, [], [np.uint8(128)])
    y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [np.float32(0.1)])
    y_zp = helper.make_tensor("y_zero_point", TensorProto.UINT8, [], [np.uint8(128)])

    input_x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 4, 4])
    output = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 1, 2, 2])

    node = helper.make_node(
        "QLinearAveragePool",
        inputs=["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        outputs=["y"],
        domain=COM_MICROSOFT,
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph = helper.make_graph(
        [node],
        "qlinearaveragepool_graph",
        [input_x],
        [output],
        initializer=[x_scale, x_zp, y_scale, y_zp],
    )
    model = helper.make_model(
        graph,
        producer_name="generate_microsoft_op_testdata",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid(COM_MICROSOFT, 1),
        ],
    )
    model.ir_version = 7

    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.integers(100, 200, size=(1, 1, 4, 4), dtype=np.uint8),
    }
    outputs = _run_ort(model, inputs)
    _save_test_data(model, inputs, outputs, _OUTPUT_ROOT / "test_qlinear_average_pool")


_GENERATORS = [
    _generate_qlinear_add,
    _generate_qlinear_mul,
    _generate_qlinear_softmax,
    _generate_qlinear_average_pool,
]


def main() -> int:
    print(
        f"Generating Microsoft op test data under {_OUTPUT_ROOT.relative_to(_REPO_ROOT)}/"
    )
    for gen in _GENERATORS:
        try:
            gen()
        except (
            Exception
        ) as exc:  # noqa: BLE001 – propagate any ORT/ONNX error as a clean message
            traceback.print_exc()
            print(f"  ERROR in {gen.__name__}: {exc}", file=sys.stderr)
            return 1
    print("Done.")
    print()
    print("Next step: run")
    print("  UPDATE_REFS=1 pytest tests/test_official_onnx_files.py -n auto -q")
    print("to regenerate the expected-error baselines for the new test data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
