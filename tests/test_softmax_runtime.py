from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_softmax_model(axis: int, shape: list[int]) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    node = helper.make_node("Softmax", inputs=["in0"], outputs=["out"], axis=axis)
    graph = helper.make_graph([node], "softmax_graph", [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_runtime_matches_onnxruntime(axis: int) -> None:
    shape = [2, 3, 4]
    model = _make_softmax_model(axis, shape)
    inputs = np.array(
        [
            [
                [1000.0, 1001.0, 1002.0, 1003.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [-2.0, -1.0, 0.0, 1.0],
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        ],
        dtype=np.float32,
    )
    compiler = Compiler()
    outputs = compiler.run(model, {"in0": inputs})
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = sess.run(None, {"in0": inputs})
    np.testing.assert_allclose(outputs["out"], ort_out, rtol=1e-4, atol=1e-5)
