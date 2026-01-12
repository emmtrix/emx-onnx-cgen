from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_nll_model(
    *, reduction: str, with_weight: bool, ignore_index: int
) -> onnx.ModelProto:
    input_shape = [2, 3, 2]
    target_shape = [2, 2]
    output_shape = target_shape if reduction == "none" else []
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    target_info = helper.make_tensor_value_info(
        "target", TensorProto.INT64, target_shape
    )
    inputs = [input_info, target_info]
    input_names = ["input", "target"]
    if with_weight:
        weight_info = helper.make_tensor_value_info(
            "weight", TensorProto.FLOAT, [input_shape[1]]
        )
        inputs.append(weight_info)
        input_names.append("weight")
    output_info = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=input_names,
        outputs=[output_info.name],
        reduction=reduction,
        ignore_index=ignore_index,
    )
    graph = helper.make_graph(
        [node],
        "nllloss_graph",
        inputs,
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _expected_nll(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int,
) -> np.ndarray:
    input_shape = values.shape
    n = input_shape[0]
    c = input_shape[1]
    if len(input_shape) != 3:
        values = values.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = values.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if target[i][d_index] != ignore_index:
                loss[i][d_index] = -values[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target.shape)
    if weight is not None:
        gather_weight = np.take(
            weight, np.array(target, dtype=np.int32), mode="clip"
        )
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight)
        loss = loss * gather_weight
        if reduction == "mean":
            return (loss.sum() / gather_weight.sum()).astype(values.dtype)
    if reduction == "mean":
        return np.mean(loss).astype(values.dtype)
    if reduction == "sum":
        return np.sum(loss).astype(values.dtype)
    return loss.astype(values.dtype)


def test_nllloss_reduction_none_matches_expected() -> None:
    model = _make_nll_model(
        reduction="none", with_weight=True, ignore_index=2
    )
    values = np.array(
        [
            [[1.0, 2.0], [2.5, 3.5], [4.0, 5.0]],
            [[-1.0, 0.5], [1.5, -2.5], [3.0, -4.0]],
        ],
        dtype=np.float32,
    )
    target = np.array([[2, 1], [0, 2]], dtype=np.int64)
    weight = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compiler = Compiler()
    outputs = compiler.run(
        model, {"input": values, "target": target, "weight": weight}
    )
    expected = _expected_nll(
        values, target, weight, reduction="none", ignore_index=2
    )
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-4, atol=1e-5)


def test_nllloss_reduction_mean_matches_onnxruntime() -> None:
    model = _make_nll_model(
        reduction="mean", with_weight=False, ignore_index=-1
    )
    values = np.array(
        [
            [[1.5, 0.5], [2.0, -1.0], [0.0, 3.0]],
            [[-2.0, 1.0], [1.0, 2.0], [0.5, -0.5]],
        ],
        dtype=np.float32,
    )
    target = np.array([[0, 2], [1, 2]], dtype=np.int64)
    compiler = Compiler()
    outputs = compiler.run(model, {"input": values, "target": target})
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = sess.run(None, {"input": values, "target": target})
    np.testing.assert_allclose(outputs["out"], ort_out, rtol=1e-4, atol=1e-5)
