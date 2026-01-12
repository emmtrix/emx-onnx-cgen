from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler


def _make_sce_model(
    *, reduction: str, with_weight: bool, ignore_index: int | None, with_log_prob: bool
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
    outputs = [
        helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    ]
    if with_log_prob:
        outputs.append(
            helper.make_tensor_value_info(
                "log_prob", TensorProto.FLOAT, input_shape
            )
        )
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=input_names,
        outputs=[output.name for output in outputs],
        reduction=reduction,
        ignore_index=ignore_index,
    )
    graph = helper.make_graph(
        [node],
        "sce_graph",
        inputs,
        outputs,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _expected_sce(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int | None,
    return_log_prob: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    input_shape = values.shape
    target_shape = target.shape
    n = input_shape[0]
    c = input_shape[1]
    max_values = np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values - max_values)
    log_prob = np.log(exp_values / np.sum(exp_values, axis=1, keepdims=True))
    log_prob_output = log_prob if return_log_prob else None
    gather_weight = None
    if weight is not None:
        gather_weight = np.take(weight, target.astype(np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=values.dtype
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(
            dtype=values.dtype
        )
    if len(input_shape) != 3:
        log_prob = log_prob.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = log_prob.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if ignore_index is None or target[i][d_index] != ignore_index:
                loss[i][d_index] = -log_prob[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            loss = loss.astype(values.dtype)
            if return_log_prob and log_prob_output is not None:
                return loss, log_prob_output.astype(values.dtype)
            return loss, None
    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    loss = loss.astype(values.dtype)
    if return_log_prob and log_prob_output is not None:
        return loss, log_prob_output.astype(values.dtype)
    return loss, None


def test_softmax_cross_entropy_loss_reduction_none_matches_expected() -> None:
    model = _make_sce_model(
        reduction="none", with_weight=True, ignore_index=2, with_log_prob=False
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
    expected, _ = _expected_sce(
        values,
        target,
        weight,
        reduction="none",
        ignore_index=2,
        return_log_prob=False,
    )
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-4, atol=1e-5)


def test_softmax_cross_entropy_loss_log_prob_matches_onnxruntime() -> None:
    model = _make_sce_model(
        reduction="mean", with_weight=False, ignore_index=None, with_log_prob=True
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
    ort_out, ort_log_prob = sess.run(None, {"input": values, "target": target})
    np.testing.assert_allclose(outputs["out"], ort_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        outputs["log_prob"], ort_log_prob, rtol=1e-4, atol=1e-5
    )
