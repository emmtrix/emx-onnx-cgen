from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def _run_reference_vs_ort(model: onnx.ModelProto) -> None:
    initializer_names = {init.name for init in model.graph.initializer}
    rng = np.random.default_rng(0)
    inputs: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        shape = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
        inputs[value_info.name] = rng.standard_normal(shape).astype(np.float32)

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, inputs)
    ref_outputs = ReferenceEvaluator(model).run(None, inputs)

    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        np.testing.assert_allclose(ref_output, ort_output, rtol=1e-4, atol=1e-5)


def _make_onehot_model() -> onnx.ModelProto:
    indices_shape = [2, 3]
    indices_info = helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape)
    depth_tensor = helper.make_tensor("depth", TensorProto.INT64, dims=[], vals=[4])
    values_tensor = helper.make_tensor(
        "values", TensorProto.FLOAT, dims=[2], vals=[0.0, 1.0]
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4, 3])
    node = helper.make_node(
        "OneHot",
        inputs=["indices", "depth", "values"],
        outputs=[output.name],
        axis=1,
    )
    graph = helper.make_graph(
        [node],
        "onehot_divergence_graph",
        [indices_info],
        [output],
        initializer=[depth_tensor, values_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_batchnorm_model() -> onnx.ModelProto:
    input_shape = [2, 3, 2, 2]
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shape)
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [3], [1.0, 1.5, -0.5])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.0, 0.1, -0.2])
    mean = helper.make_tensor("mean", TensorProto.FLOAT, [3], [0.5, -0.5, 1.0])
    var = helper.make_tensor("var", TensorProto.FLOAT, [3], [0.25, 0.5, 1.5])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, input_shape)
    node = helper.make_node(
        "BatchNormalization",
        inputs=["in0", "scale", "bias", "mean", "var"],
        outputs=[output.name],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node],
        "batchnorm_divergence_graph",
        [input_info],
        [output],
        initializer=[scale, bias, mean, var],
    )
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


@pytest.mark.xfail(
    reason="Known upstream divergence: ONNX ReferenceEvaluator vs ORT (OneHot)",
    strict=True,
)
def test_ort_vs_reference_onehot_divergence() -> None:
    _run_reference_vs_ort(_make_onehot_model())


@pytest.mark.xfail(
    reason="Known upstream divergence: ONNX ReferenceEvaluator vs ORT (BatchNormalization)",
    strict=True,
)
def test_ort_vs_reference_batchnorm_divergence() -> None:
    _run_reference_vs_ort(_make_batchnorm_model())
