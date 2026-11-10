from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from emx_onnx_cgen import cli
from emx_onnx_cgen.explicit_padding import make_padding_explicit

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_TEST_ROOT = REPO_ROOT / "onnx-backend-test-data" / "node"
EXPECTED_ERRORS_ROOT = REPO_ROOT / "tests" / "expected_errors"
NODE_TEST_PREFIXES = (
    "test_basic_conv",
    "test_conv_",
    "test_convinteger",
    "test_qlinearconv",
)


def _model(
    op_type: str,
    input_shape: list[int],
    attrs: dict[str, object],
    *,
    elem_type: int = TensorProto.FLOAT,
    extra_inputs: list[onnx.TensorProto] = (),
    outputs: int = 1,
    opset: int = 19,
    domain: str = "",
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", elem_type, input_shape)
    output_infos = [
        helper.make_tensor_value_info(f"y{index}", TensorProto.UNDEFINED, None)
        for index in range(outputs)
    ]
    node = helper.make_node(
        op_type,
        ["x", *(init.name for init in extra_inputs)],
        [info.name for info in output_infos],
        name=f"{op_type.lower()}_node",
        domain=domain,
        **attrs,
    )
    graph = helper.make_graph(
        [node], "explicit_padding", [x], output_infos, initializer=list(extra_inputs)
    )
    opsets = [helper.make_opsetid("", opset)]
    model = helper.make_model_gen_version(graph, opset_imports=opsets)
    return onnx.shape_inference.infer_shapes(model)


def _conv_weights(
    out_channels: int, in_channels: int, kernel: list[int]
) -> onnx.TensorProto:
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(out_channels, in_channels, *kernel)).astype(np.float32)
    return numpy_helper.from_array(weights, "w")


def _run_ort(model: onnx.ModelProto, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, inputs)


def _assert_equivalent(
    model: onnx.ModelProto, x: np.ndarray, expected_pads: int
) -> None:
    rewritten, count = make_padding_explicit(model)
    assert count == expected_pads
    for expected, actual in zip(
        _run_ort(model, {"x": x}), _run_ort(rewritten, {"x": x})
    ):
        np.testing.assert_array_equal(actual, expected)
    for node in rewritten.graph.node:
        if node.op_type != "Pad":
            pads = [a for a in node.attribute if a.name == "pads"]
            assert all(value == 0 for attr in pads for value in attr.ints)


def _random_input(shape: list[int], dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)
    return rng.normal(size=shape).astype(dtype)


@pytest.mark.parametrize(
    "attrs",
    [
        {"pads": [1, 1, 1, 1]},
        {"pads": [0, 2, 1, 0], "strides": [2, 1]},
        {"pads": [2, 2, 2, 2], "dilations": [2, 2]},
        {"auto_pad": "SAME_UPPER", "strides": [2, 2]},
        {"auto_pad": "SAME_LOWER"},
    ],
    ids=["symmetric", "asymmetric_strided", "dilated", "same_upper", "same_lower"],
)
def test_conv_padding_becomes_pad_node(attrs: dict[str, object]) -> None:
    model = _model(
        "Conv",
        [1, 2, 7, 6],
        {"kernel_shape": [3, 3], **attrs},
        extra_inputs=[_conv_weights(3, 2, [3, 3])],
    )
    _assert_equivalent(model, _random_input([1, 2, 7, 6]), 1)


def test_conv_without_padding_is_kept() -> None:
    model = _model(
        "Conv",
        [1, 2, 5, 5],
        {"kernel_shape": [3, 3], "auto_pad": "VALID"},
        extra_inputs=[_conv_weights(3, 2, [3, 3])],
    )
    assert make_padding_explicit(model) == (model, 0)


def test_conv_opset_10_uses_pad_attributes() -> None:
    model = _model(
        "Conv",
        [1, 1, 5, 5],
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        extra_inputs=[_conv_weights(2, 1, [3, 3])],
        opset=10,
    )
    rewritten, _ = make_padding_explicit(model)
    pad = next(node for node in rewritten.graph.node if node.op_type == "Pad")
    assert list(pad.input) == ["x"]
    _assert_equivalent(model, _random_input([1, 1, 5, 5]), 1)


@pytest.mark.parametrize(
    ("op_type", "attrs"),
    [
        ("MaxPool", {}),
        ("AveragePool", {"count_include_pad": 1}),
        ("LpPool", {"p": 2}),
    ],
    ids=["maxpool", "averagepool", "lppool"],
)
def test_pooling_padding_is_kept(op_type: str, attrs: dict[str, object]) -> None:
    model = _model(
        op_type,
        [1, 1, 6, 6],
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], **attrs},
    )
    assert make_padding_explicit(model) == (model, 0)


def _scalar(name: str, value: float | int, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def test_qlinearconv_pads_with_input_zero_point() -> None:
    scalar = _scalar
    weights = numpy_helper.from_array(
        np.random.default_rng(2).integers(0, 255, size=(2, 1, 3, 3)).astype(np.uint8),
        "w",
    )
    inputs = [
        scalar("x_scale", 0.05, np.float32),
        scalar("x_zero_point", 128, np.uint8),
        weights,
        scalar("w_scale", 0.02, np.float32),
        scalar("w_zero_point", 120, np.uint8),
        scalar("y_scale", 0.5, np.float32),
        scalar("y_zero_point", 100, np.uint8),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, None)
    node = helper.make_node(
        "QLinearConv",
        ["x", *(init.name for init in inputs)],
        ["y"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "qlinearconv", [x], [y], initializer=inputs)
    model = onnx.shape_inference.infer_shapes(
        helper.make_model_gen_version(
            graph, opset_imports=[helper.make_opsetid("", 19)]
        )
    )
    rewritten, _ = make_padding_explicit(model)
    value = next(
        init for init in rewritten.graph.initializer if init.name.endswith("_value")
    )
    assert numpy_helper.to_array(value) == 128
    _assert_equivalent(model, _random_input([1, 1, 5, 5], np.uint8), 1)


def _node_test_names() -> list[str]:
    if not NODE_TEST_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in NODE_TEST_ROOT.iterdir()
        if path.name.startswith(NODE_TEST_PREFIXES) and "transpose" not in path.name
    )


def _expected_result(name: str) -> str:
    expectation = EXPECTED_ERRORS_ROOT / (
        f"onnx-backend-test-data__node__{name}__model.onnx.json"
    )
    if not expectation.is_file():
        return ""
    return json.loads(expectation.read_text(encoding="utf-8")).get("error", "")


@pytest.mark.parametrize("name", _node_test_names())
def test_official_node_tests_verify_with_explicit_padding(name: str) -> None:
    if not _expected_result(name).startswith("OK"):
        pytest.skip("Not verified without explicit padding either")
    model_dir = NODE_TEST_ROOT / name
    result = cli.run_cli_command(
        [
            "emx-onnx-cgen",
            "verify",
            "--model-base-dir",
            str(model_dir.relative_to(REPO_ROOT)),
            "model.onnx",
            "--test-data-dir",
            "test_data_set_0",
            "--explicit-padding",
        ]
    )
    assert result.exit_code == 0, result.result
    assert (result.result or "").startswith("OK"), result.result


def test_padded_tensor_gets_value_info() -> None:
    model = _model(
        "Conv",
        [1, 2, 7, 6],
        {"kernel_shape": [3, 3], "pads": [1, 0, 2, 3]},
        extra_inputs=[_conv_weights(3, 2, [3, 3])],
    )
    rewritten, _ = make_padding_explicit(model)
    pad = next(node for node in rewritten.graph.node if node.op_type == "Pad")
    info = next(v for v in rewritten.graph.value_info if v.name == pad.output[0])
    assert [dim.dim_value for dim in info.type.tensor_type.shape.dim] == [1, 2, 10, 9]


def test_same_upper_without_kernel_shape_uses_weights() -> None:
    model = _model(
        "Conv",
        [1, 2, 7, 6],
        {"auto_pad": "SAME_UPPER"},
        extra_inputs=[_conv_weights(3, 2, [3, 2])],
    )
    _assert_equivalent(model, _random_input([1, 2, 7, 6]), 1)


def test_opset_1_is_kept() -> None:
    model = _model(
        "Conv",
        [1, 1, 5, 5],
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        extra_inputs=[_conv_weights(2, 1, [3, 3])],
        opset=1,
    )
    assert make_padding_explicit(model) == (model, 0)


def test_new_names_do_not_collide_with_subgraphs() -> None:
    model = _model(
        "Conv",
        [1, 1, 5, 5],
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        extra_inputs=[_conv_weights(2, 1, [3, 3])],
    )
    body = helper.make_graph(
        [helper.make_node("Identity", ["cond_in"], ["x_padded"])],
        "body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("x_padded", TensorProto.BOOL, [])],
    )
    trip = numpy_helper.from_array(np.array(1, dtype=np.int64), "trip")
    cond = numpy_helper.from_array(np.array(True), "cond")
    model.graph.initializer.extend([trip, cond])
    model.graph.node.append(helper.make_node("Loop", ["trip", "cond"], [], body=body))
    rewritten, count = make_padding_explicit(model)
    assert count == 1
    assert "x_padded" not in {
        name for node in rewritten.graph.node for name in node.output
    }


def _verify_c(model: onnx.ModelProto, tmp_path: Path) -> None:
    """The generated C code of `model` with explicit padding matches ONNX Runtime."""
    onnx.checker.check_model(model)
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    result = cli.run_cli_command(
        ["emx-onnx-cgen", "verify", str(model_path), "--explicit-padding"]
    )
    assert result.exit_code == 0, result.result
    assert (result.result or "").startswith("OK"), result.result


def _graph_model(
    node: onnx.NodeProto,
    inputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
    opsets: list[onnx.OperatorSetIdProto],
    output: onnx.ValueInfoProto | None = None,
) -> onnx.ModelProto:
    if output is None:
        output = helper.make_tensor_value_info(
            node.output[0], TensorProto.UNDEFINED, None
        )
    graph = helper.make_graph(
        [node], "explicit_padding", inputs, [output], initializer=initializers
    )
    # Derive the IR version from the opsets: the onnx default can be newer than
    # onnxruntime supports. Custom domains such as com.microsoft are unknown to
    # make_model_gen_version, hence ignore_unknown.
    model = helper.make_model(
        graph,
        opset_imports=opsets,
        ir_version=helper.find_min_ir_version_for(opsets, ignore_unknown=True),
    )
    return onnx.shape_inference.infer_shapes(model)


def test_c_convinteger_with_zero_point(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    weights = numpy_helper.from_array(
        rng.integers(0, 255, size=(4, 1, 3, 2)).astype(np.uint8), "w"
    )
    node = helper.make_node(
        "ConvInteger",
        ["x", "w", "x_zero_point"],
        ["y"],
        kernel_shape=[3, 2],
        pads=[0, 2, 1, 0],
        group=2,
        dilations=[2, 1],
    )
    model = _graph_model(
        node,
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 2, 6, 5])],
        [weights, _scalar("x_zero_point", 37, np.uint8)],
        [helper.make_opsetid("", 19)],
    )
    assert make_padding_explicit(model)[1] == 1
    _verify_c(model, tmp_path)


def test_c_qlinearconv_int8_3d(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    weights = numpy_helper.from_array(
        rng.integers(-100, 100, size=(2, 1, 2, 3, 2)).astype(np.int8), "w"
    )
    initializers = [
        _scalar("x_scale", 0.05, np.float32),
        _scalar("x_zero_point", -7, np.int8),
        weights,
        _scalar("w_scale", 0.02, np.float32),
        _scalar("w_zero_point", 0, np.int8),
        _scalar("y_scale", 0.3, np.float32),
        _scalar("y_zero_point", 5, np.int8),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["x", *(init.name for init in initializers)],
        ["y"],
        kernel_shape=[2, 3, 2],
        pads=[1, 0, 1, 0, 2, 1],
        group=2,
    )
    model = _graph_model(
        node,
        [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 4, 5, 3])],
        initializers,
        [helper.make_opsetid("", 19)],
    )
    assert make_padding_explicit(model)[1] == 1
    _verify_c(model, tmp_path)


def test_c_fused_conv(tmp_path: Path) -> None:
    node = helper.make_node(
        "FusedConv",
        ["x", "w"],
        ["y"],
        domain="com.microsoft",
        kernel_shape=[3, 3],
        pads=[1, 0, 0, 2],
        activation="Relu",
    )
    model = _graph_model(
        node,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 5, 6])],
        [_conv_weights(3, 2, [3, 3])],
        [helper.make_opsetid("", 19), helper.make_opsetid("com.microsoft", 1)],
        # ONNX shape inference does not know the contrib operator
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 6]),
    )
    assert make_padding_explicit(model)[1] == 1
    _verify_c(model, tmp_path)


def test_c_conv1d_same_upper_end_padding(tmp_path: Path) -> None:
    # Even input, stride 2, kernel 3: SAME_UPPER pads only at the end
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        kernel_shape=[3],
        strides=[2],
        auto_pad="SAME_UPPER",
        group=3,
    )
    model = _graph_model(
        node,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8])],
        [_conv_weights(3, 1, [3])],
        [helper.make_opsetid("", 19)],
    )
    assert make_padding_explicit(model)[1] == 1
    _verify_c(model, tmp_path)
