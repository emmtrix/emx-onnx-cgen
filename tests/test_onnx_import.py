from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper
from shared.scalar_types import ScalarType

from emx_onnx_cgen.onnx_import import import_onnx
from emx_onnx_cgen.dtypes import scalar_type_from_onnx


def _make_constant_model() -> tuple[onnx.ModelProto, np.ndarray]:
    const_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = helper.make_tensor(
        "const_tensor",
        TensorProto.FLOAT,
        dims=const_values.shape,
        vals=const_values.flatten().tolist(),
    )
    node = helper.make_node("Constant", inputs=[], outputs=["const_out"], value=tensor)
    output = helper.make_tensor_value_info(
        "const_out", TensorProto.FLOAT, const_values.shape
    )
    graph = helper.make_graph([node], "const_graph", [], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model, const_values


def test_import_constant_creates_initializer() -> None:
    model, const_values = _make_constant_model()
    graph = import_onnx(model)
    assert not graph.nodes
    assert len(graph.initializers) == 1
    initializer = graph.initializers[0]
    assert initializer.name == "const_out"
    np.testing.assert_array_equal(initializer.data, const_values)


def test_import_constant_with_value_strings_creates_string_initializer() -> None:
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_out"],
        value_strings=[b"alpha", b"beta"],
    )
    output = helper.make_tensor_value_info("const_out", TensorProto.STRING, [2])
    graph = helper.make_graph([node], "const_string_graph", [], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    imported = import_onnx(model)
    assert not imported.nodes
    assert len(imported.initializers) == 1
    initializer = imported.initializers[0]
    assert initializer.type.dtype == ScalarType.STRING
    np.testing.assert_array_equal(
        initializer.data, np.array([b"alpha", b"beta"], dtype=object)
    )


def test_scalar_type_from_onnx_maps_bfloat16_to_bfloat16() -> None:
    assert scalar_type_from_onnx(TensorProto.BFLOAT16) == ScalarType.BF16


def test_import_bfloat16_value_info_keeps_bfloat16() -> None:
    input_value = helper.make_tensor_value_info("in0", TensorProto.BFLOAT16, [2, 2])
    output_value = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [2, 2])
    node = helper.make_node("Identity", inputs=["in0"], outputs=["out"])
    graph = helper.make_graph([node], "bf16_graph", [input_value], [output_value])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    imported = import_onnx(model)
    assert imported.inputs[0].type.dtype == ScalarType.BF16
    assert imported.outputs[0].type.dtype == ScalarType.BF16


def test_bfloat16_scalar_type_uses___bf16_c_type() -> None:
    assert ScalarType.BF16.c_type == "__bf16"


def test_import_if_with_tensor_branches_expands_to_where() -> None:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])

    then_const = helper.make_tensor("then_const", TensorProto.FLOAT, [2], [1.0, 2.0])
    else_const = helper.make_tensor("else_const", TensorProto.FLOAT, [2], [3.0, 4.0])
    then_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", inputs=[], outputs=["then_out"], value=then_const
            )
        ],
        "then_graph",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", inputs=[], outputs=["else_out"], value=else_const
            )
        ],
        "else_graph",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "if_graph", [cond], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    imported = import_onnx(model)

    assert [node.op_type for node in imported.nodes] == ["Where"]
    assert len(imported.initializers) == 2
