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


def test_import_if_with_sequence_construct_branches_expands_to_where_and_sequence_construct() -> (
    None
):
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    output = helper.make_tensor_sequence_value_info("out", TensorProto.FLOAT, [2])

    then_tensor = helper.make_tensor("then_tensor", TensorProto.FLOAT, [2], [1.0, 2.0])
    else_tensor = helper.make_tensor("else_tensor", TensorProto.FLOAT, [2], [3.0, 4.0])
    then_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", inputs=[], outputs=["then_elem"], value=then_tensor
            ),
            helper.make_node(
                "SequenceConstruct", inputs=["then_elem"], outputs=["then_out"]
            ),
        ],
        "then_graph",
        [],
        [helper.make_tensor_sequence_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", inputs=[], outputs=["else_elem"], value=else_tensor
            ),
            helper.make_node(
                "SequenceConstruct", inputs=["else_elem"], outputs=["else_out"]
            ),
        ],
        "else_graph",
        [],
        [helper.make_tensor_sequence_value_info("else_out", TensorProto.FLOAT, [2])],
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "if_sequence_graph", [cond], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    imported = import_onnx(model)

    op_types = [node.op_type for node in imported.nodes]
    assert op_types[-2:] == ["Where", "SequenceConstruct"]
    assert len(imported.initializers) == 2


def test_import_if_with_optional_sequence_branches_expands_to_supported_ops() -> None:
    model = onnx.load("onnx-org/onnx/backend/test/data/node/test_if_opt/model.onnx")

    imported = import_onnx(model)

    assert all(node.op_type != "If" for node in imported.nodes)
    assert all(node.op_type != "Optional" for node in imported.nodes)


def test_import_if_without_parent_value_info_uses_branch_output_types() -> None:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])

    then_graph = helper.make_graph(
        [helper.make_node("Identity", inputs=["x"], outputs=["then_out"])],
        "then_graph",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", inputs=["y"], outputs=["else_out"])],
        "else_graph",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["branch_out"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    add_node = helper.make_node("Add", inputs=["branch_out", "x"], outputs=["out"])
    graph = helper.make_graph([if_node, add_node], "if_graph", [cond, x, y], [out])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)

    imported = import_onnx(model)

    assert [node.op_type for node in imported.nodes] == [
        "Identity",
        "Identity",
        "Where",
        "Add",
    ]


def test_import_gradient_of_add_expands_to_supported_nodes() -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2])
    dc_da = helper.make_tensor_value_info("dc_da", TensorProto.FLOAT, [2])
    dc_db = helper.make_tensor_value_info("dc_db", TensorProto.FLOAT, [2])

    add_node = helper.make_node("Add", inputs=["a", "b"], outputs=["c"])
    gradient_node = helper.make_node(
        "Gradient",
        inputs=["a", "b"],
        outputs=["dc_da", "dc_db"],
        domain="ai.onnx.preview.training",
        y="c",
        xs=["a", "b"],
    )
    graph = helper.make_graph(
        [add_node, gradient_node],
        "gradient_add_graph",
        [a, b],
        [c, dc_da, dc_db],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("ai.onnx.preview.training", 1),
        ],
    )
    model.ir_version = 7

    imported = import_onnx(model)

    assert all(node.op_type != "Gradient" for node in imported.nodes)
    assert imported.nodes[0].op_type == "Add"
    assert [node.op_type for node in imported.nodes[-2:]] == ["Identity", "Identity"]


def test_import_gradient_of_add_and_mul_keeps_forward_and_gradient_paths() -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2])
    d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [2])
    dd_da = helper.make_tensor_value_info("dd_da", TensorProto.FLOAT, [2])
    dd_db = helper.make_tensor_value_info("dd_db", TensorProto.FLOAT, [2])

    add_node = helper.make_node("Add", inputs=["a", "b"], outputs=["c"])
    mul_node = helper.make_node("Mul", inputs=["c", "a"], outputs=["d"])
    gradient_node = helper.make_node(
        "Gradient",
        inputs=["a", "b"],
        outputs=["dd_da", "dd_db"],
        domain="ai.onnx.preview.training",
        y="d",
        xs=["a", "b"],
    )
    graph = helper.make_graph(
        [add_node, mul_node, gradient_node],
        "gradient_add_mul_graph",
        [a, b],
        [d, dd_da, dd_db],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("ai.onnx.preview.training", 1),
        ],
    )
    model.ir_version = 7

    imported = import_onnx(model)

    op_types = [node.op_type for node in imported.nodes]
    assert "Gradient" not in op_types
    assert op_types[:2] == ["Add", "Mul"]
    assert op_types.count("Mul") >= 3
    assert op_types[-2:] == ["Identity", "Identity"]


def test_import_allows_anonymous_dynamic_dims() -> None:
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [2, 3])
    output_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    output_dim = output_info.type.tensor_type.shape.dim[0]
    output_dim.ClearField("dim_value")
    if output_dim.HasField("dim_param"):
        output_dim.ClearField("dim_param")

    node = helper.make_node("Shape", inputs=["in0"], outputs=["out"])
    graph = helper.make_graph(
        [node], "anonymous_dynamic_dim_graph", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7

    imported = import_onnx(model)
    output_type = imported.find_value("out").type

    assert output_type.shape == (1,)
    assert output_type.dim_params == ("out_dim_0",)


def test_import_optional_sequence_value_info_marks_sequence_optional() -> None:
    sequence_type = helper.make_sequence_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [2])
    )
    optional_sequence_type = helper.make_optional_type_proto(sequence_type)
    input_info = helper.make_value_info("opt_seq", optional_sequence_type)
    output_info = helper.make_tensor_value_info("out", TensorProto.BOOL, [])
    node = helper.make_node("OptionalHasElement", inputs=["opt_seq"], outputs=["out"])
    graph = helper.make_graph(
        [node], "optional_sequence_graph", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 8

    imported = import_onnx(model)

    assert imported.inputs[0].name == "opt_seq"
    assert imported.inputs[0].type.is_optional
