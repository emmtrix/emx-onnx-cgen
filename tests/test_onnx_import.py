from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper
from shared.scalar_types import ScalarType

from emx_onnx_cgen.onnx_import import (
    import_onnx,
    _fold_constant_of_shape_inputs,
    _maybe_infer_shapes,
    prepare_onnx_model,
)
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


def test_import_if_with_sequence_construct_branches_keeps_if_node() -> None:
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
    assert op_types == ["If"]
    assert len(imported.initializers) == 0


def test_import_if_with_optional_sequence_branches_keeps_if_node() -> None:
    model = onnx.load("onnx-org/onnx/backend/test/data/node/test_if_opt/model.onnx")

    imported = import_onnx(model)

    assert any(node.op_type == "If" for node in imported.nodes)
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

    assert output_type.shape == (-1,)
    assert output_type.dim_params == ("out_dim_0",)


def test_import_named_dynamic_dims_use_explicit_unknown_shape() -> None:
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    output_dim = output_info.type.tensor_type.shape.dim[0]
    output_dim.ClearField("dim_value")
    output_dim.dim_param = "N"

    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node], "named_dynamic_dim_graph", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7

    imported = import_onnx(model)
    output_type = imported.find_value("y").type

    assert output_type.shape == (-1,)
    assert output_type.dim_params == ("N",)


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


def _make_flexattention_model(
    *,
    score_mod: onnx.GraphProto | None = None,
) -> onnx.ModelProto:
    shape = [1, 2, 3, 4]
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, shape)
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, shape)
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
    attrs: dict[str, object] = {}
    if score_mod is not None:
        attrs["score_mod"] = score_mod
    node = helper.make_node(
        "FlexAttention",
        inputs=["Q", "K", "V"],
        outputs=["Y"],
        domain="ai.onnx.preview",
        **attrs,
    )
    graph = helper.make_graph([node], "flexattention_graph", [q, k, v], [y])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[
            helper.make_operatorsetid("", 26),
            helper.make_operatorsetid("ai.onnx.preview", 1),
        ],
    )
    model.ir_version = 10
    return model


def test_import_flexattention_expands_to_primitive_ops() -> None:
    model = _make_flexattention_model()

    imported = import_onnx(model)

    op_types = [node.op_type for node in imported.nodes]
    assert "FlexAttention" not in op_types
    # Scaled dot-product attention decomposes into two MatMuls around a Softmax.
    assert op_types.count("MatMul") == 2
    assert "Softmax" in op_types


def test_import_flexattention_inlines_score_mod_subgraph() -> None:
    score_mod = helper.make_graph(
        [helper.make_node("Add", inputs=["scores", "scores"], outputs=["scores_out"])],
        "score_mod",
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2, 3, 3])],
        [helper.make_tensor_value_info("scores_out", TensorProto.FLOAT, [1, 2, 3, 3])],
    )
    model = _make_flexattention_model(score_mod=score_mod)

    imported = import_onnx(model)

    op_types = [node.op_type for node in imported.nodes]
    assert "FlexAttention" not in op_types
    # The score_mod body (an Add) is inlined into the expanded graph.
    assert "Add" in op_types


def _static_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def test_fold_constant_of_shape_resolves_computed_shape() -> None:
    # ConstantOfShape fed by Shape/Div/Concat: ONNX shape inference cannot fold
    # the Div, so the output stays dynamic until the constant folder runs.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 16])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    heads = helper.make_tensor("heads", TensorProto.INT64, [1], [4])
    nodes = [
        helper.make_node("Shape", ["x"], ["batch"], start=0, end=1),
        helper.make_node("Shape", ["x"], ["hidden"], start=1, end=2),
        helper.make_node("Div", ["hidden", "heads"], ["head_size"]),
        helper.make_node("Concat", ["batch", "head_size"], ["state_shape"], axis=0),
        helper.make_node("ConstantOfShape", ["state_shape"], ["state"]),
        helper.make_node("Identity", ["state"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "fold_graph", [x], [out], initializer=[heads])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 27)],
    )
    model.ir_version = 10

    model = _maybe_infer_shapes(model)
    model, folded = _fold_constant_of_shape_inputs(model)
    assert folded
    model = _maybe_infer_shapes(model)

    resolved = {value_info.name: value_info for value_info in model.graph.value_info}
    resolved.update({out.name: out for out in model.graph.output})
    assert _static_shape(resolved["state"]) == (4, 4)


def _make_scan_model(opset: int) -> onnx.ModelProto:
    body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "scan_in"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["scan_out"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_in", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2]),
        ],
    )
    scan = helper.make_node(
        "Scan",
        ["init_state", "x"],
        ["final_state", "y"],
        num_scan_inputs=1,
        body=body,
    )
    graph = helper.make_graph(
        [scan],
        "scan_graph",
        [
            helper.make_tensor_value_info("init_state", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 2]),
        ],
        [
            helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def test_scan_expansion_uses_input_style_slice_for_modern_opset() -> None:
    prepared = prepare_onnx_model(_make_scan_model(27))
    slices = [node for node in prepared.graph.node if node.op_type == "Slice"]
    squeezes = [node for node in prepared.graph.node if node.op_type == "Squeeze"]
    assert slices
    # Opset >= 10 Slice takes starts/ends/axes as inputs, not attributes.
    assert all(len(node.input) >= 3 for node in slices)
    assert all(not node.attribute for node in slices)
    # Opset >= 13 Squeeze takes axes as an input rather than an attribute.
    assert all(len(node.input) == 2 for node in squeezes)


def test_scan_expansion_uses_attribute_slice_for_legacy_opset() -> None:
    prepared = prepare_onnx_model(_make_scan_model(9))
    slices = [node for node in prepared.graph.node if node.op_type == "Slice"]
    assert slices
    # Opset 9 Slice still carries starts/ends/axes as attributes.
    assert all(len(node.input) == 1 for node in slices)
    assert all(node.attribute for node in slices)
