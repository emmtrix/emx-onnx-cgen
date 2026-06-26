from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from emx_onnx_cgen.input_dim_overrides import (
    apply_input_dim_overrides,
    collect_dynamic_input_dims,
    parse_input_dim_overrides,
)


def _relu_model(*, input_dims: list, output_dims: list) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_dims)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_dims)
    node = helper.make_node("Relu", ["x"], ["out"])
    graph = helper.make_graph([node], "g", [x], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _input_dims(model: onnx.ModelProto, name: str) -> list:
    for value_info in model.graph.input:
        if value_info.name == name:
            return [
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                for dim in value_info.type.tensor_type.shape.dim
            ]
    raise AssertionError(name)


def _output_dims(model: onnx.ModelProto, name: str) -> list:
    for value_info in model.graph.output:
        if value_info.name == name:
            return [
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                for dim in value_info.type.tensor_type.shape.dim
            ]
    raise AssertionError(name)


def test_parse_param_and_position_forms() -> None:
    overrides = parse_input_dim_overrides(["batch=4", "images:0=1"])
    assert overrides.by_param == {"batch": 4}
    assert overrides.by_position == {("images", 0): 1}


@pytest.mark.parametrize(
    "spec",
    ["batch", "batch=0", "batch=-1", "batch=x", ":0=1", "1bad=2"],
)
def test_parse_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_input_dim_overrides([spec])


def test_parse_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        parse_input_dim_overrides(["batch=1", "batch=2"])
    with pytest.raises(ValueError):
        parse_input_dim_overrides(["x:0=1", "x:0=2"])


def test_collect_dynamic_input_dims_reports_param_and_unknown() -> None:
    model = _relu_model(input_dims=["batch", 3, None], output_dims=["batch", 3, None])
    dims = collect_dynamic_input_dims(model)
    assert [(d.input_name, d.axis, d.dim_param) for d in dims] == [
        ("x", 0, "batch"),
        ("x", 2, None),
    ]


def test_collect_dynamic_input_dims_skips_initializers() -> None:
    model = _relu_model(input_dims=["batch", 2], output_dims=["batch", 2])
    weight = helper.make_tensor("x", TensorProto.FLOAT, [2, 2], [0.0] * 4)
    model.graph.initializer.append(weight)
    assert collect_dynamic_input_dims(model) == []


def test_apply_param_override_fixes_inputs_and_outputs() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    applied = apply_input_dim_overrides(model, parse_input_dim_overrides(["batch=8"]))
    assert [a.format() for a in applied] == ["x[0]=8"]
    assert _input_dims(model, "x") == [8, 3]
    assert _output_dims(model, "out") == [8, 3]


def test_apply_position_override_pins_param_graph_wide() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    apply_input_dim_overrides(model, parse_input_dim_overrides(["x:0=2"]))
    # The axis carries dim_param "batch", so the output is pinned too.
    assert _input_dims(model, "x") == [2, 3]
    assert _output_dims(model, "out") == [2, 3]


def test_apply_position_override_fixes_unknown_axis_only() -> None:
    model = _relu_model(input_dims=[None, 3], output_dims=[None, 3])
    apply_input_dim_overrides(model, parse_input_dim_overrides(["x:0=5"]))
    assert _input_dims(model, "x") == [5, 3]
    # The unknown output axis is left dynamic (no shared dim_param).
    assert _output_dims(model, "out")[1] == 3


def test_apply_rejects_unknown_param() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    with pytest.raises(ValueError, match="no dynamic input dimension"):
        apply_input_dim_overrides(model, parse_input_dim_overrides(["seq=4"]))


def test_apply_rejects_unknown_input_name() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    with pytest.raises(ValueError, match="no model input named"):
        apply_input_dim_overrides(model, parse_input_dim_overrides(["missing:0=4"]))


def test_apply_rejects_static_axis() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    with pytest.raises(ValueError, match="not a dynamic dimension"):
        apply_input_dim_overrides(model, parse_input_dim_overrides(["x:1=4"]))


def test_apply_clears_stale_dynamic_value_info() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    # An intermediate left dynamic by some unrelated axis must be dropped so
    # shape inference recomputes it from the now-static inputs.
    dynamic_vi = helper.make_tensor_value_info(
        "intermediate", TensorProto.FLOAT, ["other", 3]
    )
    static_vi = helper.make_tensor_value_info("fixed", TensorProto.FLOAT, [2, 3])
    model.graph.value_info.extend([dynamic_vi, static_vi])
    apply_input_dim_overrides(model, parse_input_dim_overrides(["batch=4"]))
    remaining = {vi.name for vi in model.graph.value_info}
    assert "intermediate" not in remaining
    assert "fixed" in remaining


def test_apply_noop_without_overrides() -> None:
    model = _relu_model(input_dims=["batch", 3], output_dims=["batch", 3])
    assert apply_input_dim_overrides(model, parse_input_dim_overrides([])) == []
    assert _input_dims(model, "x") == ["batch", 3]
