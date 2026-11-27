"""Named dynamic dimensions that travel through shape values (Qwen2.5 decoder).

Each model is the smallest graph that reproduces one way ONNX shape inference
loses an interface dim_param such as ``n + 1`` in the Qwen2.5-0.5B decoder
export (onnx-community/Qwen2.5-0.5B-Instruct) when ``past_sequence_length``
stays dynamic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pytest
import sympy
from onnx import TensorProto, helper, numpy_helper, shape_inference

from dynamic_dim_verify import verify_for_dims
from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.dim_params import parse_dim_param
from emx_onnx_cgen.symbolic_shapes import (
    _decide_min_max,
    _SymbolicEvaluator,
    infer_dim_param_shapes,
)

_INT64_MAX = 2**63 - 1


def _dims(value_info: onnx.ValueInfoProto) -> list[object]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _value_info(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    for value_info in (*model.graph.value_info, *model.graph.output):
        if value_info.name == name:
            return value_info
    raise KeyError(name)


def _is_unknown(dim: object, interface: set[str]) -> bool:
    return isinstance(dim, str) and dim not in interface


def _make_model(nodes, inputs, outputs, initializers) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, "dim_param_graph", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _make_range_over_mask_length_model() -> onnx.ModelProto:
    """``Range(0, Shape(mask)[0], 1)``: the causal-mask positions in Qwen.

    ONNX infers the Range length as a fresh unknown dimension, although it is
    exactly the mask length ``n + 1``.
    """

    mask = helper.make_tensor_value_info("mask", TensorProto.INT64, ["n + 1"])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n + 1"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n + 1"])
    initializers = [
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
    ]
    nodes = [
        helper.make_node("Shape", ["mask"], ["mask_shape"]),
        helper.make_node("Gather", ["mask_shape", "zero"], ["length"], axis=0),
        helper.make_node("Range", ["zero", "length", "one"], ["positions"]),
        helper.make_node("Cast", ["positions"], ["positions_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["x", "positions_f"], ["y"]),
    ]
    return _make_model(nodes, [mask, x], [y], initializers)


def _make_slice_to_shape_value_model() -> onnx.ModelProto:
    """``x[:, :Shape(x)[1]]`` followed by a Where, as in Qwen's attention mask.

    ONNX gives the Slice output and everything downstream unknown dimensions
    (even the static batch axis), and keeps them when it later runs again.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "n + 1"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "n + 1"])
    initializers = [
        numpy_helper.from_array(np.array(1, dtype=np.int64), "axis_index"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f"),
    ]
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "axis_index"], ["length"], axis=0),
        helper.make_node("Unsqueeze", ["length", "starts"], ["ends"]),
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["sliced"]),
        helper.make_node("Greater", ["sliced", "zero_f"], ["positive"]),
        helper.make_node("Where", ["positive", "sliced", "zero_f"], ["y"]),
    ]
    return _make_model(nodes, [x], [y], initializers)


def _make_repeat_kv_model() -> onnx.ModelProto:
    """``repeat_kv`` of the PyTorch export: Expand to ``Where(shape == -1, 1, shape)``.

    The Expand target is a guarded shape vector, so ONNX cannot infer the
    expanded KV tensor; with non-negative dims the guard is decidable.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1, "n", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, "n", 4])
    initializers = [
        numpy_helper.from_array(np.array(2, dtype=np.int64), "two"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "heads"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), "head_dim"),
        numpy_helper.from_array(np.array([-1] * 4, dtype=np.int64), "minus_ones"),
        numpy_helper.from_array(np.array([1] * 4, dtype=np.int64), "ones"),
    ]
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "two"], ["length"], axis=0),
        helper.make_node("Unsqueeze", ["length", "axis0"], ["length_1d"]),
        helper.make_node(
            "Concat", ["heads", "length_1d", "head_dim"], ["target_raw"], axis=0
        ),
        helper.make_node("Equal", ["target_raw", "minus_ones"], ["is_unset"]),
        helper.make_node("Where", ["is_unset", "ones", "target_raw"], ["target"]),
        helper.make_node("Expand", ["x", "target"], ["expanded"]),
        helper.make_node("Relu", ["expanded"], ["y"]),
    ]
    return _make_model(nodes, [x], [y], initializers)


def _make_table_prefix_model() -> onnx.ModelProto:
    """``table[:Shape(x)[0]]``: Qwen's rotary cos/sin cache for the positions.

    The prefix of a table with 8 rows has length ``Min(8, n)``; it is ``n``
    only for ``n <= 8``, which needs ``--input-dim n<=8``.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 4])
    table = np.arange(32, dtype=np.float32).reshape(8, 4)
    initializers = [
        numpy_helper.from_array(table, "table"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
    ]
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "zero"], ["length"], axis=0),
        helper.make_node("Unsqueeze", ["length", "axis0"], ["ends"]),
        helper.make_node("Slice", ["table", "axis0", "ends", "axis0"], ["prefix"]),
        helper.make_node("Add", ["x", "prefix"], ["y"]),
    ]
    return _make_model(nodes, [x], [y], initializers)


@dataclass(frozen=True)
class _Case:
    builder: Callable[[], onnx.ModelProto]
    tensor: str  # the internal tensor whose dim ONNX inference loses
    interface: str  # the dim_param it should get
    dim_sets: tuple[dict[str, int], ...]
    upper_bounds: dict[str, int] | None = None


_MODELS = {
    "range_over_mask_length": _Case(
        _make_range_over_mask_length_model,
        "positions",
        "n + 1",
        ({"n + 1": 3}, {"n + 1": 6}),
    ),
    "slice_to_shape_value": _Case(
        _make_slice_to_shape_value_model,
        "sliced",
        "n + 1",
        ({"n + 1": 3}, {"n + 1": 6}),
    ),
    "repeat_kv": _Case(_make_repeat_kv_model, "expanded", "n", ({"n": 2}, {"n": 5})),
    "table_prefix": _Case(
        _make_table_prefix_model, "prefix", "n", ({"n": 3}, {"n": 8}), {"n": 8}
    ),
}


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_onnx_inference_loses_the_interface_dim_param(case: str) -> None:
    # Precondition for the pass: plain ONNX inference leaves the dim unknown.
    spec = _MODELS[case]
    model = shape_inference.infer_shapes(spec.builder(), data_prop=True)
    dims = _dims(_value_info(model, spec.tensor))
    assert any(_is_unknown(dim, {spec.interface}) for dim in dims)


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_infer_dim_param_shapes_names_the_dimension(case: str) -> None:
    spec = _MODELS[case]
    model = shape_inference.infer_shapes(spec.builder(), data_prop=True)

    model, changed = infer_dim_param_shapes(model, upper_bounds=spec.upper_bounds)

    assert changed
    dims = _dims(_value_info(model, spec.tensor))
    assert spec.interface in dims
    assert not any(_is_unknown(dim, {spec.interface}) for dim in dims)


def test_table_prefix_needs_the_upper_bound() -> None:
    model = shape_inference.infer_shapes(_make_table_prefix_model(), data_prop=True)
    model, _ = infer_dim_param_shapes(model)
    # The pass may drop the stale shape entirely; it must not name it n.
    prefix_dims = [_dims(vi) for vi in model.graph.value_info if vi.name == "prefix"]
    assert not prefix_dims or "n" not in prefix_dims[0]
    with pytest.raises(Exception, match="requires static shapes"):
        Compiler(CompilerOptions()).compile(_make_table_prefix_model())


def test_downstream_shapes_are_recomputed_after_naming() -> None:
    model = shape_inference.infer_shapes(
        _make_slice_to_shape_value_model(), data_prop=True
    )
    model, _ = infer_dim_param_shapes(model)
    model = shape_inference.infer_shapes(model, data_prop=True)

    assert _dims(_value_info(model, "positive")) == [1, "n + 1"]


def test_unprovable_slice_length_stays_unknown() -> None:
    # x[:, :3] has length Min(3, n + 1): it must be named neither n + 1 nor 3.
    model = _make_slice_to_shape_value_model()
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([3], dtype=np.int64), "three")
    )
    for node in model.graph.node:
        if node.op_type == "Slice":
            node.input[2] = "three"
    model = shape_inference.infer_shapes(model, data_prop=True)

    model, _ = infer_dim_param_shapes(model)

    for value_info in model.graph.value_info:
        if value_info.name == "sliced":
            assert _dims(value_info)[1] not in ("n + 1", 3)


@pytest.mark.parametrize(
    ("expression", "bounds", "expected"),
    [
        ("Min(8, n)", {"n": 8}, "n"),
        ("Min(8, n)", {"n": 9}, "Min(8, n)"),
        ("Min(8, n)", {}, "Min(8, n)"),
        ("Max(0, n + 1)", {}, "n + 1"),
        ("Min(8, n + 1)", {"n": 7}, "n + 1"),
    ],
)
def test_decide_min_max(expression: str, bounds: dict, expected: str) -> None:
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    parsed = sympy.sympify(expression, locals={"n": n})
    decided = _decide_min_max(parsed, {n: bound for bound in bounds.values()})
    assert str(decided) == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, _INT64_MAX, "n + 1"),
        (0, -1, "n"),
        (1, _INT64_MAX, "n"),
        (-1, _INT64_MAX, "1"),
        (0, 3, None),
    ],
)
def test_symbolic_slice_length(start: int, end: int, expected: str | None) -> None:
    dim = parse_dim_param("n + 1")
    length = _SymbolicEvaluator._symbolic_slice_length(
        dim, sympy.Integer(start), sympy.Integer(end)
    )
    if expected is None:
        assert length.has(sympy.Min)
    else:
        assert sympy.expand(length - parse_dim_param(expected)) == 0


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_compile_keeps_dim_param_as_runtime_dimension(case: str) -> None:
    spec = _MODELS[case]
    options = CompilerOptions(dim_param_upper_bounds=spec.upper_bounds)
    generated = Compiler(options).compile(spec.builder())
    assert "unk__" not in generated


@pytest.mark.parametrize("case", sorted(_MODELS))
def test_verify_against_onnxruntime_for_two_lengths(case: str, tmp_path: Path) -> None:
    spec = _MODELS[case]
    extra_args = [
        f"--input-dim={name}<={bound}"
        for name, bound in (spec.upper_bounds or {}).items()
    ]
    verify_for_dims(spec.builder(), tmp_path, spec.dim_sets, extra_args)


def test_value_info_of_ops_without_schema_keeps_its_type() -> None:
    # ONNX cannot re-infer a custom op, so naming another dim must not drop
    # the custom output's value_info (and with it its element type).
    model = _make_range_over_mask_length_model()
    model.graph.node.append(
        helper.make_node("Opaque", ["x"], ["custom"], domain="test.custom")
    )
    model.opset_import.append(helper.make_operatorsetid("test.custom", 1))
    model.graph.value_info.append(
        helper.make_tensor_value_info("custom", TensorProto.FLOAT, ["unk__9", 4])
    )
    model = shape_inference.infer_shapes(model, data_prop=True)

    model, changed = infer_dim_param_shapes(model)

    assert changed
    custom = _value_info(model, "custom")
    assert custom.type.tensor_type.elem_type == TensorProto.FLOAT
    assert _dims(custom)[1] == 4
