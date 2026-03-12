"""Tests for the central invariant gate checks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from shared.scalar_types import ScalarType

from emx_onnx_cgen.errors import ShapeInferenceError, UnsupportedOpError
from emx_onnx_cgen.invariants import (
    check_graph_integrity,
    check_inferred_shapes,
    check_inferred_types,
    check_lowered_ops,
)
from emx_onnx_cgen.ir.context import GraphContext
from emx_onnx_cgen.ir.model import Graph, Node, TensorType, Value
from emx_onnx_cgen.ir.op_base import EmitContext, Emitter, RenderableOpBase
from emx_onnx_cgen.ir.op_context import OpContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_type(
    dtype: ScalarType = ScalarType.F32,
    shape: tuple[int, ...] = (2, 3),
    dim_params: tuple[str | None, ...] | None = None,
) -> TensorType:
    if dim_params is None:
        dim_params = (None,) * len(shape)
    return TensorType(
        dtype=dtype,
        shape=shape,
        dim_params=dim_params,
    )


def _value(
    name: str,
    shape: tuple[int, ...] = (2, 3),
    dim_params: tuple[str | None, ...] | None = None,
) -> Value:
    return Value(name=name, type=_tensor_type(shape=shape, dim_params=dim_params))


def _node(
    op_type: str = "Relu",
    inputs: tuple[str, ...] = ("x",),
    outputs: tuple[str, ...] = ("y",),
) -> Node:
    return Node(op_type=op_type, name=None, inputs=inputs, outputs=outputs, attrs={})


def _simple_graph(
    *,
    inputs: tuple[Value, ...] | None = None,
    outputs: tuple[Value, ...] | None = None,
    nodes: tuple[Node, ...] | None = None,
    values: tuple[Value, ...] = (),
) -> Graph:
    if inputs is None:
        inputs = (_value("x"),)
    if outputs is None:
        outputs = (_value("y"),)
    if nodes is None:
        nodes = (_node(),)
    return Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=(),
        values=values,
    )


@dataclass(frozen=True)
class _DummyOp(RenderableOpBase):
    """Minimal op for testing with a single input/output."""

    __io_inputs__ = ("input",)
    __io_outputs__ = ("output",)

    input: str
    output: str
    dtype: ScalarType = ScalarType.F32

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return ""


# ---------------------------------------------------------------------------
# check_graph_integrity
# ---------------------------------------------------------------------------


class TestCheckGraphIntegrity:
    def test_valid_graph_passes(self) -> None:
        graph = _simple_graph()
        check_graph_integrity(graph)

    def test_no_outputs_raises(self) -> None:
        graph = _simple_graph(outputs=())
        with pytest.raises(UnsupportedOpError, match="at least one output"):
            check_graph_integrity(graph)

    def test_no_nodes_raises(self) -> None:
        graph = _simple_graph(nodes=())
        with pytest.raises(UnsupportedOpError, match="at least one node"):
            check_graph_integrity(graph)

    def test_negative_dim_in_input_raises(self) -> None:
        graph = _simple_graph(inputs=(_value("x", shape=(2, -1)),))
        with pytest.raises(ShapeInferenceError, match="Negative dimension"):
            check_graph_integrity(graph)

    def test_negative_dim_in_output_raises(self) -> None:
        graph = _simple_graph(outputs=(_value("y", shape=(-1, 3)),))
        with pytest.raises(ShapeInferenceError, match="Negative dimension"):
            check_graph_integrity(graph)

    def test_negative_dim_in_values_raises(self) -> None:
        graph = _simple_graph(values=(_value("v", shape=(4, -2)),))
        with pytest.raises(ShapeInferenceError, match="Negative dimension"):
            check_graph_integrity(graph)

    def test_negative_dim_with_dim_param_passes(self) -> None:
        graph = _simple_graph(
            inputs=(_value("x", shape=(2, -1), dim_params=(None, "N")),)
        )
        check_graph_integrity(graph)

    def test_zero_dim_passes(self) -> None:
        graph = _simple_graph(outputs=(_value("y", shape=(0, 3)),))
        check_graph_integrity(graph)

    def test_scalar_passes(self) -> None:
        graph = _simple_graph(outputs=(_value("y", shape=()),))
        check_graph_integrity(graph)


# ---------------------------------------------------------------------------
# check_lowered_ops
# ---------------------------------------------------------------------------


class TestCheckLoweredOps:
    def test_valid_ops_pass(self) -> None:
        ops = [_DummyOp(input="x", output="y")]
        check_lowered_ops(ops)

    def test_empty_output_name_raises(self) -> None:
        ops = [_DummyOp(input="x", output="")]
        with pytest.raises(UnsupportedOpError, match="empty output name"):
            check_lowered_ops(ops)

    def test_multiple_valid_ops_pass(self) -> None:
        ops = [
            _DummyOp(input="x", output="y"),
            _DummyOp(input="y", output="z"),
        ]
        check_lowered_ops(ops)


# ---------------------------------------------------------------------------
# check_inferred_types
# ---------------------------------------------------------------------------


class TestCheckInferredTypes:
    def test_resolved_dtype_passes(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        ops = [_DummyOp(input="x", output="y")]
        check_inferred_types(ops, op_ctx)

    def test_missing_dtype_raises(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        ops = [_DummyOp(input="x", output="nonexistent")]
        with pytest.raises(ShapeInferenceError, match="Missing dtype"):
            check_inferred_types(ops, op_ctx)


# ---------------------------------------------------------------------------
# check_inferred_shapes
# ---------------------------------------------------------------------------


class TestCheckInferredShapes:
    def test_valid_shapes_pass(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        ops = [_DummyOp(input="x", output="y")]
        check_inferred_shapes(ops, op_ctx)

    def test_negative_dim_raises(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        op_ctx.set_shape("y", (2, -1))
        ops = [_DummyOp(input="x", output="y")]
        with pytest.raises(ShapeInferenceError, match="negative dimension"):
            check_inferred_shapes(ops, op_ctx)

    def test_negative_dim_with_dim_param_passes(self) -> None:
        graph = _simple_graph(outputs=(_value("y", shape=(2, -1), dim_params=(None, "N")),))
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        op_ctx.set_shape("y", (2, -1))
        ops = [_DummyOp(input="x", output="y")]
        check_inferred_shapes(ops, op_ctx)

    def test_zero_dim_passes(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        op_ctx.set_shape("y", (2, 0))
        ops = [_DummyOp(input="x", output="y")]
        check_inferred_shapes(ops, op_ctx)

    def test_scalar_output_passes(self) -> None:
        graph = _simple_graph(outputs=(_value("y", shape=()),))
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        ops = [_DummyOp(input="x", output="y")]
        check_inferred_shapes(ops, op_ctx)

    def test_missing_shape_raises(self) -> None:
        graph = _simple_graph()
        ctx = GraphContext(graph)
        op_ctx = OpContext(ctx)
        ops = [_DummyOp(input="x", output="nonexistent")]
        with pytest.raises(ShapeInferenceError, match="Missing shape"):
            check_inferred_shapes(ops, op_ctx)
