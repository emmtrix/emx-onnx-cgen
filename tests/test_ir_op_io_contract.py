from __future__ import annotations

from pathlib import Path

from shared.scalar_functions import ScalarFunction

from emx_onnx_cgen.ir.ops.elementwise import BinaryOp, ClipOp, UnaryOp, VariadicOp
from emx_onnx_cgen.ir.ops.misc import GatherOp
from emx_onnx_cgen.ir.ops.reduce import ReduceOp, TopKOp
from emx_onnx_cgen.ops import OperatorKind


def test_unary_binary_variadic_io_contract() -> None:
    unary = UnaryOp(input0="x", output="y", function=ScalarFunction.TANH)
    binary = BinaryOp(
        input0="x",
        input1="w",
        output="y",
        function=ScalarFunction.ADD,
        operator_kind=OperatorKind.INFIX,
    )
    variadic = VariadicOp(
        op_type="Sum",
        inputs=("x", "y", "z"),
        output="out",
        function=ScalarFunction.ADD,
        operator_kind=OperatorKind.INFIX,
    )

    assert unary.input_names == ("x",)
    assert unary.output_names == ("y",)
    assert binary.input_names == ("x", "w")
    assert binary.output_names == ("y",)
    assert variadic.input_names == ("x", "y", "z")
    assert variadic.output_names == ("out",)


def test_reduce_and_gather_io_contract() -> None:
    reduce_op = ReduceOp(
        input0="x",
        output="y",
        axes=(1,),
        axes_input="axes",
        keepdims=True,
        noop_with_empty_axes=False,
        reduce_kind="mean",
        reduce_count=None,
    )
    gather = GatherOp(data="data", indices="indices", output="out", axis=0)

    assert reduce_op.input_names == ("x", "axes")
    assert reduce_op.output_names == ("y",)
    assert gather.input_names == ("data", "indices")
    assert gather.output_names == ("out",)


def test_optional_and_multi_output_io_contract() -> None:
    clip = ClipOp(
        input0="x",
        input_min=None,
        input_max="max",
        output="out",
    )
    topk = TopKOp(
        input0="x",
        k_input="k",
        output_values="values",
        output_indices="indices",
        axis=1,
        k=2,
        largest=True,
        sorted=True,
    )

    assert clip.input_names == ("x", "max")
    assert clip.output_names == ("out",)
    assert topk.input_names == ("x", "k")
    assert topk.output_names == ("values", "indices")


def test_codegen_name_derivation_uses_canonical_io_contract() -> None:
    emitter_source = Path("src/emx_onnx_cgen/codegen/c_emitter.py").read_text(
        encoding="utf-8"
    )
    start = emitter_source.index("def _op_names")
    end = emitter_source.index("def _build_name_map", start)
    op_names_body = emitter_source[start:end]

    assert "op.input_names" in op_names_body
    assert "op.output_names" in op_names_body
    assert "isinstance(" not in op_names_body
