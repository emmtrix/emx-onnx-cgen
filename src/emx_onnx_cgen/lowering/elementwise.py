from __future__ import annotations

from shared.scalar_functions import ScalarFunction, ScalarFunctionError
from shared.scalar_types import ScalarType

from ..ir.op_base import BroadcastingOpBase
from ..ir.ops import BinaryOp, ClipOp, PowOp, UnaryOp
from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..lowering.common import node_dtype, optional_name, value_dtype, value_shape
from ..lowering.registry import register_lowering, register_lowering_if_missing
from ..ops import (
    BINARY_OP_TYPES,
    COMPARE_FUNCTIONS,
    UNARY_OP_TYPES,
    binary_op_symbol,
    unary_op_symbol,
    validate_unary_attrs,
)
from ..lowering.variadic import VARIADIC_OP_FUNCTIONS


@register_lowering("Clip")
def lower_clip(graph: Graph, node: Node) -> ClipOp:
    if not node.inputs or len(node.outputs) != 1:
        raise UnsupportedOpError("Clip must have 1 output")
    input_name = node.inputs[0]
    if not input_name:
        raise UnsupportedOpError("Clip input must be provided")
    min_name = optional_name(node.inputs, 1)
    max_name = optional_name(node.inputs, 2)
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Clip expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if min_name is not None:
        min_dtype = value_dtype(graph, min_name, node)
        if min_dtype != input_dtype:
            raise UnsupportedOpError(
                "Clip min dtype must match input dtype, "
                f"got {min_dtype.onnx_name}"
            )
    if max_name is not None:
        max_dtype = value_dtype(graph, max_name, node)
        if max_dtype != input_dtype:
            raise UnsupportedOpError(
                "Clip max dtype must match input dtype, "
                f"got {max_dtype.onnx_name}"
            )
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise UnsupportedOpError("Clip input and output shapes must match")
    min_shape = value_shape(graph, min_name, node) if min_name else None
    max_shape = value_shape(graph, max_name, node) if max_name else None
    return ClipOp(
        input0=input_name,
        input_min=min_name,
        input_max=max_name,
        output=node.outputs[0],
    )


@register_lowering("Celu")
def lower_celu(graph: Graph, node: Node) -> UnaryOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Celu must have 1 input and 1 output")
    dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not dtype.is_float:
        raise UnsupportedOpError("Celu only supports floating-point inputs")
    alpha = float(node.attrs.get("alpha", 1.0))
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=ScalarFunction.CELU,
        params=(alpha,),
    )


@register_lowering("Swish")
def lower_swish(graph: Graph, node: Node) -> UnaryOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Swish must have 1 input and 1 output")
    dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not dtype.is_float:
        raise UnsupportedOpError("Swish only supports floating-point inputs")
    alpha = float(node.attrs.get("alpha", 1.0))
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=ScalarFunction.SWISH,
        params=(alpha,),
    )


@register_lowering("Shrink")
def lower_shrink(graph: Graph, node: Node) -> UnaryOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Shrink must have 1 input and 1 output")
    dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not dtype.is_float:
        raise UnsupportedOpError("Shrink only supports floating-point inputs")
    bias = float(node.attrs.get("bias", 0.0))
    lambd = float(node.attrs.get("lambd", 0.5))
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=ScalarFunction.SHRINK,
        params=(bias, lambd),
    )


@register_lowering("Pow")
def lower_pow(graph: Graph, node: Node) -> PowOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Pow must have 2 inputs and 1 output")
    op_dtype = value_dtype(graph, node.inputs[0], node)
    op_spec = binary_op_symbol(ScalarFunction.POW, node.attrs, dtype=op_dtype)
    if op_spec is None:
        raise UnsupportedOpError("Unsupported op Pow")
    return PowOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
        function=ScalarFunction.POW,
        operator_kind=op_spec.kind,
    )


def _lower_binary_unary(graph: Graph | GraphContext, node: Node) -> BinaryOp | UnaryOp:
    if node.op_type == "BitShift":
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("BitShift must have 2 inputs and 1 output")
        direction_attr = node.attrs.get("direction", "LEFT")
        if isinstance(direction_attr, bytes):
            direction = direction_attr.decode()
        else:
            direction = str(direction_attr)
        if direction not in {"LEFT", "RIGHT"}:
            raise UnsupportedOpError(
                "BitShift direction must be LEFT or RIGHT"
            )
        op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
        if not op_dtype.is_integer:
            raise UnsupportedOpError("BitShift expects integer inputs")
        function = (
            ScalarFunction.BITWISE_LEFT_SHIFT
            if direction == "LEFT"
            else ScalarFunction.BITWISE_RIGHT_SHIFT
        )
        op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
        if op_spec is None:
            raise UnsupportedOpError("Unsupported op BitShift")
        input0_shape = value_shape(graph, node.inputs[0], node)
        input1_shape = value_shape(graph, node.inputs[1], node)
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
        )
    if node.op_type == "Mod":
        fmod = int(node.attrs.get("fmod", 0))
        if fmod not in {0, 1}:
            raise UnsupportedOpError("Mod only supports fmod=0 or fmod=1")
        function = (
            ScalarFunction.FMOD if fmod == 1 else ScalarFunction.REMAINDER
        )
    else:
        try:
            function = ScalarFunction.from_onnx_op(node.op_type)
        except ScalarFunctionError as exc:
            raise UnsupportedOpError(
                f"Unsupported op {node.op_type}"
            ) from exc
    validate_unary_attrs(node.op_type, node.attrs)
    if function in COMPARE_FUNCTIONS:
        input_dtype = node_dtype(graph, node, *node.inputs)
        output_dtype = value_dtype(graph, node.outputs[0], node)
        op_spec = binary_op_symbol(function, node.attrs, dtype=input_dtype)
        if op_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{node.op_type} expects bool output, got {output_dtype.onnx_name}"
            )
        input0_shape = value_shape(graph, node.inputs[0], node)
        input1_shape = value_shape(graph, node.inputs[1], node)
        output_shape = value_shape(graph, node.outputs[0], node)
        op = BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
        )
        if isinstance(graph, GraphContext):
            inferred_shape = BroadcastingOpBase.broadcast_shapes(
                input0_shape, input1_shape
            )
            graph.set_shape(node.outputs[0], inferred_shape)
        return op
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
    unary_symbol = unary_op_symbol(function, dtype=op_dtype)
    if op_spec is None and unary_symbol is None:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if op_spec is not None:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        input0_shape = value_shape(graph, node.inputs[0], node)
        input1_shape = value_shape(graph, node.inputs[1], node)
        output_shape = value_shape(graph, node.outputs[0], node)
        op = BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
        )
        if isinstance(graph, GraphContext):
            inferred_shape = BroadcastingOpBase.broadcast_shapes(
                input0_shape, input1_shape
            )
            graph.set_shape(node.outputs[0], inferred_shape)
        return op
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 input and 1 output"
        )
    output_shape = value_shape(graph, node.outputs[0], node)
    op = UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=function,
        params=(),
    )
    if isinstance(graph, GraphContext):
        inferred_shape = value_shape(graph, node.inputs[0], node)
        graph.set_shape(node.outputs[0], inferred_shape)
    return op


_DEFAULT_ELEMENTWISE_TYPES = (
    BINARY_OP_TYPES.union(UNARY_OP_TYPES) - set(VARIADIC_OP_FUNCTIONS.keys())
)

for _op_type in _DEFAULT_ELEMENTWISE_TYPES:
    register_lowering_if_missing(_op_type)(_lower_binary_unary)


@register_lowering("IsInf")
def lower_isinf(graph: Graph, node: Node) -> UnaryOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("IsInf must have 1 input and 1 output")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if not input_dtype.is_float:
        raise UnsupportedOpError("IsInf only supports floating-point inputs")
    if output_dtype != ScalarType.BOOL:
        raise UnsupportedOpError("IsInf output must be bool")
    detect_negative = int(node.attrs.get("detect_negative", 1))
    detect_positive = int(node.attrs.get("detect_positive", 1))
    if detect_negative not in {0, 1} or detect_positive not in {0, 1}:
        raise UnsupportedOpError(
            "IsInf detect_negative and detect_positive must be 0 or 1"
        )
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=ScalarFunction.ISINF,
        params=(float(detect_negative), float(detect_positive)),
    )


@register_lowering("IsNaN")
def lower_isnan(graph: Graph, node: Node) -> UnaryOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("IsNaN must have 1 input and 1 output")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if not input_dtype.is_float:
        raise UnsupportedOpError("IsNaN only supports floating-point inputs")
    if output_dtype != ScalarType.BOOL:
        raise UnsupportedOpError("IsNaN output must be bool")
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=ScalarFunction.ISNAN,
        params=(),
    )
