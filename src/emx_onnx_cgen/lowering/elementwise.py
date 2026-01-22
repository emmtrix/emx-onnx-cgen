from __future__ import annotations

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ..codegen.c_emitter import ClipOp, UnaryOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import node_dtype, optional_name, value_dtype, value_shape
from ..lowering.registry import register_lowering


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
        input_shape=input_shape,
        min_shape=min_shape,
        max_shape=max_shape,
        output_shape=output_shape,
        dtype=input_dtype,
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
        shape=output_shape,
        dtype=dtype,
        input_dtype=dtype,
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
        shape=output_shape,
        dtype=dtype,
        input_dtype=dtype,
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
        shape=output_shape,
        dtype=dtype,
        input_dtype=dtype,
        params=(bias, lambd),
    )


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
        shape=output_shape,
        dtype=output_dtype,
        input_dtype=input_dtype,
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
        shape=output_shape,
        dtype=output_dtype,
        input_dtype=input_dtype,
        params=(),
    )
