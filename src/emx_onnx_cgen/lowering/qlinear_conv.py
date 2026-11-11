from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import QLinearConvOp
from .common import _find_initializer
from .common import optional_name, value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .conv import resolve_conv_spec
from .registry import register_lowering


def _ensure_scalar_shape(shape: tuple[int, ...], label: str) -> None:
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearConv {label} must be a scalar, got shape {shape}"
        )


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not dtype.is_float:
        raise UnsupportedOpError(f"QLinearConv {label} must be float16/float/double")


def _resolve_weight_quant_shape(
    shape: tuple[int, ...], out_channels: int, label: str
) -> bool:
    if shape in {(), (1,)}:
        return False
    if shape == (out_channels,):
        return True
    raise UnsupportedOpError(
        f"QLinearConv {label} must be scalar or 1D per output channel, "
        f"got shape {shape}"
    )


def _fold_zero_points(
    graph: Graph,
    input_zero_name: str,
    weight_name: str,
    weight_zero_name: str,
    bias_name: str | None,
    out_channels: int,
    padded: bool,
) -> tuple[bool, tuple[int, ...] | None]:
    """(w_zero_point is all zero, folded bias) from constant initializers.

    sum((x - x_zp) * (w - w_zp)) = sum(x * (w - w_zp)) - x_zp * sum(w - w_zp). The second term is a
    per-channel constant only when every tap is accumulated, i.e. no padding tap is skipped, or when
    x_zp is 0.
    """
    w_zero = _find_initializer(graph, weight_zero_name)
    if w_zero is None:
        return False, None
    w_zp = np.broadcast_to(w_zero.data.reshape(-1), (out_channels,)).astype(np.int64)
    weight_zero_is_zero = not w_zp.any()
    x_zero = _find_initializer(graph, input_zero_name)
    weights = _find_initializer(graph, weight_name)
    bias = _find_initializer(graph, bias_name) if bias_name is not None else None
    if x_zero is None or weights is None or (bias_name is not None and bias is None):
        return weight_zero_is_zero, None
    x_zp = int(x_zero.data.reshape(-1)[0])
    if padded and x_zp != 0:
        return weight_zero_is_zero, None
    w = weights.data.astype(np.int64).reshape(out_channels, -1)
    folded = (
        bias.data.astype(np.int64)
        if bias is not None
        else np.zeros(out_channels, np.int64)
    ) - x_zp * (w - w_zp[:, None]).sum(axis=1)
    if np.abs(folded).max(initial=0) >= 2**31:
        return weight_zero_is_zero, None
    return weight_zero_is_zero, tuple(int(value) for value in folded)


@register_lowering("QLinearConv")
def lower_qlinear_conv(graph: Graph, node: Node) -> QLinearConvOp:
    if len(node.inputs) not in {8, 9} or len(node.outputs) != 1:
        raise UnsupportedOpError("QLinearConv must have 8 or 9 inputs and 1 output")
    input_name = node.inputs[0]
    input_scale_name = node.inputs[1]
    input_zero_name = node.inputs[2]
    weight_name = node.inputs[3]
    weight_scale_name = node.inputs[4]
    weight_zero_name = node.inputs[5]
    output_scale_name = node.inputs[6]
    output_zero_name = node.inputs[7]
    bias_name = optional_name(node.inputs, 8)

    input_dtype = _value_dtype(graph, input_name, node)
    weight_dtype = _value_dtype(graph, weight_name, node)
    try:
        output_dtype = _value_dtype(graph, node.outputs[0], node)
    except ShapeInferenceError:
        output_dtype = input_dtype
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearConv supports uint8/int8 inputs only")
    if weight_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearConv supports uint8/int8 weights only")
    if output_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearConv supports uint8/int8 outputs only")

    input_scale_dtype = _value_dtype(graph, input_scale_name, node)
    weight_scale_dtype = _value_dtype(graph, weight_scale_name, node)
    output_scale_dtype = _value_dtype(graph, output_scale_name, node)
    _ensure_scale_dtype(input_scale_dtype, "x_scale")
    _ensure_scale_dtype(weight_scale_dtype, "w_scale")
    _ensure_scale_dtype(output_scale_dtype, "y_scale")

    input_zero_dtype = _value_dtype(graph, input_zero_name, node)
    weight_zero_dtype = _value_dtype(graph, weight_zero_name, node)
    output_zero_dtype = _value_dtype(graph, output_zero_name, node)
    if input_zero_dtype != input_dtype:
        raise UnsupportedOpError(
            "QLinearConv x_zero_point dtype must match input dtype"
        )
    if weight_zero_dtype != weight_dtype:
        raise UnsupportedOpError(
            "QLinearConv w_zero_point dtype must match weight dtype"
        )
    if output_zero_dtype != output_dtype:
        raise UnsupportedOpError(
            "QLinearConv y_zero_point dtype must match output dtype"
        )

    input_scale_shape = _value_shape(graph, input_scale_name, node)
    _ensure_scalar_shape(input_scale_shape, "x_scale")
    output_scale_shape = _value_shape(graph, output_scale_name, node)
    _ensure_scalar_shape(output_scale_shape, "y_scale")
    input_zero_shape = _value_shape(graph, input_zero_name, node)
    _ensure_scalar_shape(input_zero_shape, "x_zero_point")
    output_zero_shape = _value_shape(graph, output_zero_name, node)
    _ensure_scalar_shape(output_zero_shape, "y_zero_point")

    spec = resolve_conv_spec(
        graph,
        node,
        input_name=input_name,
        weight_name=weight_name,
        bias_name=None,
        require_output_shape=False,
    )

    weight_scale_shape = _value_shape(graph, weight_scale_name, node)
    weight_scale_per_channel = _resolve_weight_quant_shape(
        weight_scale_shape, spec.out_channels, "w_scale"
    )
    weight_zero_shape = _value_shape(graph, weight_zero_name, node)
    weight_zero_per_channel = _resolve_weight_quant_shape(
        weight_zero_shape, spec.out_channels, "w_zero_point"
    )

    if bias_name is not None:
        bias_shape = _value_shape(graph, bias_name, node)
        if bias_shape != (spec.out_channels,):
            raise UnsupportedOpError(
                "QLinearConv bias must be 1D with out_channels elements, "
                f"got shape {bias_shape}"
            )
        if _value_dtype(graph, bias_name, node) != ScalarType.I32:
            raise UnsupportedOpError("QLinearConv bias must have int32 dtype")

    weight_zero_is_zero, folded_bias = _fold_zero_points(
        graph,
        input_zero_name,
        weight_name,
        weight_zero_name,
        bias_name,
        spec.out_channels,
        padded=any(spec.pads),
    )
    lowered = QLinearConvOp(
        input0=input_name,
        input_scale=input_scale_name,
        input_zero_point=input_zero_name,
        weights=weight_name,
        weight_scale=weight_scale_name,
        weight_zero_point=weight_zero_name,
        output_scale=output_scale_name,
        output_zero_point=output_zero_name,
        bias=bias_name,
        output=node.outputs[0],
        batch=spec.batch,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        spatial_rank=spec.spatial_rank,
        in_spatial=spec.in_spatial,
        out_spatial=spec.out_spatial,
        kernel_shape=spec.kernel_shape,
        strides=spec.strides,
        pads=spec.pads,
        dilations=spec.dilations,
        group=spec.group,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        dtype=output_dtype,
        input_scale_dtype=input_scale_dtype,
        weight_scale_dtype=weight_scale_dtype,
        output_scale_dtype=output_scale_dtype,
        input_scale_shape=input_scale_shape,
        weight_scale_shape=weight_scale_shape,
        output_scale_shape=output_scale_shape,
        input_zero_shape=input_zero_shape,
        weight_zero_shape=weight_zero_shape,
        output_zero_shape=output_zero_shape,
        weight_scale_per_channel=weight_scale_per_channel,
        weight_zero_per_channel=weight_zero_per_channel,
        weight_zero_is_zero=weight_zero_is_zero,
        folded_bias=folded_bias,
    )
    if isinstance(graph, GraphContext):
        graph.set_shape(
            node.outputs[0], (spec.batch, spec.out_channels, *spec.out_spatial)
        )
        graph.set_dtype(node.outputs[0], output_dtype)
    return lowered
