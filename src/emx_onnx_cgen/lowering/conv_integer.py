from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ConvIntegerOp
from .common import optional_name, value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .conv import resolve_conv_spec
from .registry import register_lowering


def _ensure_scalar_shape(shape: tuple[int, ...], label: str) -> None:
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"ConvInteger {label} must be a scalar, got shape {shape}"
        )


def _resolve_w_zero_point_shape(shape: tuple[int, ...], out_channels: int) -> bool:
    if shape in {(), (1,)}:
        return False
    if shape == (out_channels,):
        return True
    raise UnsupportedOpError(
        "ConvInteger w_zero_point must be scalar or 1D per output channel, "
        f"got shape {shape}"
    )


@register_lowering("ConvInteger")
def lower_conv_integer(graph: Graph, node: Node) -> ConvIntegerOp:
    if len(node.inputs) not in {2, 3, 4} or len(node.outputs) != 1:
        raise UnsupportedOpError("ConvInteger must have 2 to 4 inputs and 1 output")
    input_name = node.inputs[0]
    weight_name = node.inputs[1]
    x_zero_point_name = optional_name(node.inputs, 2)
    w_zero_point_name = optional_name(node.inputs, 3)
    input_dtype = _value_dtype(graph, input_name, node)
    weight_dtype = _value_dtype(graph, weight_name, node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("ConvInteger supports uint8/int8 inputs only")
    if weight_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("ConvInteger supports uint8/int8 weights only")
    if output_dtype != ScalarType.I32:
        raise UnsupportedOpError("ConvInteger expects int32 outputs only")
    x_zero_shape = None
    if x_zero_point_name is not None:
        x_zero_shape = _value_shape(graph, x_zero_point_name, node)
        _ensure_scalar_shape(x_zero_shape, "x_zero_point")
        if _value_dtype(graph, x_zero_point_name, node) != input_dtype:
            raise UnsupportedOpError(
                "ConvInteger x_zero_point dtype must match input dtype"
            )
    w_zero_shape = None
    w_zero_point_per_channel = False
    if w_zero_point_name is not None:
        w_zero_shape = _value_shape(graph, w_zero_point_name, node)
        if _value_dtype(graph, w_zero_point_name, node) != weight_dtype:
            raise UnsupportedOpError(
                "ConvInteger w_zero_point dtype must match weight dtype"
            )
    spec = resolve_conv_spec(
        graph,
        node,
        input_name=input_name,
        weight_name=weight_name,
        bias_name=None,
    )
    if w_zero_shape is not None:
        w_zero_point_per_channel = _resolve_w_zero_point_shape(
            w_zero_shape, spec.out_channels
        )
    return ConvIntegerOp(
        input0=input_name,
        weights=weight_name,
        x_zero_point=x_zero_point_name,
        w_zero_point=w_zero_point_name,
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
        x_zero_point_shape=x_zero_shape,
        w_zero_point_shape=w_zero_shape,
        w_zero_point_per_channel=w_zero_point_per_channel,
    )
