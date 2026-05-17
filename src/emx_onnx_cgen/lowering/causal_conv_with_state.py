from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import CausalConvWithStateOp
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering

_SUPPORTED_ATTRS = {"activation", "ndim"}
_SUPPORTED_ACTIVATIONS = {"none", "silu", "swish"}


def _parse_activation(node: Node) -> str:
    activation_raw = node.attrs.get("activation", b"none")
    if isinstance(activation_raw, bytes):
        activation = activation_raw.decode("utf-8", errors="ignore")
    elif isinstance(activation_raw, str):
        activation = activation_raw
    else:
        raise UnsupportedOpError(
            "CausalConvWithState activation must be a string attribute"
        )
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise UnsupportedOpError(
            "CausalConvWithState activation must be one of "
            f"{sorted(_SUPPORTED_ACTIVATIONS)}, got {activation!r}"
        )
    return "silu" if activation == "swish" else activation


@register_lowering("CausalConvWithState")
def lower_causal_conv_with_state(
    graph: Graph | GraphContext, node: Node
) -> CausalConvWithStateOp:
    if len(node.inputs) != 4 or len(node.outputs) != 2:
        raise UnsupportedOpError("CausalConvWithState must have 4 inputs and 2 outputs")
    unsupported_attrs = set(node.attrs) - _SUPPORTED_ATTRS
    if unsupported_attrs:
        raise UnsupportedOpError(
            "CausalConvWithState has unsupported attributes: "
            f"{sorted(unsupported_attrs)}"
        )

    ndim = int(node.attrs.get("ndim", 1))
    if ndim != 1:
        raise UnsupportedOpError("CausalConvWithState currently supports ndim=1 only")

    bias_name = node.inputs[2] or None
    past_state_name = node.inputs[3] or None

    op_dtype = _node_dtype(
        graph,
        node,
        *(name for name in (*node.inputs, *node.outputs) if name),
    )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "CausalConvWithState supports float16, bfloat16, float, and double only"
        )

    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    if len(input_shape) != 3:
        raise UnsupportedOpError(
            f"CausalConvWithState input must have shape (B, C, L), got {input_shape}"
        )
    if len(weight_shape) != 3:
        raise UnsupportedOpError(
            f"CausalConvWithState weight must have shape (C, 1, K), got {weight_shape}"
        )

    batch, channels, seq_len = input_shape
    weight_channels, group_channels, kernel_size = weight_shape
    if weight_channels != channels:
        raise ShapeInferenceError(
            "CausalConvWithState weight channels must match input channels, "
            f"got {weight_channels} and {channels}"
        )
    if group_channels != 1:
        raise UnsupportedOpError(
            "CausalConvWithState requires depthwise weights with shape (C, 1, K)"
        )
    if kernel_size <= 0:
        raise UnsupportedOpError("CausalConvWithState kernel size must be positive")

    pad = kernel_size - 1
    if bias_name is not None:
        bias_shape = _value_shape(graph, bias_name, node)
        if bias_shape != (channels,):
            raise ShapeInferenceError(
                "CausalConvWithState bias must have shape "
                f"{(channels,)}, got {bias_shape}"
            )
    if past_state_name is not None:
        past_shape = _value_shape(graph, past_state_name, node)
        expected_past_shape = (batch, channels, pad)
        if past_shape != expected_past_shape:
            raise ShapeInferenceError(
                "CausalConvWithState past_state must have shape "
                f"{expected_past_shape}, got {past_shape}"
            )

    expected_output_shape = input_shape
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "CausalConvWithState output must have shape "
            f"{expected_output_shape}, got {output_shape}"
        )

    expected_state_shape = (batch, channels, pad)
    present_state_shape = _value_shape(graph, node.outputs[1], node)
    if present_state_shape != expected_state_shape:
        raise ShapeInferenceError(
            "CausalConvWithState present_state must have shape "
            f"{expected_state_shape}, got {present_state_shape}"
        )

    activation = _parse_activation(node)
    lowered = CausalConvWithStateOp(
        input0=node.inputs[0],
        weights=node.inputs[1],
        bias=bias_name,
        past_state=past_state_name,
        output=node.outputs[0],
        present_state=node.outputs[1],
        batch=batch,
        channels=channels,
        seq_len=seq_len,
        kernel_size=kernel_size,
        pad=pad,
        dtype=op_dtype,
        activation=activation,
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], expected_output_shape)
        graph.set_shape(node.outputs[1], expected_state_shape)
        graph.set_dtype(node.outputs[0], op_dtype)
        graph.set_dtype(node.outputs[1], op_dtype)

    return lowered
