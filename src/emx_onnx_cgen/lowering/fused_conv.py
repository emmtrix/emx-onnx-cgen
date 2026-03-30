from __future__ import annotations

from ..ir.ops import FusedConvOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .conv import resolve_conv_spec
from .registry import register_lowering

_SUPPORTED_ACTIVATIONS = {"Relu", "HardSigmoid"}

_ACTIVATION_ATTR_KEYS = {"activation", "activation_params"}


@register_lowering("FusedConv")
def lower_fused_conv(graph: Graph, node: Node) -> FusedConvOp:
    """Lower com.microsoft::FusedConv to FusedConvOp."""
    if len(node.inputs) not in {2, 3, 4} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "FusedConv must have 2–4 inputs (X, W[, B[, Z]]) and 1 output"
        )
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "FusedConv supports float16, float, and double inputs only"
        )

    activation_raw = node.attrs.get("activation", b"")
    if isinstance(activation_raw, bytes):
        activation = activation_raw.decode("utf-8")
    elif isinstance(activation_raw, str):
        activation = activation_raw
    else:
        raise UnsupportedOpError(
            f"FusedConv 'activation' attribute must be a string, "
            f"got {type(activation_raw).__name__}"
        )

    node_label = f" (node '{node.name}')" if node.name else ""
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise UnsupportedOpError(
            f"FusedConv activation '{activation}' is not supported{node_label}; "
            f"supported: {sorted(_SUPPORTED_ACTIVATIONS)}"
        )

    activation_params_raw = node.attrs.get("activation_params", ())
    activation_params = tuple(float(v) for v in activation_params_raw)

    # resolve_conv_spec validates Conv-specific attributes; we strip FusedConv extras
    # by temporarily removing them from node.attrs view via a wrapper node.
    conv_node = _ConvAttrView(node)
    bias_name = node.inputs[2] if len(node.inputs) >= 3 else None
    z_name = node.inputs[3] if len(node.inputs) == 4 else None

    spec = resolve_conv_spec(
        graph,
        conv_node,
        input_name=node.inputs[0],
        weight_name=node.inputs[1],
        bias_name=bias_name,
    )

    return FusedConvOp(
        input0=node.inputs[0],
        weights=node.inputs[1],
        bias=bias_name,
        z_input=z_name,
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
        dtype=op_dtype,
        activation=activation,
        activation_params=activation_params,
    )


class _ConvAttrView:
    """Wraps a Node, hiding FusedConv-specific attributes from resolve_conv_spec."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self.inputs = node.inputs
        self.outputs = node.outputs
        self.op_type = node.op_type
        self.name = node.name
        self.attrs = {
            k: v
            for k, v in node.attrs.items()
            if k not in _ACTIVATION_ATTR_KEYS
        }
