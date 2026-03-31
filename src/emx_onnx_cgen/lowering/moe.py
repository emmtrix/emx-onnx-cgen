from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MoEOp
from .common import node_dtype, optional_name, value_shape
from .registry import register_lowering


@register_lowering("MoE")
def lower_moe(graph: Graph, node: Node) -> MoEOp:
    # Inputs:
    # 0: input          [batch, model_dim]
    # 1: router_probs   [batch, num_experts]
    # 2: fc1_experts_weights  [num_experts, model_dim, fc1_out_size]
    # 3: fc1_experts_bias     (optional) [num_experts, fc1_out_size]
    # 4: fc2_experts_weights  [num_experts, fc2_in_size, model_dim]
    # 5: fc2_experts_bias     (optional) [num_experts, model_dim]
    # 6-7: additional optional inputs (not used)

    if len(node.inputs) < 5:
        raise UnsupportedOpError("MoE expects at least 5 inputs")

    input_name = node.inputs[0]
    router_name = node.inputs[1]
    fc1_w_name = node.inputs[2]
    fc1_bias_name = optional_name(node.inputs, 3)
    fc2_w_name = node.inputs[4]
    fc2_bias_name = optional_name(node.inputs, 5)

    output_name = optional_name(node.outputs, 0)
    if output_name is None:
        raise UnsupportedOpError("MoE expects an output")

    op_dtype = node_dtype(graph, node, input_name, fc1_w_name, fc2_w_name, output_name)
    if not op_dtype.is_float:
        raise UnsupportedOpError("MoE supports float inputs only")

    # Read attributes
    k = int(node.attrs.get("k", 1))
    normalize_routing_weights = int(node.attrs.get("normalize_routing_weights", 0))
    activation_type = node.attrs.get("activation_type", b"swiglu")
    if isinstance(activation_type, bytes):
        activation_type = activation_type.decode("utf-8")
    if activation_type != "swiglu":
        raise UnsupportedOpError(
            f"MoE: only activation_type='swiglu' is supported, got '{activation_type}'"
        )
    swiglu_fusion = int(node.attrs.get("swiglu_fusion", 0))
    if swiglu_fusion != 1:
        raise UnsupportedOpError("MoE: only swiglu_fusion=1 is supported")
    activation_beta = float(node.attrs.get("activation_beta", 0.0))

    # Read shapes
    inp_shape = value_shape(graph, input_name, node)
    if len(inp_shape) != 2:
        raise UnsupportedOpError(f"MoE: input must be rank 2, got {inp_shape}")
    batch, model_dim = inp_shape

    router_shape = value_shape(graph, router_name, node)
    if len(router_shape) != 2 or router_shape[0] != batch:
        raise UnsupportedOpError(
            f"MoE: router_probs must be [batch, num_experts], got {router_shape}"
        )
    num_experts = router_shape[1]

    fc1_shape = value_shape(graph, fc1_w_name, node)
    if len(fc1_shape) != 3 or fc1_shape[0] != num_experts or fc1_shape[1] != model_dim:
        raise UnsupportedOpError(
            f"MoE: fc1_w must be [num_experts, model_dim, fc1_out_size], got {fc1_shape}"
        )
    fc1_out_size = fc1_shape[2]
    fc2_in_size = fc1_out_size // 2  # SwiGLU splits fc1_out into gate + value

    fc2_shape = value_shape(graph, fc2_w_name, node)
    if (
        len(fc2_shape) != 3
        or fc2_shape[0] != num_experts
        or fc2_shape[1] != fc2_in_size
    ):
        raise UnsupportedOpError(
            f"MoE: fc2_w must be [num_experts, fc2_in_size={fc2_in_size}, model_dim], got {fc2_shape}"
        )

    return MoEOp(
        input=input_name,
        router_probs=router_name,
        fc1_w=fc1_w_name,
        fc1_bias=fc1_bias_name,
        fc2_w=fc2_w_name,
        fc2_bias=fc2_bias_name,
        output=output_name,
        batch=batch,
        model_dim=model_dim,
        num_experts=num_experts,
        k=k,
        fc1_out_size=fc1_out_size,
        fc2_in_size=fc2_in_size,
        normalize_routing_weights=normalize_routing_weights,
        activation_beta=activation_beta,
        dtype=op_dtype,
    )
