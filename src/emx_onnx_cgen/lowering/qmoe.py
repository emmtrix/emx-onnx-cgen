from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import QMoEOp
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering


@register_lowering("QMoE")
def lower_qmoe(graph: Graph, node: Node) -> QMoEOp:
    # Positional inputs (empty string means absent):
    # 0:  input               [batch, model_dim]         float
    # 1:  router_probs        [batch, num_experts]        float
    # 2:  fc1_experts_weights [num_experts, fc1_out, model_dim]  int8/uint8
    # 3:  fc1_scales          [num_experts, fc1_out]      float
    # 4:  fc1_experts_bias    (optional) [num_experts, fc1_out]  float
    # 5:  fc2_experts_weights [num_experts, fc2_in, model_dim]   int8/uint8
    # 6:  fc2_scales          [num_experts, fc2_in]       float
    # 7:  fc2_experts_bias    (optional) [num_experts, model_dim] float
    # 8-13: reserved optional inputs (not supported)
    # 14: router_weights      (optional) [batch, num_experts]    float

    if len(node.inputs) < 6:
        raise UnsupportedOpError("QMoE expects at least 6 inputs")

    input_name = node.inputs[0]
    router_name = node.inputs[1]
    fc1_w_name = node.inputs[2]
    fc1_scales_name = node.inputs[3]
    fc1_bias_name = optional_name(node.inputs, 4)
    fc2_w_name = node.inputs[5]
    fc2_scales_name = optional_name(node.inputs, 6)
    fc2_bias_name = optional_name(node.inputs, 7)
    router_weights_name = optional_name(node.inputs, 14)

    if not fc2_scales_name:
        raise UnsupportedOpError("QMoE: fc2_scales (input 6) is required")

    output_name = optional_name(node.outputs, 0)
    if output_name is None:
        raise UnsupportedOpError("QMoE expects an output")

    # Validate weight dtype
    w_dtype = value_dtype(graph, fc1_w_name, node)
    if w_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            f"QMoE: fc1_experts_weights must be int8/uint8, got {w_dtype.onnx_name}"
        )
    fc2_w_dtype = value_dtype(graph, fc2_w_name, node)
    if fc2_w_dtype != w_dtype:
        raise UnsupportedOpError(
            "QMoE: fc1 and fc2 weight dtypes must match, "
            f"got {w_dtype.onnx_name} vs {fc2_w_dtype.onnx_name}"
        )

    # Validate float dtype (input / output / scales)
    op_dtype = value_dtype(graph, input_name, node)
    if not op_dtype.is_float:
        raise UnsupportedOpError(f"QMoE: input must be float, got {op_dtype.onnx_name}")
    scale_dtype = value_dtype(graph, fc1_scales_name, node)
    if not scale_dtype.is_float:
        raise UnsupportedOpError(
            f"QMoE: fc1_scales must be float, got {scale_dtype.onnx_name}"
        )

    # Read attributes
    expert_weight_bits = int(node.attrs.get("expert_weight_bits", 8))
    if expert_weight_bits != 8:
        raise UnsupportedOpError(
            f"QMoE: only expert_weight_bits=8 is supported, got {expert_weight_bits}"
        )
    k = int(node.attrs.get("k", 1))
    normalize_routing_weights = int(node.attrs.get("normalize_routing_weights", 0))
    activation_type = node.attrs.get("activation_type", b"swiglu")
    if isinstance(activation_type, bytes):
        activation_type = activation_type.decode("utf-8")
    if activation_type != "swiglu":
        raise UnsupportedOpError(
            f"QMoE: only activation_type='swiglu' is supported, got '{activation_type}'"
        )
    swiglu_fusion = int(node.attrs.get("swiglu_fusion", 0))
    if swiglu_fusion != 1:
        raise UnsupportedOpError("QMoE: only swiglu_fusion=1 is supported")
    activation_beta = float(node.attrs.get("activation_beta", 0.0))

    # Resolve shapes
    inp_shape = value_shape(graph, input_name, node)
    if len(inp_shape) != 2:
        raise UnsupportedOpError(f"QMoE: input must be rank 2, got {inp_shape}")
    batch, model_dim = inp_shape

    router_shape = value_shape(graph, router_name, node)
    if len(router_shape) != 2 or router_shape[0] != batch:
        raise UnsupportedOpError(
            f"QMoE: router_probs must be [batch, num_experts], got {router_shape}"
        )
    num_experts = router_shape[1]

    fc1_shape = value_shape(graph, fc1_w_name, node)
    if len(fc1_shape) != 3 or fc1_shape[0] != num_experts or fc1_shape[2] != model_dim:
        raise UnsupportedOpError(
            "QMoE: fc1_experts_weights must be [num_experts, fc1_out_size, model_dim],"
            f" got {fc1_shape}"
        )
    fc1_out_size = fc1_shape[1]
    fc2_in_size = fc1_out_size // 2  # SwiGLU splits fc1 output into gate + value

    fc2_shape = value_shape(graph, fc2_w_name, node)
    if (
        len(fc2_shape) != 3
        or fc2_shape[0] != num_experts
        or fc2_shape[1] != fc2_in_size
        or fc2_shape[2] != model_dim
    ):
        raise UnsupportedOpError(
            "QMoE: fc2_experts_weights must be"
            f" [num_experts, fc2_in_size={fc2_in_size}, model_dim], got {fc2_shape}"
        )

    return QMoEOp(
        input=input_name,
        router_probs=router_name,
        fc1_w=fc1_w_name,
        fc1_scales=fc1_scales_name,
        fc1_bias=fc1_bias_name,
        fc2_w=fc2_w_name,
        fc2_scales=fc2_scales_name,
        fc2_bias=fc2_bias_name,
        router_weights=router_weights_name,
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
        weight_dtype=w_dtype,
        scale_dtype=scale_dtype,
    )
