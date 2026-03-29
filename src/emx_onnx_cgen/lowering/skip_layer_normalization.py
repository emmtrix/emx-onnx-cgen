from __future__ import annotations

from ..ir.ops import SkipLayerNormalizationOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype, optional_name, shape_product, value_dtype, value_shape
from .registry import register_lowering


def _validate_broadcast(
    name: str,
    input_shape: tuple[int, ...],
    tensor_shape: tuple[int, ...],
) -> None:
    """Validate that tensor_shape is broadcast-compatible with input_shape."""
    if len(tensor_shape) > len(input_shape):
        raise ShapeInferenceError(
            f"SkipLayerNormalization {name} rank {len(tensor_shape)} exceeds "
            f"input rank {len(input_shape)}"
        )
    offset = len(input_shape) - len(tensor_shape)
    for i, (ts_dim, in_dim) in enumerate(
        zip(tensor_shape, input_shape[offset:]), start=offset
    ):
        if ts_dim not in {1, in_dim}:
            raise ShapeInferenceError(
                f"SkipLayerNormalization {name} dimension {i} size {ts_dim} "
                f"is not broadcastable to {in_dim}"
            )


@register_lowering("SkipLayerNormalization")
def lower_skip_layer_normalization(
    graph: Graph, node: Node
) -> SkipLayerNormalizationOp:
    if len(node.inputs) < 3:
        raise UnsupportedOpError(
            "SkipLayerNormalization requires at least 3 inputs (input, skip, gamma)"
        )
    if len(node.outputs) < 1:
        raise UnsupportedOpError("SkipLayerNormalization requires at least 1 output")

    input_name = node.inputs[0]
    skip_name = node.inputs[1]
    gamma_name = node.inputs[2]
    beta_name = optional_name(node.inputs, 3)
    bias_name = optional_name(node.inputs, 4)

    # Outputs: output (0), mean (1, ignored), inv_std (2, ignored),
    #          skip_input_bias_add_output (3, optional)
    output_name = node.outputs[0]
    skip_input_bias_add_name = optional_name(node.outputs, 3)

    relevant_inputs = [n for n in [input_name, skip_name, gamma_name] if n]
    relevant_outputs = [n for n in [output_name] if n]
    op_dtype = node_dtype(graph, node, *relevant_inputs, *relevant_outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "SkipLayerNormalization supports float16, float, and double inputs only"
        )

    input_shape = value_shape(graph, input_name, node)
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "SkipLayerNormalization input must have at least 2 dimensions"
        )

    output_shape = value_shape(graph, output_name, node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            f"SkipLayerNormalization output shape {output_shape} must match "
            f"input shape {input_shape}"
        )

    hidden_size = input_shape[-1]

    skip_shape = value_shape(graph, skip_name, node)
    _validate_broadcast("skip", input_shape, skip_shape)

    gamma_shape = value_shape(graph, gamma_name, node)
    if gamma_shape != (hidden_size,):
        raise ShapeInferenceError(
            f"SkipLayerNormalization gamma shape {gamma_shape} must be ({hidden_size},)"
        )

    beta_shape: tuple[int, ...] | None = None
    if beta_name is not None:
        beta_dtype = value_dtype(graph, beta_name, node)
        if beta_dtype != op_dtype:
            raise UnsupportedOpError(
                "SkipLayerNormalization beta dtype must match input dtype"
            )
        beta_shape = value_shape(graph, beta_name, node)
        if beta_shape != (hidden_size,):
            raise ShapeInferenceError(
                f"SkipLayerNormalization beta shape {beta_shape} must be "
                f"({hidden_size},)"
            )

    bias_shape: tuple[int, ...] | None = None
    if bias_name is not None:
        bias_dtype = value_dtype(graph, bias_name, node)
        if bias_dtype != op_dtype:
            raise UnsupportedOpError(
                "SkipLayerNormalization bias dtype must match input dtype"
            )
        bias_shape = value_shape(graph, bias_name, node)
        if bias_shape != (hidden_size,):
            raise ShapeInferenceError(
                f"SkipLayerNormalization bias shape {bias_shape} must be "
                f"({hidden_size},)"
            )

    if skip_input_bias_add_name is not None:
        add_out_shape = value_shape(graph, skip_input_bias_add_name, node)
        if add_out_shape != input_shape:
            raise ShapeInferenceError(
                f"SkipLayerNormalization skip_input_bias_add_output shape "
                f"{add_out_shape} must match input shape {input_shape}"
            )

    epsilon = float(node.attrs.get("epsilon", 1e-12))
    outer = shape_product(input_shape[:-1])

    return SkipLayerNormalizationOp(
        input0=input_name,
        skip=skip_name,
        gamma=gamma_name,
        beta=beta_name,
        bias=bias_name,
        output=output_name,
        skip_input_bias_add_output=skip_input_bias_add_name,
        shape=input_shape,
        skip_shape=skip_shape,
        gamma_shape=gamma_shape,
        beta_shape=beta_shape,
        bias_shape=bias_shape,
        hidden_size=hidden_size,
        outer=outer,
        epsilon=epsilon,
        dtype=op_dtype,
    )
