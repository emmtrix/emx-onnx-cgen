from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import AdamOp
from .common import value_dtype, value_shape
from .registry import register_lowering


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("Adam")
def lower_adam(graph: Graph, node: Node) -> AdamOp:
    if len(node.inputs) < 6:
        raise UnsupportedOpError("Adam must have at least 6 inputs")
    if len(node.outputs) < 3:
        raise UnsupportedOpError("Adam must have at least 3 outputs")
    if (len(node.inputs) - 2) % 4 != 0:
        raise UnsupportedOpError(
            "Adam inputs must be R, T, Xs, Gs, Vs, Hs with matching counts"
        )
    tensor_count = (len(node.inputs) - 2) // 4
    if len(node.outputs) != tensor_count * 3:
        raise UnsupportedOpError(
            "Adam outputs must be X_news followed by V_news followed by H_news"
        )

    rate_name = node.inputs[0]
    timestep_name = node.inputs[1]
    rate_shape = value_shape(graph, rate_name, node)
    timestep_shape = value_shape(graph, timestep_name, node)
    if not _is_scalar_shape(rate_shape):
        raise UnsupportedOpError("Adam R input must be a scalar")
    if not _is_scalar_shape(timestep_shape):
        raise UnsupportedOpError("Adam T input must be a scalar")

    rate_dtype = value_dtype(graph, rate_name, node)
    if rate_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError("Adam R input must be float or double")
    timestep_dtype = value_dtype(graph, timestep_name, node)
    if timestep_dtype != ScalarType.I64:
        raise UnsupportedOpError("Adam T input must be int64")

    inputs = node.inputs[2 : 2 + tensor_count]
    gradients = node.inputs[2 + tensor_count : 2 + tensor_count * 2]
    velocities = node.inputs[2 + tensor_count * 2 : 2 + tensor_count * 3]
    moments = node.inputs[2 + tensor_count * 3 : 2 + tensor_count * 4]

    outputs = node.outputs[:tensor_count]
    velocity_outputs = node.outputs[tensor_count : tensor_count * 2]
    moment_outputs = node.outputs[tensor_count * 2 : tensor_count * 3]

    if not inputs or not gradients or not velocities or not moments:
        raise UnsupportedOpError("Adam requires X, G, V, H inputs")

    dtype = value_dtype(graph, inputs[0], node)
    if dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError("Adam supports float and double tensors only")
    if rate_dtype != dtype:
        raise UnsupportedOpError("Adam R input dtype must match tensor dtype")

    input_shapes: list[tuple[int, ...]] = []
    output_shapes: list[tuple[int, ...]] = []
    for index, (
        x_name,
        g_name,
        v_name,
        h_name,
        out_name,
        v_out_name,
        h_out_name,
    ) in enumerate(
        zip(
            inputs,
            gradients,
            velocities,
            moments,
            outputs,
            velocity_outputs,
            moment_outputs,
        )
    ):
        x_dtype = value_dtype(graph, x_name, node)
        g_dtype = value_dtype(graph, g_name, node)
        v_dtype = value_dtype(graph, v_name, node)
        h_dtype = value_dtype(graph, h_name, node)
        out_dtype = value_dtype(graph, out_name, node)
        v_out_dtype = value_dtype(graph, v_out_name, node)
        h_out_dtype = value_dtype(graph, h_out_name, node)
        if {
            x_dtype,
            g_dtype,
            v_dtype,
            h_dtype,
            out_dtype,
            v_out_dtype,
            h_out_dtype,
        } != {dtype}:
            raise UnsupportedOpError(
                "Adam inputs and outputs must share the same dtype"
            )

        x_shape = value_shape(graph, x_name, node)
        g_shape = value_shape(graph, g_name, node)
        v_shape = value_shape(graph, v_name, node)
        h_shape = value_shape(graph, h_name, node)
        out_shape = value_shape(graph, out_name, node)
        v_out_shape = value_shape(graph, v_out_name, node)
        h_out_shape = value_shape(graph, h_out_name, node)
        if x_shape != g_shape or x_shape != v_shape or x_shape != h_shape:
            raise ShapeInferenceError(
                f"Adam inputs X/G/V/H shapes must match for tensor {index}"
            )
        if out_shape != x_shape or v_out_shape != x_shape or h_out_shape != x_shape:
            raise ShapeInferenceError(
                f"Adam outputs must match X shape for tensor {index}"
            )
        input_shapes.append(x_shape)
        output_shapes.append(out_shape)

    alpha = float(node.attrs.get("alpha", 0.9))
    beta = float(node.attrs.get("beta", 0.999))
    epsilon = float(node.attrs.get("epsilon", 0.0))
    norm_coefficient = float(node.attrs.get("norm_coefficient", 0.0))
    norm_coefficient_post = float(node.attrs.get("norm_coefficient_post", 0.0))

    return AdamOp(
        rate=rate_name,
        timestep=timestep_name,
        inputs=tuple(inputs),
        gradients=tuple(gradients),
        velocities=tuple(velocities),
        moments=tuple(moments),
        outputs=tuple(outputs),
        velocity_outputs=tuple(velocity_outputs),
        moment_outputs=tuple(moment_outputs),
        rate_shape=rate_shape,
        timestep_shape=timestep_shape,
        tensor_shapes=tuple(input_shapes),
        output_shapes=tuple(output_shapes),
        dtype=dtype,
        rate_dtype=rate_dtype,
        timestep_dtype=timestep_dtype,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        norm_coefficient=norm_coefficient,
        norm_coefficient_post=norm_coefficient_post,
    )
