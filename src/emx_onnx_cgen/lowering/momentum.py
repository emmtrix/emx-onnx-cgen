from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import MomentumOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("Momentum")
def lower_momentum(graph: Graph, node: Node) -> MomentumOp:
    if len(node.inputs) < 5:
        raise UnsupportedOpError("Momentum must have at least 5 inputs")
    if len(node.outputs) < 2:
        raise UnsupportedOpError("Momentum must have at least 2 outputs")
    if (len(node.inputs) - 2) % 3 != 0:
        raise UnsupportedOpError(
            "Momentum inputs must be R, T, Xs, Gs, Vs with matching counts"
        )
    tensor_count = (len(node.inputs) - 2) // 3
    if len(node.outputs) != tensor_count * 2:
        raise UnsupportedOpError("Momentum outputs must be X_news followed by V_news")
    rate_name = node.inputs[0]
    timestep_name = node.inputs[1]
    rate_shape = value_shape(graph, rate_name, node)
    timestep_shape = value_shape(graph, timestep_name, node)
    if not _is_scalar_shape(rate_shape):
        raise UnsupportedOpError("Momentum R input must be a scalar")
    if not _is_scalar_shape(timestep_shape):
        raise UnsupportedOpError("Momentum T input must be a scalar")
    rate_dtype = value_dtype(graph, rate_name, node)
    if rate_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError("Momentum R input must be float or double")
    timestep_dtype = value_dtype(graph, timestep_name, node)
    if timestep_dtype != ScalarType.I64:
        raise UnsupportedOpError("Momentum T input must be int64")

    inputs = node.inputs[2 : 2 + tensor_count]
    gradients = node.inputs[2 + tensor_count : 2 + tensor_count * 2]
    velocities = node.inputs[2 + tensor_count * 2 : 2 + tensor_count * 3]
    outputs = node.outputs[:tensor_count]
    velocity_outputs = node.outputs[tensor_count:]
    if not inputs or not gradients or not velocities:
        raise UnsupportedOpError("Momentum requires X, G, V inputs")
    dtype = value_dtype(graph, inputs[0], node)
    if dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError("Momentum supports float and double tensors only")
    if rate_dtype != dtype:
        raise UnsupportedOpError("Momentum R input dtype must match tensor dtype")

    mode_attr = node.attrs.get("mode", "standard")
    if isinstance(mode_attr, bytes):
        mode = mode_attr.decode()
    else:
        mode = str(mode_attr)
    if mode not in {"standard", "nesterov"}:
        raise UnsupportedOpError(
            f"Momentum mode must be 'standard' or 'nesterov', got '{mode}'"
        )

    input_shapes: list[tuple[int, ...]] = []
    output_shapes: list[tuple[int, ...]] = []
    for index, (x_name, g_name, v_name, out_name, v_out_name) in enumerate(
        zip(inputs, gradients, velocities, outputs, velocity_outputs)
    ):
        x_dtype = value_dtype(graph, x_name, node)
        g_dtype = value_dtype(graph, g_name, node)
        v_dtype = value_dtype(graph, v_name, node)
        out_dtype = value_dtype(graph, out_name, node)
        v_out_dtype = value_dtype(graph, v_out_name, node)
        if {x_dtype, g_dtype, v_dtype, out_dtype, v_out_dtype} != {dtype}:
            raise UnsupportedOpError(
                "Momentum inputs and outputs must share the same dtype"
            )
        x_shape = value_shape(graph, x_name, node)
        g_shape = value_shape(graph, g_name, node)
        v_shape = value_shape(graph, v_name, node)
        out_shape = value_shape(graph, out_name, node)
        v_out_shape = value_shape(graph, v_out_name, node)
        if x_shape != g_shape or x_shape != v_shape:
            raise ShapeInferenceError(
                f"Momentum inputs X/G/V shapes must match for tensor {index}"
            )
        if out_shape != x_shape or v_out_shape != x_shape:
            raise ShapeInferenceError(
                f"Momentum outputs must match X shape for tensor {index}"
            )
        input_shapes.append(x_shape)
        output_shapes.append(out_shape)

    norm_coefficient = float(node.attrs.get("norm_coefficient", 0.0))
    alpha = float(node.attrs.get("alpha", 0.0))
    beta = float(node.attrs.get("beta", 0.0))

    return MomentumOp(
        rate=rate_name,
        timestep=timestep_name,
        inputs=tuple(inputs),
        gradients=tuple(gradients),
        velocities=tuple(velocities),
        outputs=tuple(outputs),
        velocity_outputs=tuple(velocity_outputs),
        rate_shape=rate_shape,
        timestep_shape=timestep_shape,
        tensor_shapes=tuple(input_shapes),
        output_shapes=tuple(output_shapes),
        dtype=dtype,
        rate_dtype=rate_dtype,
        timestep_dtype=timestep_dtype,
        norm_coefficient=norm_coefficient,
        alpha=alpha,
        beta=beta,
        mode=mode,
    )
