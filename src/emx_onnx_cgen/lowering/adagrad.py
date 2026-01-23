from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import AdagradOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("Adagrad")
def lower_adagrad(graph: Graph, node: Node) -> AdagradOp:
    if len(node.inputs) < 5:
        raise UnsupportedOpError("Adagrad must have at least 5 inputs")
    if len(node.outputs) < 2:
        raise UnsupportedOpError("Adagrad must have at least 2 outputs")
    if (len(node.inputs) - 2) % 3 != 0:
        raise UnsupportedOpError(
            "Adagrad inputs must be R, T, Xs, Gs, Hs with matching counts"
        )
    tensor_count = (len(node.inputs) - 2) // 3
    if len(node.outputs) != tensor_count * 2:
        raise UnsupportedOpError(
            "Adagrad outputs must be X_news followed by H_news"
        )
    rate_name = node.inputs[0]
    timestep_name = node.inputs[1]
    rate_shape = value_shape(graph, rate_name, node)
    timestep_shape = value_shape(graph, timestep_name, node)
    if not _is_scalar_shape(rate_shape):
        raise UnsupportedOpError("Adagrad R input must be a scalar")
    if not _is_scalar_shape(timestep_shape):
        raise UnsupportedOpError("Adagrad T input must be a scalar")
    rate_dtype = value_dtype(graph, rate_name, node)
    if rate_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(
            "Adagrad R input must be float or double"
        )
    timestep_dtype = value_dtype(graph, timestep_name, node)
    if timestep_dtype != ScalarType.I64:
        raise UnsupportedOpError("Adagrad T input must be int64")

    inputs = node.inputs[2 : 2 + tensor_count]
    gradients = node.inputs[2 + tensor_count : 2 + tensor_count * 2]
    accumulators = node.inputs[2 + tensor_count * 2 : 2 + tensor_count * 3]
    outputs = node.outputs[:tensor_count]
    accumulator_outputs = node.outputs[tensor_count:]
    if not inputs or not gradients or not accumulators:
        raise UnsupportedOpError("Adagrad requires X, G, H inputs")
    dtype = value_dtype(graph, inputs[0], node)
    if dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError("Adagrad supports float and double tensors only")
    if rate_dtype != dtype:
        raise UnsupportedOpError(
            "Adagrad R input dtype must match tensor dtype"
        )
    input_shapes: list[tuple[int, ...]] = []
    output_shapes: list[tuple[int, ...]] = []
    for index, (x_name, g_name, h_name, out_name, h_out_name) in enumerate(
        zip(inputs, gradients, accumulators, outputs, accumulator_outputs)
    ):
        x_dtype = value_dtype(graph, x_name, node)
        g_dtype = value_dtype(graph, g_name, node)
        h_dtype = value_dtype(graph, h_name, node)
        out_dtype = value_dtype(graph, out_name, node)
        h_out_dtype = value_dtype(graph, h_out_name, node)
        if {x_dtype, g_dtype, h_dtype, out_dtype, h_out_dtype} != {dtype}:
            raise UnsupportedOpError(
                "Adagrad inputs and outputs must share the same dtype"
            )
        x_shape = value_shape(graph, x_name, node)
        g_shape = value_shape(graph, g_name, node)
        h_shape = value_shape(graph, h_name, node)
        out_shape = value_shape(graph, out_name, node)
        h_out_shape = value_shape(graph, h_out_name, node)
        if x_shape != g_shape or x_shape != h_shape:
            raise ShapeInferenceError(
                f"Adagrad inputs X/G/H shapes must match for tensor {index}"
            )
        if out_shape != x_shape or h_out_shape != x_shape:
            raise ShapeInferenceError(
                f"Adagrad outputs must match X shape for tensor {index}"
            )
        input_shapes.append(x_shape)
        output_shapes.append(out_shape)

    norm_coefficient = float(node.attrs.get("norm_coefficient", 0.0))
    epsilon = float(node.attrs.get("epsilon", 0.0))
    decay_factor = float(node.attrs.get("decay_factor", 0.0))

    return AdagradOp(
        rate=rate_name,
        timestep=timestep_name,
        inputs=tuple(inputs),
        gradients=tuple(gradients),
        accumulators=tuple(accumulators),
        outputs=tuple(outputs),
        accumulator_outputs=tuple(accumulator_outputs),
        rate_shape=rate_shape,
        timestep_shape=timestep_shape,
        tensor_shapes=tuple(input_shapes),
        output_shapes=tuple(output_shapes),
        dtype=dtype,
        rate_dtype=rate_dtype,
        timestep_dtype=timestep_dtype,
        norm_coefficient=norm_coefficient,
        epsilon=epsilon,
        decay_factor=decay_factor,
    )
