from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import numpy as np

from shared.scalar_types import ScalarType
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.attention import resolve_attention_spec
from ..lowering.average_pool import lower_average_pool, lower_global_average_pool
from ..lowering.adagrad import lower_adagrad
from ..lowering.batch_normalization import lower_batch_normalization
from ..lowering.concat import lower_concat
from ..lowering.constant_of_shape import lower_constant_of_shape
from ..lowering.conv import resolve_conv_spec
from ..lowering.conv_transpose import resolve_conv_transpose_spec
from ..lowering.dropout import lower_dropout
from ..lowering.cumsum import lower_cumsum
from ..lowering.einsum import lower_einsum
from ..lowering.flatten import lower_flatten
from ..lowering.gemm import resolve_gemm_spec
from ..lowering.logsoftmax import lower_logsoftmax
from ..lowering.hardmax import lower_hardmax
from ..lowering.lp_normalization import lower_lp_normalization
from ..lowering.lp_pool import lower_lp_pool
from ..lowering.grid_sample import lower_grid_sample
from ..lowering.instance_normalization import lower_instance_normalization
from ..lowering.group_normalization import lower_group_normalization
from ..lowering.layer_normalization import lower_layer_normalization
from ..lowering.mean_variance_normalization import (
    lower_mean_variance_normalization,
)
from ..lowering.global_max_pool import lower_global_max_pool
from ..lowering.negative_log_likelihood_loss import (
    lower_negative_log_likelihood_loss,
)
from ..lowering.nonzero import lower_nonzero
from ..lowering.pad import lower_pad
from ..lowering.expand import lower_expand
from ..lowering.range import lower_range
from ..lowering.one_hot import lower_onehot
from ..lowering.split import lower_split
from ..lowering.softmax_cross_entropy_loss import (
    lower_softmax_cross_entropy_loss,
)
from ..lowering.arg_reduce import lower_arg_reduce
from ..lowering.topk import lower_topk
from ..lowering.lstm import ACTIVATION_KIND_BY_NAME, resolve_lstm_spec
from ..lowering.lrn import resolve_lrn_spec
from ..lowering.matmul import lower_matmul
from ..lowering.maxpool import resolve_maxpool_spec
from ..lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
    normalize_reduce_axes,
    resolve_reduce_axes,
)
from ..lowering.reshape import lower_reshape
from ..lowering.scatter_nd import lower_scatternd
from ..lowering.tensor_scatter import lower_tensor_scatter
from ..lowering.slice import _normalize_slices
from ..lowering.shape import lower_shape
from ..lowering.size import lower_size
from ..lowering.softmax import lower_softmax
from ..lowering.rms_normalization import lower_rms_normalization
from ..lowering.rotary_embedding import lower_rotary_embedding
from ..lowering.squeeze import lower_squeeze
from ..lowering.transpose import lower_transpose
from ..lowering.unsqueeze import lower_unsqueeze
from ..lowering.where import lower_where
from ..lowering.quantize_linear import resolve_quantize_spec
from ..lowering.variadic import BINARY_ONLY_OPS, VARIADIC_OP_FUNCTIONS
from ..lowering.registry import resolve_dispatch
from ..lowering.common import node_dtype, optional_name, value_dtype, value_shape
from ..ops import (
    BINARY_OP_TYPES,
    COMPARE_FUNCTIONS,
    UNARY_OP_TYPES,
    apply_binary_op,
    apply_unary_op,
    binary_op_symbol,
    unary_op_symbol,
    validate_unary_attrs,
)
from shared.scalar_functions import ScalarFunction, ScalarFunctionError
from ..validation import normalize_axis

Handler = Callable[["Evaluator", Node], None]
_EVAL_REGISTRY: dict[str, Handler] = {}


def register_evaluator(op_type: str) -> Callable[[Handler], Handler]:
    def decorator(func: Handler) -> Handler:
        _EVAL_REGISTRY[op_type] = func
        return func

    return decorator


class Evaluator:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._values: dict[str, np.ndarray] = {}

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def values(self) -> dict[str, np.ndarray]:
        return self._values

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = {
            initializer.name: initializer.data
            for initializer in self._graph.initializers
        }
        values.update(feeds)
        self._values = values
        for node in self._graph.nodes:
            self._dispatch(node)
        return {
            output.name: self._values[output.name]
            for output in self._graph.outputs
        }

    def _dispatch(self, node: Node) -> None:
        handler = resolve_dispatch(
            node.op_type,
            _EVAL_REGISTRY,
            binary_types=BINARY_OP_TYPES,
            unary_types=UNARY_OP_TYPES,
            binary_fallback=lambda: _eval_binary_unary,
            unary_fallback=lambda: _eval_binary_unary,
        )
        handler(self, node)


@register_evaluator("MatMul")
def _eval_matmul(evaluator: Evaluator, node: Node) -> None:
    lower_matmul(evaluator.graph, node)
    left = evaluator.values[node.inputs[0]]
    right = evaluator.values[node.inputs[1]]
    evaluator.values[node.outputs[0]] = _apply_matmul(left, right)


@register_evaluator("Einsum")
def _eval_einsum(evaluator: Evaluator, node: Node) -> None:
    lower_einsum(evaluator.graph, node)
    equation_value = node.attrs.get("equation")
    if equation_value is None:
        raise UnsupportedOpError("Einsum equation attribute is required")
    equation = (
        equation_value.decode()
        if isinstance(equation_value, (bytes, bytearray))
        else str(equation_value)
    )
    inputs = [evaluator.values[name] for name in node.inputs]
    evaluator.values[node.outputs[0]] = np.einsum(equation, *inputs)


@register_evaluator("Adagrad")
def _eval_adagrad(evaluator: Evaluator, node: Node) -> None:
    op = lower_adagrad(evaluator.graph, node)
    rate = evaluator.values[op.rate]
    timestep = evaluator.values[op.timestep]
    rate_value = (
        np.array(rate, dtype=op.dtype.np_dtype).reshape(-1)[0].item()
    )
    timestep_value = (
        np.array(timestep, dtype=np.int64).reshape(-1)[0].item()
    )
    r = op.dtype.np_dtype.type(
        rate_value / (1.0 + float(timestep_value) * op.decay_factor)
    )
    for x_name, g_name, h_name, out_name, h_out_name in zip(
        op.inputs,
        op.gradients,
        op.accumulators,
        op.outputs,
        op.accumulator_outputs,
    ):
        x = evaluator.values[x_name]
        g = evaluator.values[g_name]
        h = evaluator.values[h_name]
        g_regularized = op.norm_coefficient * x + g
        h_new = h + g_regularized * g_regularized
        h_adaptive = np.sqrt(h_new) + op.epsilon
        evaluator.values[out_name] = x - r * g_regularized / h_adaptive
        evaluator.values[h_out_name] = h_new


@register_evaluator("Clip")
def _eval_clip(evaluator: Evaluator, node: Node) -> None:
    if not node.inputs or len(node.outputs) != 1:
        raise UnsupportedOpError("Clip must have 1 output")
    input_name = node.inputs[0]
    if not input_name:
        raise UnsupportedOpError("Clip input must be provided")
    x = evaluator.values[input_name]
    min_name = optional_name(node.inputs, 1)
    max_name = optional_name(node.inputs, 2)
    dtype = value_dtype(evaluator.graph, input_name, node)
    if min_name is None:
        min_val = (
            -np.inf
            if dtype.is_float
            else np.iinfo(dtype.np_dtype).min
        )
    else:
        min_val = evaluator.values[min_name]
    if max_name is None:
        max_val = (
            np.inf
            if dtype.is_float
            else np.iinfo(dtype.np_dtype).max
        )
    else:
        max_val = evaluator.values[max_name]
    evaluator.values[node.outputs[0]] = np.clip(x, min_val, max_val)


def _exclusive_cumsum(data: np.ndarray, axis: int) -> np.ndarray:
    result = np.zeros_like(data)
    if data.shape[axis] == 0:
        return result
    cumsum = np.cumsum(data, axis=axis, dtype=data.dtype)
    src_slice = [slice(None)] * data.ndim
    dst_slice = [slice(None)] * data.ndim
    src_slice[axis] = slice(None, -1)
    dst_slice[axis] = slice(1, None)
    result[tuple(dst_slice)] = cumsum[tuple(src_slice)]
    return result


@register_evaluator("CumSum")
def _eval_cumsum(evaluator: Evaluator, node: Node) -> None:
    op = lower_cumsum(evaluator.graph, node)
    x = evaluator.values[op.input0]
    axis = op.axis
    if axis is None:
        axis_values = evaluator.values[op.axis_input].astype(np.int64, copy=False)
        axis_values = axis_values.reshape(-1)
        if axis_values.size != 1:
            raise UnsupportedOpError("CumSum axis input must be scalar")
        axis = normalize_axis(int(axis_values[0]), op.input_shape, node)
    data = np.flip(x, axis=axis) if op.reverse else x
    if op.exclusive:
        result = _exclusive_cumsum(data, axis)
    else:
        result = np.cumsum(data, axis=axis, dtype=data.dtype)
    if op.reverse:
        result = np.flip(result, axis=axis)
    evaluator.values[op.output] = result


@register_evaluator("Pad")
def _eval_pad(evaluator: Evaluator, node: Node) -> None:
    op = lower_pad(evaluator.graph, node)
    x = evaluator.values[op.input0]
    if op.value_input is not None:
        value_array = evaluator.values[op.value_input]
        pad_value = np.array(value_array, dtype=op.dtype.np_dtype).reshape(-1)[0].item()
    else:
        pad_value = np.array(op.value, dtype=op.dtype.np_dtype).item()
    rank = len(op.input_shape)
    if op.axes_input is not None:
        axes_values = evaluator.values[op.axes_input].astype(
            np.int64, copy=False
        )
        axes_values = axes_values.reshape(-1)
        if op.pads_input is not None:
            pads_values = evaluator.values[op.pads_input].astype(
                np.int64, copy=False
            )
            pads_values = pads_values.reshape(-1)
        else:
            pads_values = np.array(op.pads_values, dtype=np.int64).reshape(-1)
        axis_count = len(axes_values)
        pads_begin = np.zeros(rank, dtype=np.int64)
        pads_end = np.zeros(rank, dtype=np.int64)
        for index, axis_value in enumerate(axes_values):
            axis = int(axis_value)
            if axis < 0:
                axis += rank
            pads_begin[axis] = int(pads_values[index])
            pads_end[axis] = int(pads_values[index + axis_count])
        pad_width = tuple(
            (int(pads_begin[index]), int(pads_end[index]))
            for index in range(rank)
        )
    elif op.pads_input is not None:
        pads_values = evaluator.values[op.pads_input].astype(np.int64, copy=False)
        pads_values = pads_values.reshape(-1)
        if op.pads_axis_map is not None:
            axis_count = sum(
                1 for axis_index in op.pads_axis_map if axis_index is not None
            )
            pads_begin = np.zeros(rank, dtype=np.int64)
            pads_end = np.zeros(rank, dtype=np.int64)
            for axis, pad_index in enumerate(op.pads_axis_map):
                if pad_index is not None:
                    pads_begin[axis] = int(pads_values[pad_index])
                    pads_end[axis] = int(
                        pads_values[pad_index + axis_count]
                    )
            pad_width = tuple(
                (int(pads_begin[index]), int(pads_end[index]))
                for index in range(rank)
            )
        else:
            pads_begin = pads_values[:rank]
            pads_end = pads_values[rank: rank * 2]
            pad_width = tuple(
                (int(pads_begin[index]), int(pads_end[index]))
                for index in range(rank)
            )
    else:
        pad_width = tuple(zip(op.pads_begin or (), op.pads_end or ()))
    pad_kwargs = {}
    if op.mode == "constant":
        pad_kwargs["constant_values"] = pad_value
    evaluator.values[op.output] = np.pad(
        x,
        pad_width,
        mode=op.mode,
        **pad_kwargs,
    )


@register_evaluator("ScatterND")
def _eval_scatternd(evaluator: Evaluator, node: Node) -> None:
    op = lower_scatternd(evaluator.graph, node)
    data = evaluator.values[op.data]
    indices = evaluator.values[op.indices]
    updates = evaluator.values[op.updates]
    output = np.array(data, copy=True)
    index_depth = op.indices_shape[-1]
    update_indices_shape = op.indices_shape[:-1]
    update_count = int(np.prod(update_indices_shape)) if update_indices_shape else 1
    flat_indices = indices.astype(np.int64, copy=False).reshape(
        update_count, index_depth
    )
    tail_shape = op.data_shape[index_depth:]
    updates_reshaped = updates.reshape((update_count,) + tail_shape)
    for index, index_values in enumerate(flat_indices):
        output_index: list[int | slice] = []
        for axis, value in enumerate(index_values):
            axis_size = op.data_shape[axis]
            idx = int(value)
            if idx < 0:
                idx += axis_size
            if idx < 0 or idx >= axis_size:
                raise UnsupportedOpError(
                    "ScatterND indices must be within data bounds"
                )
            output_index.append(idx)
        output_index.extend([slice(None)] * len(tail_shape))
        target = tuple(output_index)
        update_value = updates_reshaped[index]
        if op.reduction == "none":
            output[target] = update_value
        elif op.reduction == "add":
            output[target] = output[target] + update_value
        elif op.reduction == "mul":
            output[target] = output[target] * update_value
        elif op.reduction == "min":
            output[target] = np.minimum(output[target], update_value)
        elif op.reduction == "max":
            output[target] = np.maximum(output[target], update_value)
        else:
            raise UnsupportedOpError(
                f"Unsupported ScatterND reduction {op.reduction}"
            )
    evaluator.values[op.output] = output


@register_evaluator("TensorScatter")
def _eval_tensor_scatter(evaluator: Evaluator, node: Node) -> None:
    op = lower_tensor_scatter(evaluator.graph, node)
    past_cache = evaluator.values[op.past_cache]
    update = evaluator.values[op.update]
    if op.write_indices is None:
        write_indices = np.zeros((past_cache.shape[0],), dtype=np.int64)
    else:
        write_indices = evaluator.values[op.write_indices].astype(
            np.int64, copy=False
        )
    axis = op.axis
    max_sequence_length = past_cache.shape[axis]
    sequence_length = update.shape[axis]
    output = np.array(past_cache, copy=True)
    for prefix_idx in np.ndindex(past_cache.shape[:axis]):
        batch_idx = prefix_idx[0]
        base_index = int(write_indices[batch_idx])
        for sequence_idx in range(sequence_length):
            cache_idx = (*prefix_idx, base_index + sequence_idx)
            if op.mode == "circular":
                cache_idx = tuple(
                    np.mod(np.asarray(cache_idx), max_sequence_length)
                )
            update_idx = (*prefix_idx, sequence_idx)
            output[cache_idx] = update[update_idx]
    evaluator.values[op.output] = output


@register_evaluator("Celu")
def _eval_celu(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Celu must have 1 input and 1 output")
    dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    if not dtype.is_float:
        raise UnsupportedOpError("Celu only supports floating-point inputs")
    alpha = float(node.attrs.get("alpha", 1.0))
    x = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = np.where(
        x > 0,
        x,
        alpha * (np.exp(x / alpha) - 1.0),
    )


@register_evaluator("Swish")
def _eval_swish(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Swish must have 1 input and 1 output")
    dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    if not dtype.is_float:
        raise UnsupportedOpError("Swish only supports floating-point inputs")
    alpha = float(node.attrs.get("alpha", 1.0))
    x = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = x / (1.0 + np.exp(-alpha * x))


def _grid_sample_denormalize(
    value: float, length: int, *, align_corners: bool
) -> float:
    if align_corners:
        return (value + 1.0) * (length - 1) / 2.0
    return ((value + 1.0) * length - 1.0) / 2.0


def _grid_sample_reflect(value: float, x_min: float, x_max: float) -> float:
    rng = x_max - x_min
    if rng == 0:
        return x_min
    if value < x_min:
        dx = x_min - value
        n = int(dx / rng)
        r = dx - n * rng
        return x_min + r if n % 2 == 0 else x_max - r
    if value > x_max:
        dx = value - x_max
        n = int(dx / rng)
        r = dx - n * rng
        return x_max - r if n % 2 == 0 else x_min + r
    return value


def _grid_sample_border(
    dims: tuple[int, ...], *, align_corners: bool
) -> tuple[list[float], list[float]]:
    min_vals: list[float] = []
    max_vals: list[float] = []
    for dim in dims:
        if align_corners:
            min_vals.append(0.0)
            max_vals.append(dim - 1.0)
        else:
            min_vals.append(-0.5)
            max_vals.append(dim - 0.5)
    return min_vals, max_vals


def _grid_sample_pixel_at(
    data: np.ndarray,
    indices: list[int],
    border_min: list[float],
    border_max: list[float],
    padding_mode: str,
) -> float:
    if padding_mode == "zeros":
        for idx, dim in zip(indices, data.shape):
            if idx < 0 or idx >= dim:
                return data.dtype.type(0)
        return data[tuple(indices)]
    if padding_mode == "border":
        clamped = [
            0 if idx < 0 else dim - 1 if idx >= dim else idx
            for idx, dim in zip(indices, data.shape)
        ]
        return data[tuple(clamped)]
    reflected = [
        int(_grid_sample_reflect(idx, border_min[i], border_max[i]))
        for i, idx in enumerate(indices)
    ]
    return data[tuple(reflected)]


def _grid_sample_linear_1d(
    data: np.ndarray,
    coord: float,
    border_min: float,
    border_max: float,
    padding_mode: str,
) -> float:
    base = int(np.floor(coord))
    weight = coord - base
    lower = _grid_sample_pixel_at(
        data, [base], [border_min], [border_max], padding_mode
    )
    upper = _grid_sample_pixel_at(
        data, [base + 1], [border_min], [border_max], padding_mode
    )
    return (1.0 - weight) * lower + weight * upper


def _grid_sample_cubic_coeffs(x: float) -> np.ndarray:
    alpha = -0.75
    abs_x = abs(x)
    coeffs = np.empty((4,), dtype=np.float64)
    coeffs[0] = (
        (alpha * (abs_x + 1.0) - 5.0 * alpha) * (abs_x + 1.0) + 8.0 * alpha
    ) * (abs_x + 1.0) - 4.0 * alpha
    coeffs[1] = ((alpha + 2.0) * abs_x - (alpha + 3.0)) * abs_x * abs_x + 1.0
    inv_x = 1.0 - abs_x
    coeffs[2] = ((alpha + 2.0) * inv_x - (alpha + 3.0)) * inv_x * inv_x + 1.0
    span = 2.0 - abs_x
    coeffs[3] = (
        (alpha * span - 5.0 * alpha) * span + 8.0 * alpha
    ) * span - 4.0 * alpha
    return coeffs


def _grid_sample_cubic_1d(
    data: np.ndarray,
    coord: float,
    border_min: float,
    border_max: float,
    padding_mode: str,
) -> float:
    base = int(np.floor(coord))
    coeffs = _grid_sample_cubic_coeffs(coord - base)
    values = np.empty((4,), dtype=np.float64)
    for offset in range(4):
        values[offset] = _grid_sample_pixel_at(
            data,
            [base - 1 + offset],
            [border_min],
            [border_max],
            padding_mode,
        )
    return float(coeffs @ values)


def _grid_sample_linear_nd(
    data: np.ndarray,
    coords: np.ndarray,
    border_min: list[float],
    border_max: list[float],
    padding_mode: str,
) -> float:
    if data.ndim == 1:
        return _grid_sample_linear_1d(
            data, float(coords[0]), border_min[0], border_max[0], padding_mode
        )
    reduced = np.array(
        [
            _grid_sample_linear_nd(
                data[index],
                coords[1:],
                border_min[1:],
                border_max[1:],
                padding_mode,
            )
            for index in range(data.shape[0])
        ],
        dtype=np.float64,
    )
    return _grid_sample_linear_1d(
        reduced, float(coords[0]), border_min[0], border_max[0], padding_mode
    )


def _grid_sample_cubic_nd(
    data: np.ndarray,
    coords: np.ndarray,
    border_min: list[float],
    border_max: list[float],
    padding_mode: str,
) -> float:
    if data.ndim == 1:
        return _grid_sample_cubic_1d(
            data, float(coords[0]), border_min[0], border_max[0], padding_mode
        )
    reduced = np.array(
        [
            _grid_sample_cubic_nd(
                data[index],
                coords[1:],
                border_min[1:],
                border_max[1:],
                padding_mode,
            )
            for index in range(data.shape[0])
        ],
        dtype=np.float64,
    )
    return _grid_sample_cubic_1d(
        reduced, float(coords[0]), border_min[0], border_max[0], padding_mode
    )


@register_evaluator("GridSample")
def _eval_grid_sample(evaluator: Evaluator, node: Node) -> None:
    op = lower_grid_sample(evaluator.graph, node)
    input_data = evaluator.values[op.input0]
    grid_data = evaluator.values[op.grid]
    output = np.empty(op.output_shape, dtype=input_data.dtype)
    if output.size == 0:
        evaluator.values[op.output] = output
        return
    dims = op.input_spatial
    border_min, border_max = _grid_sample_border(
        dims, align_corners=op.align_corners
    )
    for n in range(op.output_shape[0]):
        grid_batch = grid_data[n]
        for c in range(op.output_shape[1]):
            input_slice = input_data[n, c]
            for out_idx in np.ndindex(*op.output_spatial):
                coords = np.array(
                    grid_batch[out_idx][::-1], dtype=np.float64
                )
                for i, dim in enumerate(dims):
                    coords[i] = _grid_sample_denormalize(
                        float(coords[i]), dim, align_corners=op.align_corners
                    )
                if op.mode == "nearest":
                    rounded = np.rint(coords).astype(int)
                    if op.padding_mode != "zeros":
                        for i, dim in enumerate(dims):
                            if (
                                rounded[i] < border_min[i]
                                or rounded[i] > border_max[i]
                            ):
                                if op.padding_mode == "border":
                                    rounded[i] = min(
                                        max(rounded[i], 0), dim - 1
                                    )
                                else:
                                    rounded[i] = int(
                                        _grid_sample_reflect(
                                            rounded[i],
                                            border_min[i],
                                            border_max[i],
                                        )
                                    )
                    value = _grid_sample_pixel_at(
                        input_slice,
                        rounded.tolist(),
                        border_min,
                        border_max,
                        op.padding_mode,
                    )
                else:
                    if op.padding_mode != "zeros":
                        for i, dim in enumerate(dims):
                            if (
                                coords[i] < border_min[i]
                                or coords[i] > border_max[i]
                            ):
                                if op.padding_mode == "border":
                                    coords[i] = min(
                                        max(coords[i], 0.0), dim - 1.0
                                    )
                                else:
                                    coords[i] = _grid_sample_reflect(
                                        coords[i],
                                        border_min[i],
                                        border_max[i],
                                    )
                    if op.mode == "linear":
                        value = _grid_sample_linear_nd(
                            input_slice,
                            coords,
                            border_min,
                            border_max,
                            op.padding_mode,
                        )
                    else:
                        value = _grid_sample_cubic_nd(
                            input_slice,
                            coords,
                            border_min,
                            border_max,
                            op.padding_mode,
                        )
                output[(n, c, *out_idx)] = value
    evaluator.values[op.output] = output


_VARIADIC_COMBINE_FUNCS: dict[
    ScalarFunction, Callable[[np.ndarray, np.ndarray], np.ndarray]
] = {
    ScalarFunction.ADD: np.add,
    ScalarFunction.MAXIMUM: np.maximum,
    ScalarFunction.MINIMUM: np.minimum,
    ScalarFunction.LOGICAL_AND: np.logical_and,
    ScalarFunction.LOGICAL_OR: np.logical_or,
    ScalarFunction.LOGICAL_XOR: np.logical_xor,
    ScalarFunction.BITWISE_AND: np.bitwise_and,
    ScalarFunction.BITWISE_OR: np.bitwise_or,
    ScalarFunction.BITWISE_XOR: np.bitwise_xor,
}


def _validate_variadic_inputs(
    evaluator: Evaluator, node: Node, *, function: ScalarFunction
) -> tuple[ScalarType, tuple[int, ...]]:
    if len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} must have 1 output")
    if node.op_type in BINARY_ONLY_OPS:
        if len(node.inputs) != 2:
            raise UnsupportedOpError(
                f"{node.op_type} must have exactly 2 inputs"
            )
    elif len(node.inputs) < 2:
        raise UnsupportedOpError(
            f"{node.op_type} must have at least 2 inputs"
        )
    for name in node.inputs:
        if not name:
            raise UnsupportedOpError(f"{node.op_type} input must be provided")
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    output_shape = value_shape(evaluator.graph, node.outputs[0], node)
    for name in node.inputs:
        input_shape = value_shape(evaluator.graph, name, node)
        if input_shape != output_shape:
            raise UnsupportedOpError(
                f"{node.op_type} expects identical input/output shapes"
            )
    if function in {
        ScalarFunction.LOGICAL_AND,
        ScalarFunction.LOGICAL_OR,
        ScalarFunction.LOGICAL_XOR,
    } and op_dtype != ScalarType.BOOL:
        raise UnsupportedOpError(f"{node.op_type} expects bool inputs")
    if function in {
        ScalarFunction.BITWISE_AND,
        ScalarFunction.BITWISE_OR,
        ScalarFunction.BITWISE_XOR,
    } and not op_dtype.is_integer:
        raise UnsupportedOpError(f"{node.op_type} expects integer inputs")
    if function == ScalarFunction.MEAN and not op_dtype.is_float:
        raise UnsupportedOpError(f"{node.op_type} expects floating-point inputs")
    return op_dtype, output_shape


def _eval_variadic(evaluator: Evaluator, node: Node) -> None:
    function = VARIADIC_OP_FUNCTIONS[node.op_type]
    _validate_variadic_inputs(evaluator, node, function=function)
    values = [evaluator.values[name] for name in node.inputs]
    if function == ScalarFunction.MEAN:
        combine_func = _VARIADIC_COMBINE_FUNCS[ScalarFunction.ADD]
    else:
        combine_func = _VARIADIC_COMBINE_FUNCS[function]
    result = values[0]
    for value in values[1:]:
        result = combine_func(result, value)
    if function == ScalarFunction.MEAN:
        result = result / len(values)
    evaluator.values[node.outputs[0]] = result


for _op_type in VARIADIC_OP_FUNCTIONS:
    register_evaluator(_op_type)(_eval_variadic)


@register_evaluator("Shrink")
def _eval_shrink(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Shrink must have 1 input and 1 output")
    bias = float(node.attrs.get("bias", 0.0))
    lambd = float(node.attrs.get("lambd", 0.5))
    x = evaluator.values[node.inputs[0]]
    result = np.where(
        x < -lambd,
        x + bias,
        np.where(x > lambd, x - bias, 0.0),
    )
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)
    evaluator.values[node.outputs[0]] = result


@register_evaluator("IsInf")
def _eval_isinf(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("IsInf must have 1 input and 1 output")
    input_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    if not input_dtype.is_float:
        raise UnsupportedOpError("IsInf only supports floating-point inputs")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if output_dtype != ScalarType.BOOL:
        raise UnsupportedOpError("IsInf output must be bool")
    x = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = np.isinf(x)


@register_evaluator("IsNaN")
def _eval_isnan(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("IsNaN must have 1 input and 1 output")
    input_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    if not input_dtype.is_float:
        raise UnsupportedOpError("IsNaN only supports floating-point inputs")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if output_dtype != ScalarType.BOOL:
        raise UnsupportedOpError("IsNaN output must be bool")
    x = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = np.isnan(x)


@register_evaluator("Gemm")
def _eval_gemm(evaluator: Evaluator, node: Node) -> None:
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    spec = resolve_gemm_spec(evaluator.graph, node, op_dtype)
    left = evaluator.values[node.inputs[0]]
    right = evaluator.values[node.inputs[1]]
    if spec.trans_a:
        left = left.T
    if spec.trans_b:
        right = right.T
    result = _apply_matmul(left, right)
    if op_dtype.is_float:
        alpha = float(spec.alpha)
        beta = float(spec.beta)
    else:
        alpha = int(spec.alpha)
        beta = int(spec.beta)
    if alpha != 1:
        result = result * alpha
    if len(node.inputs) == 3:
        bias = evaluator.values[node.inputs[2]]
        if beta != 1:
            bias = bias * beta
        result = result + bias
    evaluator.values[node.outputs[0]] = result


@register_evaluator("Cast")
def _eval_cast(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Cast must have 1 input and 1 output")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    input_value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = input_value.astype(
        output_dtype.np_dtype, copy=False
    )


@register_evaluator("CastLike")
def _eval_castlike(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("CastLike must have 2 inputs and 1 output")
    like_dtype = value_dtype(evaluator.graph, node.inputs[1], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if output_dtype != like_dtype:
        raise UnsupportedOpError(
            "CastLike output dtype must match like input dtype, "
            f"got {output_dtype.onnx_name} and {like_dtype.onnx_name}"
        )
    input_value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = input_value.astype(
        output_dtype.np_dtype, copy=False
    )


@register_evaluator("Identity")
def _eval_identity(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Identity must have 1 input and 1 output")
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = np.array(value, copy=True)


@register_evaluator("EyeLike")
def _eval_eye_like(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("EyeLike must have 1 input and 1 output")
    output_shape = value_shape(evaluator.graph, node.outputs[0], node)
    if len(output_shape) < 2:
        raise UnsupportedOpError("EyeLike expects input rank >= 2")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    k = int(node.attrs.get("k", 0))
    output = np.zeros(output_shape, dtype=output_dtype.np_dtype)
    rows, cols = output_shape[-2], output_shape[-1]
    row_start = 0 if k >= 0 else -k
    col_start = k if k >= 0 else 0
    if row_start < rows and col_start < cols:
        diag_len = min(rows - row_start, cols - col_start)
        batch_size = int(np.prod(output_shape[:-2])) if output_shape[:-2] else 1
        view = output.reshape(batch_size, rows, cols)
        diag_idx = np.arange(diag_len, dtype=np.int64)
        one = output_dtype.np_dtype.type(1)
        view[:, row_start + diag_idx, col_start + diag_idx] = one
    evaluator.values[node.outputs[0]] = output


@register_evaluator("Trilu")
def _eval_trilu(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError("Trilu must have 1 or 2 inputs and 1 output")
    value = evaluator.values[node.inputs[0]]
    if value.ndim < 2:
        raise UnsupportedOpError("Trilu expects input rank >= 2")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    input_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            "Trilu expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    k = 0
    if len(node.inputs) == 2 and node.inputs[1]:
        k_value = np.array(evaluator.values[node.inputs[1]], dtype=np.int64)
        if k_value.size != 1:
            raise UnsupportedOpError("Trilu k input must be scalar")
        k = int(k_value.reshape(-1)[0])
    upper_attr = node.attrs.get("upper", 1)
    upper = bool(int(upper_attr))
    rows, cols = value.shape[-2], value.shape[-1]
    batch_shape = value.shape[:-2]
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    view = value.reshape(batch_size, rows, cols)
    if upper:
        mask = np.triu(np.ones((rows, cols), dtype=bool), k=k)
    else:
        mask = np.tril(np.ones((rows, cols), dtype=bool), k=k)
    output = np.where(mask, view, np.zeros_like(view))
    evaluator.values[node.outputs[0]] = output.reshape(value.shape)


@register_evaluator("Tile")
def _eval_tile(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Tile must have 2 inputs and 1 output")
    value = evaluator.values[node.inputs[0]]
    repeats = evaluator.values[node.inputs[1]]
    repeats = np.array(repeats, dtype=np.int64).reshape(-1)
    if repeats.size != value.ndim:
        raise UnsupportedOpError(
            "Tile repeats must have the same rank as input shape"
        )
    evaluator.values[node.outputs[0]] = np.tile(value, repeats)


@register_evaluator("DepthToSpace")
def _eval_depth_to_space(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("DepthToSpace must have 1 input and 1 output")
    data = evaluator.values[node.inputs[0]]
    if data.ndim != 4:
        raise UnsupportedOpError("DepthToSpace only supports 4D inputs")
    blocksize = int(node.attrs.get("blocksize", 0))
    if blocksize <= 0:
        raise UnsupportedOpError(
            f"DepthToSpace blocksize must be > 0, got {blocksize}"
        )
    mode_attr = node.attrs.get("mode", "DCR")
    if isinstance(mode_attr, bytes):
        mode = mode_attr.decode()
    else:
        mode = str(mode_attr)
    if mode not in {"DCR", "CRD"}:
        raise UnsupportedOpError("DepthToSpace only supports mode DCR or CRD")
    b, c, h, w = data.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = data.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    evaluator.values[node.outputs[0]] = np.reshape(transposed, finalshape)


@register_evaluator("SpaceToDepth")
def _eval_space_to_depth(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("SpaceToDepth must have 1 input and 1 output")
    data = evaluator.values[node.inputs[0]]
    if data.ndim != 4:
        raise UnsupportedOpError("SpaceToDepth only supports 4D inputs")
    blocksize = int(node.attrs.get("blocksize", 0))
    if blocksize <= 0:
        raise UnsupportedOpError(
            f"SpaceToDepth blocksize must be > 0, got {blocksize}"
        )
    b, c, h, w = data.shape
    tmpshape = (
        b,
        c,
        h // blocksize,
        blocksize,
        w // blocksize,
        blocksize,
    )
    reshaped = np.reshape(data, tmpshape)
    transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        c * blocksize * blocksize,
        h // blocksize,
        w // blocksize,
    )
    evaluator.values[node.outputs[0]] = np.reshape(transposed, finalshape)


@register_evaluator("Where")
def _eval_where(evaluator: Evaluator, node: Node) -> None:
    lower_where(evaluator.graph, node)
    condition = evaluator.values[node.inputs[0]]
    x_value = evaluator.values[node.inputs[1]]
    y_value = evaluator.values[node.inputs[2]]
    evaluator.values[node.outputs[0]] = np.where(condition, x_value, y_value)


@register_evaluator("GatherElements")
def _eval_gather_elements(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GatherElements must have 2 inputs and 1 output")
    data = evaluator.values[node.inputs[0]]
    indices = evaluator.values[node.inputs[1]]
    if indices.dtype.type not in {np.int32, np.int64}:
        raise UnsupportedOpError(
            f"GatherElements indices must be int32 or int64, got {indices.dtype}"
        )
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data.shape, node)
    evaluator.values[node.outputs[0]] = np.take_along_axis(
        data, indices, axis=axis
    )


@register_evaluator("Gather")
def _eval_gather(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Gather must have 2 inputs and 1 output")
    data = evaluator.values[node.inputs[0]]
    indices = evaluator.values[node.inputs[1]]
    if indices.dtype.type not in {np.int32, np.int64}:
        raise UnsupportedOpError(
            f"Gather indices must be int32 or int64, got {indices.dtype}"
        )
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data.shape, node)
    evaluator.values[node.outputs[0]] = np.take(data, indices, axis=axis)


@register_evaluator("GatherND")
def _eval_gather_nd(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GatherND must have 2 inputs and 1 output")
    data = evaluator.values[node.inputs[0]]
    indices = evaluator.values[node.inputs[1]]
    if indices.dtype.type not in {np.int32, np.int64}:
        raise UnsupportedOpError(
            f"GatherND indices must be int32 or int64, got {indices.dtype}"
        )
    if indices.ndim < 1:
        raise UnsupportedOpError("GatherND indices must have rank >= 1")
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims < 0:
        raise UnsupportedOpError(
            f"GatherND batch_dims must be >= 0, got {batch_dims}"
        )
    if batch_dims > indices.ndim - 1:
        raise UnsupportedOpError(
            "GatherND batch_dims must be <= indices rank - 1, "
            f"got {batch_dims} vs {indices.ndim - 1}"
        )
    if batch_dims > data.ndim:
        raise UnsupportedOpError(
            "GatherND batch_dims must be <= data rank, "
            f"got {batch_dims} vs {data.ndim}"
        )
    if tuple(data.shape[:batch_dims]) != tuple(indices.shape[:batch_dims]):
        raise UnsupportedOpError(
            "GatherND batch_dims must match on data/indices, "
            f"got {data.shape} vs {indices.shape}"
        )
    index_depth = indices.shape[-1]
    if index_depth <= 0:
        raise UnsupportedOpError(
            "GatherND indices final dimension must be >= 1"
        )
    if index_depth > data.ndim - batch_dims:
        raise UnsupportedOpError(
            "GatherND indices final dimension must be <= data rank - "
            f"batch_dims, got {index_depth} vs {data.ndim - batch_dims}"
        )
    tail_shape = data.shape[batch_dims + index_depth :]
    output_shape = indices.shape[:-1] + tail_shape
    output = np.empty(output_shape, dtype=data.dtype)
    indices_prefix_shape = indices.shape[:-1]
    prefix_iter = (
        np.ndindex(*indices_prefix_shape) if indices_prefix_shape else [()]
    )
    for prefix in prefix_iter:
        raw_index = indices[prefix]
        if index_depth == 1:
            index_values = [int(np.asarray(raw_index).item())]
        else:
            index_values = [int(value) for value in raw_index]
        for dim_index, value in enumerate(index_values):
            if value < 0:
                index_values[dim_index] = value + data.shape[
                    batch_dims + dim_index
                ]
        data_index = list(prefix[:batch_dims]) + index_values
        data_index.extend([slice(None)] * len(tail_shape))
        output_index = prefix + (slice(None),) * len(tail_shape)
        output[output_index] = data[tuple(data_index)]
    evaluator.values[node.outputs[0]] = output


@register_evaluator("Slice")
def _eval_slice(evaluator: Evaluator, node: Node) -> None:
    input_value = evaluator.values[node.inputs[0]]
    if "starts" in node.attrs or "ends" in node.attrs:
        starts = [int(value) for value in node.attrs.get("starts", [])]
        ends = [int(value) for value in node.attrs.get("ends", [])]
        axes_attr = node.attrs.get("axes")
        axes = [int(value) for value in axes_attr] if axes_attr else None
        steps = None
    else:
        if len(node.inputs) < 3:
            raise UnsupportedOpError(
                f"{node.op_type} expects at least 3 inputs"
            )
        starts_value = evaluator.values[node.inputs[1]]
        ends_value = evaluator.values[node.inputs[2]]
        if starts_value.dtype.type not in {np.int32, np.int64}:
            raise UnsupportedOpError(
                f"{node.op_type} starts input must be int64 or int32"
            )
        if ends_value.dtype.type not in {np.int32, np.int64}:
            raise UnsupportedOpError(
                f"{node.op_type} ends input must be int64 or int32"
            )
        starts = [int(value) for value in starts_value.reshape(-1)]
        ends = [int(value) for value in ends_value.reshape(-1)]
        axes = None
        steps = None
        if len(node.inputs) >= 4 and node.inputs[3]:
            axes_value = evaluator.values[node.inputs[3]]
            if axes_value.dtype.type not in {np.int32, np.int64}:
                raise UnsupportedOpError(
                    f"{node.op_type} axes input must be int64 or int32"
                )
            axes = [int(value) for value in axes_value.reshape(-1)]
        if len(node.inputs) >= 5 and node.inputs[4]:
            steps_value = evaluator.values[node.inputs[4]]
            if steps_value.dtype.type not in {np.int32, np.int64}:
                raise UnsupportedOpError(
                    f"{node.op_type} steps input must be int64 or int32"
                )
            steps = [int(value) for value in steps_value.reshape(-1)]
    normalized_starts, normalized_steps, output_shape = _normalize_slices(
        input_value.shape, starts, ends, axes, steps, node
    )
    slices = tuple(
        slice(start, start + step * size, step)
        for start, step, size in zip(
            normalized_starts, normalized_steps, output_shape
        )
    )
    evaluator.values[node.outputs[0]] = input_value[slices]


@register_evaluator("Attention")
def _eval_attention(evaluator: Evaluator, node: Node) -> None:
    input_q = node.inputs[0]
    input_k = node.inputs[1]
    input_v = node.inputs[2]
    output_y = node.outputs[0]
    op_dtype = node_dtype(evaluator.graph, node, input_q, input_k, input_v, output_y)
    spec = resolve_attention_spec(evaluator.graph, node, op_dtype)
    attn_mask_name = optional_name(node.inputs, 3)
    past_key_name = optional_name(node.inputs, 4)
    past_value_name = optional_name(node.inputs, 5)
    nonpad_name = optional_name(node.inputs, 6)
    present_key_name = optional_name(node.outputs, 1)
    present_value_name = optional_name(node.outputs, 2)
    qk_matmul_output_name = optional_name(node.outputs, 3)
    output, present_key, present_value, qk_output = _apply_attention(
        spec,
        evaluator.values[input_q],
        evaluator.values[input_k],
        evaluator.values[input_v],
        evaluator.values[attn_mask_name] if attn_mask_name else None,
        evaluator.values[past_key_name] if past_key_name else None,
        evaluator.values[past_value_name] if past_value_name else None,
        evaluator.values[nonpad_name] if nonpad_name else None,
    )
    evaluator.values[output_y] = output
    if present_key_name is not None:
        evaluator.values[present_key_name] = present_key
    if present_value_name is not None:
        evaluator.values[present_value_name] = present_value
    if qk_matmul_output_name is not None:
        evaluator.values[qk_matmul_output_name] = qk_output


@register_evaluator("RotaryEmbedding")
def _eval_rotary_embedding(evaluator: Evaluator, node: Node) -> None:
    op = lower_rotary_embedding(evaluator.graph, node)
    x = evaluator.values[op.input0]
    cos_cache = evaluator.values[op.cos_cache]
    sin_cache = evaluator.values[op.sin_cache]
    position_ids = (
        evaluator.values[op.position_ids] if op.position_ids else None
    )
    original_shape = x.shape
    if op.input_rank == 4:
        x = np.transpose(x, (0, 2, 1, 3))
    else:
        x = x.reshape(op.batch, op.seq_len, op.num_heads, op.head_size)
    x_rotate = x[..., : op.rotary_dim]
    x_not_rotate = x[..., op.rotary_dim :]
    if position_ids is not None:
        cos_cache = cos_cache[position_ids]
        sin_cache = sin_cache[position_ids]
    cos_cache = np.expand_dims(cos_cache, axis=2)
    sin_cache = np.expand_dims(sin_cache, axis=2)
    if op.interleaved:
        x1 = x_rotate[..., 0::2]
        x2 = x_rotate[..., 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)
    real = (cos_cache * x1) - (sin_cache * x2)
    imag = (sin_cache * x1) + (cos_cache * x2)
    if op.interleaved:
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        x_rotate_concat = np.concatenate((real, imag), axis=-1)
        x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)
    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if op.input_rank == 4:
        output = np.transpose(output, (0, 2, 1, 3))
    else:
        output = output.reshape(original_shape)
    evaluator.values[node.outputs[0]] = output


def _apply_lstm_activation(
    kind: int, value: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    if kind == ACTIVATION_KIND_BY_NAME["Relu"]:
        return np.maximum(value, 0)
    if kind == ACTIVATION_KIND_BY_NAME["Tanh"]:
        return np.tanh(value)
    if kind == ACTIVATION_KIND_BY_NAME["Sigmoid"]:
        return 1 / (1 + np.exp(-value))
    if kind == ACTIVATION_KIND_BY_NAME["Affine"]:
        return alpha * value + beta
    if kind == ACTIVATION_KIND_BY_NAME["LeakyRelu"]:
        return np.where(value < 0, alpha * value, value)
    if kind == ACTIVATION_KIND_BY_NAME["ThresholdedRelu"]:
        return np.where(value > alpha, value, 0)
    if kind == ACTIVATION_KIND_BY_NAME["ScaledTanh"]:
        return alpha * np.tanh(beta * value)
    if kind == ACTIVATION_KIND_BY_NAME["HardSigmoid"]:
        return np.clip(alpha * value + beta, 0, 1)
    if kind == ACTIVATION_KIND_BY_NAME["Elu"]:
        return np.where(value >= 0, value, alpha * (np.exp(value) - 1))
    if kind == ACTIVATION_KIND_BY_NAME["Softsign"]:
        return value / (1 + np.abs(value))
    if kind == ACTIVATION_KIND_BY_NAME["Softplus"]:
        return np.log1p(np.exp(value))
    raise UnsupportedOpError(f"Unsupported LSTM activation kind {kind}")


@register_evaluator("LSTM")
def _eval_lstm(evaluator: Evaluator, node: Node) -> None:
    spec = resolve_lstm_spec(evaluator.graph, node)
    inputs = evaluator.values
    x = inputs[spec.input_x]
    w = inputs[spec.input_w]
    r = inputs[spec.input_r]
    b = inputs[spec.input_b] if spec.input_b is not None else None
    sequence_lens = (
        inputs[spec.input_sequence_lens]
        if spec.input_sequence_lens is not None
        else None
    )
    initial_h = (
        inputs[spec.input_initial_h]
        if spec.input_initial_h is not None
        else None
    )
    initial_c = (
        inputs[spec.input_initial_c]
        if spec.input_initial_c is not None
        else None
    )
    p = inputs[spec.input_p] if spec.input_p is not None else None
    output_y, output_y_h, output_y_c = _apply_lstm(
        spec,
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        initial_c,
        p,
    )
    if spec.output_y is not None:
        evaluator.values[spec.output_y] = output_y
    if spec.output_y_h is not None:
        evaluator.values[spec.output_y_h] = output_y_h
    if spec.output_y_c is not None:
        evaluator.values[spec.output_y_c] = output_y_c


@register_evaluator("Conv")
def _eval_conv(evaluator: Evaluator, node: Node) -> None:
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "Conv supports float16, float, and double inputs only"
        )
    spec = resolve_conv_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    weights = evaluator.values[node.inputs[1]]
    bias = evaluator.values[node.inputs[2]] if len(node.inputs) > 2 else None
    evaluator.values[node.outputs[0]] = _apply_conv(spec, data, weights, bias)


@register_evaluator("ConvTranspose")
def _eval_conv_transpose(evaluator: Evaluator, node: Node) -> None:
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "ConvTranspose supports float16, float, and double inputs only"
        )
    spec = resolve_conv_transpose_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    weights = evaluator.values[node.inputs[1]]
    bias = evaluator.values[node.inputs[2]] if len(node.inputs) > 2 else None
    evaluator.values[node.outputs[0]] = _apply_conv_transpose(
        spec, data, weights, bias
    )


@register_evaluator("BatchNormalization")
def _eval_batch_norm(evaluator: Evaluator, node: Node) -> None:
    op = lower_batch_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    scale = evaluator.values[op.scale].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    bias = evaluator.values[op.bias].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    mean = evaluator.values[op.mean].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    variance = evaluator.values[op.variance].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    evaluator.values[op.output] = (
        (data - mean) / np.sqrt(variance + op.epsilon) * scale + bias
    )


@register_evaluator("LpNormalization")
def _eval_lp_normalization(evaluator: Evaluator, node: Node) -> None:
    op = lower_lp_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    if op.p == 1:
        denom = np.sum(np.abs(data), axis=op.axis, keepdims=True)
    else:
        denom = np.sqrt(np.sum(data * data, axis=op.axis, keepdims=True))
    evaluator.values[op.output] = data / denom


@register_evaluator("LpPool")
def _eval_lp_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_lp_pool(evaluator.graph, node)
    data = evaluator.values[op.input0]
    output = np.zeros(
        (op.batch, op.channels, op.out_h, op.out_w), dtype=data.dtype
    )
    for n in range(op.batch):
        for c in range(op.channels):
            for out_h in range(op.out_h):
                for out_w in range(op.out_w):
                    h_start = out_h * op.stride_h - op.pad_top
                    w_start = out_w * op.stride_w - op.pad_left
                    acc = 0.0
                    for kh in range(op.kernel_h):
                        for kw in range(op.kernel_w):
                            in_h = h_start + kh
                            in_w = w_start + kw
                            if (
                                0 <= in_h < op.in_h
                                and 0 <= in_w < op.in_w
                            ):
                                value = data[(n, c, in_h, in_w)]
                                acc += abs(value) ** op.p
                    output[(n, c, out_h, out_w)] = acc ** (1.0 / op.p)
    evaluator.values[op.output] = output


@register_evaluator("QuantizeLinear")
def _eval_quantize_linear(evaluator: Evaluator, node: Node) -> None:
    spec = resolve_quantize_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    scale = evaluator.values[node.inputs[1]]
    zero_point_name = optional_name(node.inputs, 2)
    if zero_point_name is None:
        zero_point = 0
    else:
        zero_point = evaluator.values[zero_point_name]
    if spec.axis is None:
        scaled = data / scale + zero_point
    else:
        shape = [1] * data.ndim
        shape[spec.axis] = scale.shape[0]
        scaled = data / scale.reshape(shape) + np.asarray(zero_point).reshape(
            shape
        )
    rounded = np.rint(scaled)
    info = np.iinfo(spec.output_dtype.np_dtype)
    clipped = np.clip(rounded, info.min, info.max)
    evaluator.values[node.outputs[0]] = clipped.astype(
        spec.output_dtype.np_dtype, copy=False
    )

@register_evaluator("InstanceNormalization")
def _eval_instance_normalization(evaluator: Evaluator, node: Node) -> None:
    op = lower_instance_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    axes = tuple(range(2, data.ndim))
    mean = np.mean(data, axis=axes, keepdims=True)
    var = np.mean((data - mean) ** 2, axis=axes, keepdims=True)
    scale = evaluator.values[op.scale].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    bias = evaluator.values[op.bias].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    evaluator.values[op.output] = (
        (data - mean) / np.sqrt(var + op.epsilon) * scale + bias
    )


@register_evaluator("GroupNormalization")
def _eval_group_normalization(evaluator: Evaluator, node: Node) -> None:
    op = lower_group_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    batch = data.shape[0]
    spatial_shape = data.shape[2:]
    grouped = data.reshape(
        (batch, op.num_groups, op.group_size) + spatial_shape
    )
    axes = tuple(range(2, grouped.ndim))
    mean = np.mean(grouped, axis=axes, keepdims=True)
    var = np.mean((grouped - mean) ** 2, axis=axes, keepdims=True)
    normalized = (grouped - mean) / np.sqrt(var + op.epsilon)
    normalized = normalized.reshape(data.shape)
    scale = evaluator.values[op.scale].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    bias = evaluator.values[op.bias].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    evaluator.values[op.output] = normalized * scale + bias


@register_evaluator("LayerNormalization")
def _eval_layer_normalization(evaluator: Evaluator, node: Node) -> None:
    op = lower_layer_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    axes = tuple(range(op.axis, data.ndim))
    mean = np.mean(data, axis=axes, keepdims=True)
    var = np.mean((data - mean) ** 2, axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + op.epsilon)
    normalized = (data - mean) * inv_std
    scale = evaluator.values[op.scale].reshape(
        (1,) * op.axis + evaluator.values[op.scale].shape
    )
    normalized = normalized * scale
    if op.bias is not None:
        bias = evaluator.values[op.bias].reshape(
            (1,) * op.axis + evaluator.values[op.bias].shape
        )
        normalized = normalized + bias
    evaluator.values[op.output] = normalized
    if op.mean_output is not None:
        evaluator.values[op.mean_output] = mean
    if op.invstd_output is not None:
        evaluator.values[op.invstd_output] = inv_std


@register_evaluator("MeanVarianceNormalization")
def _eval_mean_variance_normalization(
    evaluator: Evaluator, node: Node
) -> None:
    op = lower_mean_variance_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    mean = np.mean(data, axis=op.axes, keepdims=True)
    variance = np.mean((data - mean) ** 2, axis=op.axes, keepdims=True)
    evaluator.values[op.output] = (data - mean) / np.sqrt(
        variance + op.epsilon
    )


@register_evaluator("RMSNormalization")
def _eval_rms_normalization(evaluator: Evaluator, node: Node) -> None:
    op = lower_rms_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    axes = tuple(range(op.axis, data.ndim))
    mean_square = np.mean(data * data, axis=axes, keepdims=True)
    rms = np.sqrt(mean_square + op.epsilon)
    normalized = data / rms
    scale = evaluator.values[op.scale].reshape(
        (1,) * op.axis + evaluator.values[op.scale].shape
    )
    evaluator.values[op.output] = normalized * scale


@register_evaluator("LRN")
def _eval_lrn(evaluator: Evaluator, node: Node) -> None:
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "LRN supports float16, float, and double inputs only"
        )
    spec = resolve_lrn_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_lrn(spec, data)


@register_evaluator("AveragePool")
def _eval_average_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_average_pool(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_average_pool(op, data)


@register_evaluator("GlobalAveragePool")
def _eval_global_average_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_global_average_pool(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_average_pool(op, data)


@register_evaluator("MaxPool")
def _eval_maxpool(evaluator: Evaluator, node: Node) -> None:
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    indices_output = node.outputs[1] if len(node.outputs) > 1 else None
    if indices_output is not None:
        indices_dtype = value_dtype(evaluator.graph, indices_output, node)
        if indices_dtype != ScalarType.I64:
            raise UnsupportedOpError("MaxPool indices output must be int64")
    if op_dtype == ScalarType.BOOL:
        raise UnsupportedOpError("MaxPool supports numeric inputs only")
    spec = resolve_maxpool_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    if indices_output is None:
        evaluator.values[node.outputs[0]] = _apply_maxpool(spec, data)
    else:
        values, indices = _apply_maxpool(spec, data, return_indices=True)
        evaluator.values[node.outputs[0]] = values
        evaluator.values[indices_output] = indices


@register_evaluator("GlobalMaxPool")
def _eval_global_max_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_global_max_pool(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    if not op.axes:
        evaluator.values[node.outputs[0]] = value.copy()
        return
    evaluator.values[node.outputs[0]] = np.max(
        value, axis=op.axes, keepdims=op.keepdims
    )


@register_evaluator("Softmax")
def _eval_softmax(evaluator: Evaluator, node: Node) -> None:
    op = lower_softmax(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_softmax(value, op.axis)


@register_evaluator("LogSoftmax")
def _eval_logsoftmax(evaluator: Evaluator, node: Node) -> None:
    op = lower_logsoftmax(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_logsoftmax(value, op.axis)


@register_evaluator("Hardmax")
def _eval_hardmax(evaluator: Evaluator, node: Node) -> None:
    op = lower_hardmax(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    max_values = np.max(value, axis=op.axis, keepdims=True)
    is_max = value == max_values
    max_index = np.argmax(is_max, axis=op.axis)
    output = np.zeros_like(value)
    ones = np.array(1.0, dtype=value.dtype)
    np.put_along_axis(output, np.expand_dims(max_index, axis=op.axis), ones, axis=op.axis)
    evaluator.values[node.outputs[0]] = output


@register_evaluator("NegativeLogLikelihoodLoss")
def _eval_negative_log_likelihood_loss(
    evaluator: Evaluator, node: Node
) -> None:
    op = lower_negative_log_likelihood_loss(evaluator.graph, node)
    input_value = evaluator.values[op.input0]
    target_value = evaluator.values[op.target]
    weight_value = evaluator.values[op.weight] if op.weight is not None else None
    evaluator.values[op.output] = _apply_negative_log_likelihood_loss(
        input_value,
        target_value,
        weight_value,
        reduction=op.reduction,
        ignore_index=op.ignore_index,
    )


@register_evaluator("SoftmaxCrossEntropyLoss")
def _eval_softmax_cross_entropy_loss(
    evaluator: Evaluator, node: Node
) -> None:
    op = lower_softmax_cross_entropy_loss(evaluator.graph, node)
    input_value = evaluator.values[op.input0]
    target_value = evaluator.values[op.target]
    weight_value = evaluator.values[op.weight] if op.weight is not None else None
    loss, log_prob = _apply_softmax_cross_entropy_loss(
        input_value,
        target_value,
        weight_value,
        reduction=op.reduction,
        ignore_index=op.ignore_index,
        return_log_prob=op.log_prob is not None,
    )
    evaluator.values[op.output] = loss
    if op.log_prob is not None and log_prob is not None:
        evaluator.values[op.log_prob] = log_prob


@register_evaluator("Dropout")
def _eval_dropout(evaluator: Evaluator, node: Node) -> None:
    op = lower_dropout(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].copy()


@register_evaluator("Concat")
def _eval_concat(evaluator: Evaluator, node: Node) -> None:
    op = lower_concat(evaluator.graph, node)
    tensors = [evaluator.values[name] for name in node.inputs]
    evaluator.values[op.output] = np.concatenate(tensors, axis=op.axis)


@register_evaluator("Transpose")
def _eval_transpose(evaluator: Evaluator, node: Node) -> None:
    op = lower_transpose(evaluator.graph, node)
    evaluator.values[op.output] = np.transpose(
        evaluator.values[op.input0], axes=tuple(op.perm)
    )


@register_evaluator("Unsqueeze")
def _eval_unsqueeze(evaluator: Evaluator, node: Node) -> None:
    op = lower_unsqueeze(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("Squeeze")
def _eval_squeeze(evaluator: Evaluator, node: Node) -> None:
    op = lower_squeeze(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("Reshape")
def _eval_reshape(evaluator: Evaluator, node: Node) -> None:
    op = lower_reshape(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("Flatten")
def _eval_flatten(evaluator: Evaluator, node: Node) -> None:
    op = lower_flatten(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("ConstantOfShape")
def _eval_constant_of_shape(evaluator: Evaluator, node: Node) -> None:
    op = lower_constant_of_shape(evaluator.graph, node)
    evaluator.values[op.output] = np.full(
        op.shape, op.value, dtype=op.dtype.np_dtype
    )


@register_evaluator("Shape")
def _eval_shape(evaluator: Evaluator, node: Node) -> None:
    op = lower_shape(evaluator.graph, node)
    evaluator.values[op.output] = np.array(op.values, dtype=np.int64)


@register_evaluator("Size")
def _eval_size(evaluator: Evaluator, node: Node) -> None:
    op = lower_size(evaluator.graph, node)
    evaluator.values[op.output] = np.array(op.value, dtype=np.int64)


@register_evaluator("NonZero")
def _eval_nonzero(evaluator: Evaluator, node: Node) -> None:
    op = lower_nonzero(evaluator.graph, node)
    values = evaluator.values[op.input0]
    indices = np.nonzero(values)
    evaluator.values[op.output] = np.stack(indices, axis=0).astype(
        np.int64, copy=False
    )


@register_evaluator("Expand")
def _eval_expand(evaluator: Evaluator, node: Node) -> None:
    op = lower_expand(evaluator.graph, node)
    value = evaluator.values[op.input0]
    evaluator.values[op.output] = np.broadcast_to(
        value, op.output_shape
    ).copy()


@register_evaluator("Range")
def _eval_range(evaluator: Evaluator, node: Node) -> None:
    op = lower_range(evaluator.graph, node)
    start_value = evaluator.values[op.start].reshape(-1)[0]
    delta_value = evaluator.values[op.delta].reshape(-1)[0]
    indices = np.arange(op.length, dtype=op.dtype.np_dtype)
    output = start_value + indices * delta_value
    evaluator.values[op.output] = output


@register_evaluator("OneHot")
def _eval_onehot(evaluator: Evaluator, node: Node) -> None:
    op = lower_onehot(evaluator.graph, node)
    indices = evaluator.values[op.indices].astype(np.int64, copy=False)
    depth_values = evaluator.values[op.depth].reshape(-1)
    if depth_values.size != 1:
        raise UnsupportedOpError("OneHot depth input must be a scalar")
    depth_value = int(depth_values[0])
    if depth_value < 0:
        raise UnsupportedOpError("OneHot depth must be non-negative")
    values = evaluator.values[op.values].reshape(-1)
    if values.size != 2:
        raise UnsupportedOpError("OneHot values input must have 2 elements")
    off_value, on_value = values[0], values[1]
    if depth_value == 0:
        evaluator.values[op.output] = np.full(
            op.output_shape, off_value, dtype=values.dtype
        )
        return
    axis = op.axis
    rank = indices.ndim
    if axis < 0:
        axis += rank + 1
    depth_range = np.arange(depth_value, dtype=np.int64)
    new_shape = (1,) * axis + (depth_value,) + (1,) * (rank - axis)
    targets = depth_range.reshape(new_shape)
    adjusted = np.mod(indices, depth_value) if depth_value > 0 else indices
    values_reshaped = np.reshape(
        adjusted, indices.shape[:axis] + (1,) + indices.shape[axis:]
    )
    valid_mask = (indices >= -depth_value) & (indices < depth_value)
    valid_mask = np.reshape(
        valid_mask, indices.shape[:axis] + (1,) + indices.shape[axis:]
    )
    one_hot = (targets == values_reshaped) & valid_mask
    output = np.where(one_hot, on_value, off_value).astype(values.dtype)
    evaluator.values[op.output] = output


@register_evaluator("Split")
def _eval_split(evaluator: Evaluator, node: Node) -> None:
    op = lower_split(evaluator.graph, node)
    data = evaluator.values[op.input0]
    split_points = np.cumsum(op.split_sizes)[:-1]
    outputs = np.split(data, split_points, axis=op.axis)
    for output_name, output_value in zip(op.outputs, outputs):
        evaluator.values[output_name] = output_value


@register_evaluator("ReduceMean")
@register_evaluator("ReduceSum")
@register_evaluator("ReduceMax")
@register_evaluator("ReduceMin")
@register_evaluator("ReduceProd")
@register_evaluator("ReduceL1")
@register_evaluator("ReduceL2")
@register_evaluator("ReduceLogSum")
@register_evaluator("ReduceLogSumExp")
@register_evaluator("ReduceSumSquare")
def _eval_reduce(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 or 2 inputs and 1 output"
        )
    op_dtype = value_dtype(evaluator.graph, node.inputs[0], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if (
        node.op_type in REDUCE_OUTPUTS_FLOAT_ONLY
        and not op_dtype.is_float
    ):
        raise UnsupportedOpError(
            f"{node.op_type} supports float16, float, and double inputs only"
        )
    value = evaluator.values[node.inputs[0]]
    input_shape = value.shape
    if len(node.inputs) > 1 and node.inputs[1]:
        axes_value = evaluator.values[node.inputs[1]]
        if axes_value.dtype.type not in {np.int32, np.int64}:
            raise UnsupportedOpError(
                f"{node.op_type} axes input must be int64 or int32"
            )
        axes = tuple(int(axis) for axis in axes_value.ravel())
        noop_with_empty_axes = bool(int(node.attrs.get("noop_with_empty_axes", 0)))
        if not axes:
            if noop_with_empty_axes:
                evaluator.values[node.outputs[0]] = value.copy()
                return
            axes = tuple(range(len(input_shape)))
        axes = normalize_reduce_axes(axes, input_shape, node)
    else:
        axes_spec, noop = resolve_reduce_axes(evaluator.graph, node, input_shape)
        if noop:
            evaluator.values[node.outputs[0]] = value.copy()
            return
        if axes_spec is None or axes_spec.axes is None:
            raise UnsupportedOpError(
                f"{node.op_type} axes input must be constant for evaluator"
            )
        axes = axes_spec.axes
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    reduce_kind = REDUCE_KIND_BY_OP[node.op_type]
    if reduce_kind == "sum":
        result = np.sum(value, axis=axes, keepdims=keepdims)
    elif reduce_kind == "mean":
        result = np.mean(value, axis=axes, keepdims=keepdims)
    elif reduce_kind == "max":
        result = np.max(value, axis=axes, keepdims=keepdims)
    elif reduce_kind == "min":
        result = np.min(value, axis=axes, keepdims=keepdims)
    elif reduce_kind == "prod":
        result = np.prod(value, axis=axes, keepdims=keepdims)
    elif reduce_kind == "l1":
        result = np.sum(np.abs(value), axis=axes, keepdims=keepdims)
    elif reduce_kind == "l2":
        result = np.sqrt(np.sum(value * value, axis=axes, keepdims=keepdims))
    elif reduce_kind == "logsum":
        result = np.log(np.sum(value, axis=axes, keepdims=keepdims))
    elif reduce_kind == "logsumexp":
        result = np.log(np.sum(np.exp(value), axis=axes, keepdims=keepdims))
    elif reduce_kind == "sumsquare":
        result = np.sum(value * value, axis=axes, keepdims=keepdims)
    else:
        raise UnsupportedOpError(f"Unsupported reduce kind {reduce_kind}")
    evaluator.values[node.outputs[0]] = result


@register_evaluator("ArgMax")
@register_evaluator("ArgMin")
def _eval_arg_reduce(evaluator: Evaluator, node: Node) -> None:
    op = lower_arg_reduce(evaluator.graph, node)
    value = evaluator.values[op.input0]
    if op.select_last_index:
        flipped = np.flip(value, axis=op.axis)
        if op.reduce_kind == "max":
            indices = np.argmax(flipped, axis=op.axis)
        elif op.reduce_kind == "min":
            indices = np.argmin(flipped, axis=op.axis)
        else:
            raise UnsupportedOpError(
                f"Unsupported arg reduce kind {op.reduce_kind}"
            )
        indices = value.shape[op.axis] - 1 - indices
    else:
        if op.reduce_kind == "max":
            indices = np.argmax(value, axis=op.axis)
        elif op.reduce_kind == "min":
            indices = np.argmin(value, axis=op.axis)
        else:
            raise UnsupportedOpError(
                f"Unsupported arg reduce kind {op.reduce_kind}"
            )
    if op.keepdims:
        indices = np.expand_dims(indices, axis=op.axis)
    evaluator.values[op.output] = indices.astype(op.output_dtype.np_dtype)


@register_evaluator("TopK")
def _eval_topk(evaluator: Evaluator, node: Node) -> None:
    op = lower_topk(evaluator.graph, node)
    value = evaluator.values[op.input0]
    moved = np.moveaxis(value, op.axis, -1)
    axis_dim = moved.shape[-1]
    flat = moved.reshape(-1, axis_dim)
    values_out = np.empty((flat.shape[0], op.k), dtype=value.dtype)
    indices_out = np.empty((flat.shape[0], op.k), dtype=np.int64)
    for row_index in range(flat.shape[0]):
        row = flat[row_index]
        order = sorted(
            range(axis_dim),
            key=lambda idx: (
                -row[idx].item() if op.largest else row[idx].item(),
                idx,
            ),
        )
        topk = order[: op.k]
        indices_out[row_index] = topk
        values_out[row_index] = row[topk]
    values_out = values_out.reshape(moved.shape[:-1] + (op.k,))
    indices_out = indices_out.reshape(moved.shape[:-1] + (op.k,))
    values_out = np.moveaxis(values_out, -1, op.axis)
    indices_out = np.moveaxis(indices_out, -1, op.axis)
    evaluator.values[op.output_values] = values_out.astype(
        op.output_values_dtype.np_dtype
    )
    evaluator.values[op.output_indices] = indices_out.astype(
        op.output_indices_dtype.np_dtype
    )


def _eval_binary_unary(evaluator: Evaluator, node: Node) -> None:
    if node.op_type == "BitShift":
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("BitShift must have 2 inputs and 1 output")
        direction_attr = node.attrs.get("direction", "LEFT")
        if isinstance(direction_attr, bytes):
            direction = direction_attr.decode()
        else:
            direction = str(direction_attr)
        if direction not in {"LEFT", "RIGHT"}:
            raise UnsupportedOpError(
                "BitShift direction must be LEFT or RIGHT"
            )
        op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
        if not op_dtype.is_integer:
            raise UnsupportedOpError("BitShift expects integer inputs")
        function = (
            ScalarFunction.BITWISE_LEFT_SHIFT
            if direction == "LEFT"
            else ScalarFunction.BITWISE_RIGHT_SHIFT
        )
        op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
        if op_spec is None:
            raise UnsupportedOpError("Unsupported op BitShift")
        left = evaluator.values[node.inputs[0]]
        right = evaluator.values[node.inputs[1]]
        evaluator.values[node.outputs[0]] = apply_binary_op(
            op_spec, left, right
        )
        return
    if node.op_type == "Mod":
        fmod = int(node.attrs.get("fmod", 0))
        if fmod not in {0, 1}:
            raise UnsupportedOpError("Mod only supports fmod=0 or fmod=1")
        function = (
            ScalarFunction.FMOD if fmod == 1 else ScalarFunction.REMAINDER
        )
    else:
        try:
            function = ScalarFunction.from_onnx_op(node.op_type)
        except ScalarFunctionError as exc:
            raise UnsupportedOpError(
                f"Unsupported op {node.op_type}"
            ) from exc
    validate_unary_attrs(node.op_type, node.attrs)
    if function in COMPARE_FUNCTIONS:
        input_dtype = node_dtype(evaluator.graph, node, *node.inputs)
        output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{node.op_type} expects bool output, got {output_dtype.onnx_name}"
            )
        op_spec = binary_op_symbol(function, node.attrs, dtype=input_dtype)
        if op_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        left = evaluator.values[node.inputs[0]]
        right = evaluator.values[node.inputs[1]]
        evaluator.values[node.outputs[0]] = apply_binary_op(
            op_spec, left, right
        )
        return
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
    unary_symbol = unary_op_symbol(function, dtype=op_dtype)
    if op_spec is None and unary_symbol is None:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if op_spec is not None:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        left = evaluator.values[node.inputs[0]]
        right = evaluator.values[node.inputs[1]]
        evaluator.values[node.outputs[0]] = apply_binary_op(
            op_spec, left, right
        )
        return
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 input and 1 output"
        )
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = apply_unary_op(
        function, value, dtype=op_dtype
    )


def _apply_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim < 1 or right.ndim < 1:
        raise UnsupportedOpError(
            "MatMul inputs must be at least 1D, "
            f"got {left.shape} x {right.shape}"
        )
    left_dim = left.shape[-1]
    right_dim = right.shape[0] if right.ndim == 1 else right.shape[-2]
    if left_dim != right_dim:
        raise ShapeInferenceError(
            "MatMul inner dimensions must match, "
            f"got {left_dim} and {right_dim}"
        )
    left_batch = left.shape[:-2] if left.ndim > 1 else ()
    right_batch = right.shape[:-2] if right.ndim > 1 else ()
    if not _matmul_batch_broadcastable(left_batch, right_batch):
        raise ShapeInferenceError(
            "MatMul batch dimensions must be broadcastable, "
            f"got {left_batch} x {right_batch}"
        )
    return np.matmul(left, right)


def _matmul_batch_broadcastable(
    left: tuple[int, ...], right: tuple[int, ...]
) -> bool:
    max_rank = max(len(left), len(right))
    left_padded = (1,) * (max_rank - len(left)) + left
    right_padded = (1,) * (max_rank - len(right)) + right
    for left_dim, right_dim in zip(left_padded, right_padded):
        if not (left_dim == right_dim or left_dim == 1 or right_dim == 1):
            return False
    return True


def _apply_softmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values - max_values)
    sum_values = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / sum_values


def _apply_logsoftmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    logsum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - logsum


def _apply_negative_log_likelihood_loss(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int,
) -> np.ndarray:
    input_shape = values.shape
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss input must be at least 2D"
        )
    target_shape = target.shape
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target spatial dimensions must match input"
        )
    n = input_shape[0]
    c = input_shape[1]
    if weight is not None:
        gather_weight = np.take(weight, target.astype(np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=values.dtype
            )
    elif ignore_index != -1:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(
            dtype=values.dtype
        )
    else:
        gather_weight = None
    if len(input_shape) != 3:
        values = values.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = values.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if target[i][d_index] != ignore_index:
                loss[i][d_index] = -values[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            weight_sum = gather_weight.sum()
            if weight_sum == 0:
                return np.array(0, dtype=values.dtype)
            loss = loss.sum() / weight_sum
            return loss.astype(values.dtype)
    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    return loss.astype(values.dtype)


def _apply_softmax_cross_entropy_loss(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int | None,
    return_log_prob: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    input_shape = values.shape
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss input must be at least 2D"
        )
    target_shape = target.shape
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target spatial dimensions must match input"
        )
    log_prob = _apply_logsoftmax(values, axis=1)
    log_prob_output = log_prob if return_log_prob else None
    if weight is not None:
        gather_weight = np.take(weight, target.astype(np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=values.dtype
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(
            dtype=values.dtype
        )
    else:
        gather_weight = None
    n = input_shape[0]
    c = input_shape[1]
    if len(input_shape) != 3:
        log_prob = log_prob.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = log_prob.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if ignore_index is None or target[i][d_index] != ignore_index:
                loss[i][d_index] = -log_prob[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            loss = loss.astype(values.dtype)
            if return_log_prob:
                return loss, log_prob.astype(values.dtype)
            return loss, None
    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    loss = loss.astype(values.dtype)
    if return_log_prob and log_prob_output is not None:
        return loss, log_prob_output.astype(values.dtype)
    return loss, None


def _apply_attention(
    spec,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: np.ndarray | None,
    past_key: np.ndarray | None,
    past_value: np.ndarray | None,
    nonpad_kv_seqlen: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if spec.q_rank == 3:
        query_4d = query.reshape(
            spec.batch, spec.q_seq, spec.q_heads, spec.qk_head_size
        ).transpose(0, 2, 1, 3)
        key_4d = key.reshape(
            spec.batch, spec.kv_seq, spec.kv_heads, spec.qk_head_size
        ).transpose(0, 2, 1, 3)
        value_4d = value.reshape(
            spec.batch, spec.kv_seq, spec.kv_heads, spec.v_head_size
        ).transpose(0, 2, 1, 3)
    else:
        query_4d = query
        key_4d = key
        value_4d = value
    if past_key is not None and past_value is not None:
        key_total = np.concatenate([past_key, key_4d], axis=2)
        value_total = np.concatenate([past_value, value_4d], axis=2)
    else:
        key_total = key_4d
        value_total = value_4d
    if spec.head_group_size > 1:
        key_total_expanded = np.repeat(key_total, spec.head_group_size, axis=1)
        value_total_expanded = np.repeat(
            value_total, spec.head_group_size, axis=1
        )
    else:
        key_total_expanded = key_total
        value_total_expanded = value_total
    k_transpose = np.transpose(key_total_expanded, (0, 1, 3, 2))
    scores = np.matmul(query_4d, k_transpose) * spec.scale
    bias = np.zeros_like(scores)
    if spec.has_attn_mask and attn_mask is not None:
        if spec.mask_is_bool:
            bias_mask = np.where(attn_mask, 0.0, -np.inf)
        else:
            bias_mask = attn_mask.astype(scores.dtype)
        if spec.mask_rank == 2:
            bias_mask = bias_mask[None, None, ...]
        elif spec.mask_rank == 3:
            bias_mask = bias_mask[:, None, ...]
        bias_mask = np.broadcast_to(
            bias_mask, (spec.batch, spec.q_heads, spec.q_seq, spec.mask_kv_seq)
        )
        if spec.mask_kv_seq < spec.total_seq:
            pad_width = spec.total_seq - spec.mask_kv_seq
            bias_mask = np.pad(
                bias_mask,
                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                constant_values=-np.inf,
            )
        bias = bias + bias_mask
    if spec.has_nonpad and nonpad_kv_seqlen is not None:
        kv_range = np.arange(spec.total_seq)[None, None, None, :]
        valid = kv_range < nonpad_kv_seqlen[:, None, None, None]
        bias = bias + np.where(valid, 0.0, -np.inf)
    if spec.is_causal:
        kv_range = np.arange(spec.total_seq)[None, :]
        q_range = np.arange(spec.q_seq)[:, None] + spec.past_seq
        causal_mask = kv_range > q_range
        bias = bias + np.where(causal_mask, -np.inf, 0.0)[None, None, :, :]
    scores_with_bias = scores + bias
    if spec.softcap != 0.0:
        scores_softcap = spec.softcap * np.tanh(scores_with_bias / spec.softcap)
    else:
        scores_softcap = scores_with_bias
    max_scores = np.max(scores_softcap, axis=-1, keepdims=True)
    weights = np.exp(scores_softcap - max_scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    output = np.matmul(weights, value_total_expanded)
    if spec.q_rank == 3:
        output = output.transpose(0, 2, 1, 3).reshape(
            spec.batch, spec.q_seq, spec.q_heads * spec.v_head_size
        )
    qk_output = None
    if spec.qk_matmul_output_mode == 0:
        qk_output = scores
    elif spec.qk_matmul_output_mode == 1:
        qk_output = scores_with_bias
    elif spec.qk_matmul_output_mode == 2:
        qk_output = scores_softcap
    else:
        qk_output = weights
    return output, key_total, value_total, qk_output


def _apply_conv(
    spec, data: np.ndarray, weights: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    output = np.zeros(
        (spec.batch, spec.out_channels, *spec.out_spatial),
        dtype=data.dtype,
    )
    pad_begin = spec.pads[: spec.spatial_rank]
    group_in_channels = spec.in_channels // spec.group
    group_out_channels = spec.out_channels // spec.group
    for n in range(spec.batch):
        for g in range(spec.group):
            oc_base = g * group_out_channels
            ic_base = g * group_in_channels
            for oc in range(group_out_channels):
                oc_global = oc_base + oc
                base = bias[oc_global] if bias is not None else 0.0
                for out_index in np.ndindex(*spec.out_spatial):
                    acc = base
                    for ic in range(group_in_channels):
                        ic_global = ic_base + ic
                        for kernel_index in np.ndindex(*spec.kernel_shape):
                            in_index = []
                            valid = True
                            for (
                                out_dim,
                                kernel_dim,
                                stride,
                                dilation,
                                pad,
                                in_size,
                            ) in zip(
                                out_index,
                                kernel_index,
                                spec.strides,
                                spec.dilations,
                                pad_begin,
                                spec.in_spatial,
                            ):
                                in_dim = out_dim * stride + kernel_dim * dilation - pad
                                if in_dim < 0 or in_dim >= in_size:
                                    valid = False
                                    break
                                in_index.append(in_dim)
                            if valid:
                                acc += data[(n, ic_global, *in_index)] * weights[
                                    (oc_global, ic, *kernel_index)
                                ]
                    output[(n, oc_global, *out_index)] = acc
    return output


def _apply_conv_transpose(
    spec, data: np.ndarray, weights: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    output = np.zeros(
        (spec.batch, spec.out_channels, *spec.out_spatial), dtype=data.dtype
    )
    if bias is not None:
        output += bias.reshape((1, spec.out_channels) + (1,) * spec.spatial_rank)
    pad_begin = spec.pads[: spec.spatial_rank]
    group_in_channels = spec.in_channels // spec.group
    group_out_channels = spec.out_channels // spec.group
    for n in range(spec.batch):
        for g in range(spec.group):
            oc_base = g * group_out_channels
            ic_base = g * group_in_channels
            for ic in range(group_in_channels):
                ic_global = ic_base + ic
                for in_index in np.ndindex(*spec.in_spatial):
                    value = data[(n, ic_global, *in_index)]
                    for oc in range(group_out_channels):
                        oc_global = oc_base + oc
                        for kernel_index in np.ndindex(*spec.kernel_shape):
                            out_index = []
                            valid = True
                            for (
                                in_dim,
                                kernel_dim,
                                stride,
                                dilation,
                                pad,
                                out_size,
                            ) in zip(
                                in_index,
                                kernel_index,
                                spec.strides,
                                spec.dilations,
                                pad_begin,
                                spec.out_spatial,
                            ):
                                out_dim = (
                                    in_dim * stride + kernel_dim * dilation - pad
                                )
                                if out_dim < 0 or out_dim >= out_size:
                                    valid = False
                                    break
                                out_index.append(out_dim)
                            if valid:
                                output[(n, oc_global, *out_index)] += (
                                    value * weights[(ic_global, oc, *kernel_index)]
                                )
    return output


def _apply_lrn(spec, data: np.ndarray) -> np.ndarray:
    output = np.empty_like(data)
    spatial_shape = spec.shape[2:]
    spatial_indices = [()]
    if spatial_shape:
        spatial_indices = list(np.ndindex(*spatial_shape))
    for n in range(spec.shape[0]):
        for c in range(spec.channels):
            start = max(0, c - spec.half)
            end = min(spec.channels - 1, c + spec.half)
            for index in spatial_indices:
                sum_val = 0.0
                for i in range(start, end + 1):
                    value = data[(n, i, *index)]
                    sum_val += value * value
                scale = spec.bias + (spec.alpha / spec.size) * sum_val
                output[(n, c, *index)] = data[(n, c, *index)] / math.pow(
                    scale, spec.beta
                )
    return output


def _apply_average_pool(op, data: np.ndarray) -> np.ndarray:
    output = np.zeros((op.batch, op.channels, op.out_h, op.out_w), dtype=data.dtype)
    for n in range(op.batch):
        for c in range(op.channels):
            for oh in range(op.out_h):
                for ow in range(op.out_w):
                    acc = 0.0
                    count = 0
                    for kh in range(op.kernel_h):
                        ih = oh * op.stride_h + kh - op.pad_top
                        if ih < 0 or ih >= op.in_h:
                            if op.count_include_pad:
                                count += op.kernel_w
                        else:
                            for kw in range(op.kernel_w):
                                iw = ow * op.stride_w + kw - op.pad_left
                                if iw < 0 or iw >= op.in_w:
                                    if op.count_include_pad:
                                        count += 1
                                else:
                                    acc += data[n, c, ih, iw]
                                    count += 1
                    output[n, c, oh, ow] = 0.0 if count == 0 else acc / float(count)
    return output


def _maxpool_min_value(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.floating):
        return -np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    raise UnsupportedOpError("MaxPool supports numeric inputs only")


def _apply_maxpool(
    spec, data: np.ndarray, *, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    min_value = _maxpool_min_value(data.dtype)
    output = np.full(
        (spec.batch, spec.channels, *spec.out_spatial),
        min_value,
        dtype=data.dtype,
    )
    indices = (
        np.zeros((spec.batch, spec.channels, *spec.out_spatial), dtype=np.int64)
        if return_indices
        else None
    )
    pad_begin = spec.pads[: spec.spatial_rank]
    for n in range(spec.batch):
        for c in range(spec.channels):
            for out_index in np.ndindex(*spec.out_spatial):
                max_value = min_value
                max_index = 0
                has_value = False
                for kernel_index in np.ndindex(*spec.kernel_shape):
                    in_index = []
                    valid = True
                    for out_dim, kernel_dim, stride, dilation, pad in zip(
                        out_index,
                        kernel_index,
                        spec.strides,
                        spec.dilations,
                        pad_begin,
                    ):
                        idx = out_dim * stride + kernel_dim * dilation - pad
                        if idx < 0 or idx >= spec.in_spatial[len(in_index)]:
                            valid = False
                            break
                        in_index.append(idx)
                    if valid:
                        value = data[(n, c, *in_index)]
                        if value > max_value or not has_value:
                            max_value = value
                            has_value = True
                            if return_indices:
                                linear_index = n * spec.channels + c
                                if spec.storage_order == 0:
                                    for idx, size in zip(
                                        in_index, spec.in_spatial
                                    ):
                                        linear_index = linear_index * size + idx
                                else:
                                    spatial_index = 0
                                    spatial_stride = 1
                                    for idx, size in zip(
                                        in_index, spec.in_spatial
                                    ):
                                        spatial_index += idx * spatial_stride
                                        spatial_stride *= size
                                    linear_index = (
                                        linear_index * spatial_stride + spatial_index
                                    )
                                max_index = linear_index
                output[(n, c, *out_index)] = max_value
                if return_indices and indices is not None:
                    indices[(n, c, *out_index)] = max_index
    if return_indices:
        if indices is None:
            raise RuntimeError("MaxPool indices were not computed")
        return output, indices
    return output


def _apply_lstm(
    spec,
    x: np.ndarray,
    w: np.ndarray,
    r: np.ndarray,
    b: np.ndarray | None,
    sequence_lens: np.ndarray | None,
    initial_h: np.ndarray | None,
    initial_c: np.ndarray | None,
    p: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if spec.layout == 1:
        x = np.swapaxes(x, 0, 1)
    seq_length = spec.seq_length
    batch_size = spec.batch_size
    hidden_size = spec.hidden_size
    num_directions = spec.num_directions
    if sequence_lens is None:
        sequence_lens = np.full((batch_size,), seq_length, dtype=np.int64)
    else:
        sequence_lens = sequence_lens.astype(np.int64, copy=False)
    if b is None:
        b = np.zeros((num_directions, 8 * hidden_size), dtype=x.dtype)
    if p is None:
        p = np.zeros((num_directions, 3 * hidden_size), dtype=x.dtype)
    if initial_h is None:
        initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
    if initial_c is None:
        initial_c = np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
    if spec.layout == 1:
        initial_h = np.swapaxes(initial_h, 0, 1)
        initial_c = np.swapaxes(initial_c, 0, 1)
    output_y = None
    if spec.output_y is not None:
        output_y = np.zeros(
            (seq_length, num_directions, batch_size, hidden_size), dtype=x.dtype
        )
    output_y_h = (
        np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
        if spec.output_y_h is not None
        else None
    )
    output_y_c = (
        np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
        if spec.output_y_c is not None
        else None
    )
    directions = (
        ("forward", "reverse")
        if spec.direction == "bidirectional"
        else (spec.direction,)
    )
    for dir_index, dir_kind in enumerate(directions):
        w_dir = w[dir_index]
        r_dir = r[dir_index]
        b_dir = b[dir_index]
        bias = b_dir[: 4 * hidden_size] + b_dir[4 * hidden_size :]
        p_dir = p[dir_index]
        p_i = p_dir[:hidden_size]
        p_o = p_dir[hidden_size : 2 * hidden_size]
        p_f = p_dir[2 * hidden_size :]
        h_prev = initial_h[dir_index].copy()
        c_prev = initial_c[dir_index].copy()
        act_offset = dir_index * 3
        act_f = spec.activation_kinds[act_offset]
        act_g = spec.activation_kinds[act_offset + 1]
        act_h = spec.activation_kinds[act_offset + 2]
        alpha_f = spec.activation_alphas[act_offset]
        alpha_g = spec.activation_alphas[act_offset + 1]
        alpha_h = spec.activation_alphas[act_offset + 2]
        beta_f = spec.activation_betas[act_offset]
        beta_g = spec.activation_betas[act_offset + 1]
        beta_h = spec.activation_betas[act_offset + 2]
        for step in range(seq_length):
            if dir_kind == "forward":
                x_t = x[step]
            else:
                t_indices = sequence_lens - 1 - step
                t_indices = np.clip(t_indices, 0, seq_length - 1)
                x_t = x[t_indices, np.arange(batch_size)]
            gates = x_t @ w_dir.T + h_prev @ r_dir.T + bias
            if spec.clip is not None and spec.clip > 0:
                gates = np.clip(gates, -spec.clip, spec.clip)
            i, o, f, c = np.split(gates, 4, axis=1)
            i = _apply_lstm_activation(act_f, i + p_i * c_prev, alpha_f, beta_f)
            if spec.input_forget:
                f = 1 - i
            else:
                f = _apply_lstm_activation(
                    act_f, f + p_f * c_prev, alpha_f, beta_f
                )
            c_tilde = _apply_lstm_activation(act_g, c, alpha_g, beta_g)
            c_new = f * c_prev + i * c_tilde
            o = _apply_lstm_activation(act_f, o + p_o * c_new, alpha_f, beta_f)
            h_new = o * _apply_lstm_activation(act_h, c_new, alpha_h, beta_h)
            active_mask = step < sequence_lens
            if not np.all(active_mask):
                h_new = np.where(active_mask[:, None], h_new, h_prev)
                c_new = np.where(active_mask[:, None], c_new, c_prev)
                if output_y is not None:
                    output_y[step, dir_index] = np.where(
                        active_mask[:, None], h_new, 0
                    )
            else:
                if output_y is not None:
                    output_y[step, dir_index] = h_new
            h_prev = h_new
            c_prev = c_new
        if output_y_h is not None:
            output_y_h[dir_index] = h_prev
        if output_y_c is not None:
            output_y_c[dir_index] = c_prev
    if spec.layout == 1:
        if output_y is not None:
            output_y = np.transpose(output_y, (2, 0, 1, 3))
        if output_y_h is not None:
            output_y_h = np.swapaxes(output_y_h, 0, 1)
        if output_y_c is not None:
            output_y_c = np.swapaxes(output_y_c, 0, 1)
    return output_y, output_y_h, output_y_c
