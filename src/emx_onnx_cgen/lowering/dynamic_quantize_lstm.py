from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype, optional_name, value_dtype, value_shape
from .lstm import (
    ACTIVATION_KIND_BY_NAME,
    _normalize_direction,
    _resolve_activation_params,
    _validate_direction,
    _expect_shape,
)
from .registry import register_lowering

if TYPE_CHECKING:
    from ..ir.ops import DynamicQuantizeLstmOp

_ACTIVATION_NAME_ALIASES: dict[str, str] = {
    name.lower(): name for name in ACTIVATION_KIND_BY_NAME
}

DEFAULT_ACTIVATIONS = ("Sigmoid", "Tanh", "Tanh")

_QUANTIZED_DTYPES = {ScalarType.I8, ScalarType.U8}


def _normalize_activation_name(name: str) -> str:
    return _ACTIVATION_NAME_ALIASES.get(name.lower(), name)


def _normalize_activation_names_dqlstm(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise UnsupportedOpError("DynamicQuantizeLSTM activations must be a sequence")
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise UnsupportedOpError("DynamicQuantizeLSTM activations must be strings")
        names.append(_normalize_activation_name(value))
    return names


def _resolve_activations_dqlstm(
    direction: str, num_directions: int, attrs: dict[str, object]
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    activations_attr = attrs.get("activations")
    if activations_attr is None:
        activations = list(DEFAULT_ACTIVATIONS)
    else:
        activations = _normalize_activation_names_dqlstm(activations_attr)
    if num_directions == 1:
        if len(activations) != 3:
            raise UnsupportedOpError("DynamicQuantizeLSTM activations must have length 3")
    else:
        if len(activations) == 3:
            activations = activations * 2
        elif len(activations) != 6:
            raise UnsupportedOpError(
                "Bidirectional DynamicQuantizeLSTM activations must be length 6"
            )
    activation_alpha = attrs.get("activation_alpha")
    activation_beta = attrs.get("activation_beta")
    return _resolve_activation_params(activations, activation_alpha, activation_beta)


@dataclass(frozen=True)
class DynamicQuantizeLstmSpec:
    input_x: str
    input_w: str
    input_r: str
    input_b: str | None
    input_sequence_lens: str | None
    input_initial_h: str | None
    input_initial_c: str | None
    input_p: str | None
    input_w_scale: str
    input_w_zero_point: str
    input_r_scale: str
    input_r_zero_point: str
    output_y: str | None
    output_y_h: str | None
    output_y_c: str | None
    seq_length: int
    batch_size: int
    input_size: int
    hidden_size: int
    num_directions: int
    direction: str
    layout: int
    input_forget: int
    clip: float | None
    activation_kinds: tuple[int, ...]
    activation_alphas: tuple[float, ...]
    activation_betas: tuple[float, ...]
    dtype: ScalarType
    weight_dtype: ScalarType
    sequence_lens_dtype: ScalarType | None
    per_channel_weights: bool


def resolve_dynamic_quantize_lstm_spec(
    graph: Graph, node: Node
) -> DynamicQuantizeLstmSpec:
    if len(node.inputs) != 12:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM expects exactly 12 inputs, got {len(node.inputs)}"
        )
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM expects between 1 and 3 outputs"
        )
    input_x = node.inputs[0]
    input_w = node.inputs[1]
    input_r = node.inputs[2]
    input_b = optional_name(node.inputs, 3)
    input_sequence_lens = optional_name(node.inputs, 4)
    input_initial_h = optional_name(node.inputs, 5)
    input_initial_c = optional_name(node.inputs, 6)
    input_p = optional_name(node.inputs, 7)
    input_w_scale = node.inputs[8]
    input_w_zero_point = node.inputs[9]
    input_r_scale = node.inputs[10]
    input_r_zero_point = node.inputs[11]
    output_y = optional_name(node.outputs, 0)
    output_y_h = optional_name(node.outputs, 1)
    output_y_c = optional_name(node.outputs, 2)
    if output_y is None and output_y_h is None and output_y_c is None:
        raise UnsupportedOpError("DynamicQuantizeLSTM expects at least one output")

    float_inputs = [input_x]
    for name in (input_b, input_initial_h, input_initial_c, input_p):
        if name:
            float_inputs.append(name)
    for name in (input_w_scale, input_r_scale):
        float_inputs.append(name)
    for name in (output_y, output_y_h, output_y_c):
        if name:
            float_inputs.append(name)
    op_dtype = node_dtype(graph, node, *float_inputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM supports float16, float, and double inputs only"
        )

    weight_dtype = value_dtype(graph, input_w, node)
    if weight_dtype not in _QUANTIZED_DTYPES:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM W must be int8 or uint8, got {weight_dtype}"
        )
    r_dtype = value_dtype(graph, input_r, node)
    if r_dtype != weight_dtype:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM W and R must have the same quantized dtype"
        )
    zp_w_dtype = value_dtype(graph, input_w_zero_point, node)
    if zp_w_dtype != weight_dtype:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM W_zero_point dtype must match W dtype"
        )
    zp_r_dtype = value_dtype(graph, input_r_zero_point, node)
    if zp_r_dtype != weight_dtype:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM R_zero_point dtype must match R dtype"
        )

    x_shape = value_shape(graph, input_x, node)
    if len(x_shape) != 3:
        raise UnsupportedOpError("DynamicQuantizeLSTM input X must be rank 3")
    layout = int(node.attrs.get("layout", 0))
    if layout not in {0, 1}:
        raise UnsupportedOpError("DynamicQuantizeLSTM layout must be 0 or 1")
    if layout == 0:
        seq_length, batch_size, input_size = x_shape
    else:
        batch_size, seq_length, input_size = x_shape

    w_shape = value_shape(graph, input_w, node)
    if len(w_shape) != 3:
        raise UnsupportedOpError("DynamicQuantizeLSTM input W must be rank 3")
    num_directions = w_shape[0]
    hidden_size_attr = node.attrs.get("hidden_size")
    if hidden_size_attr is None:
        if w_shape[2] % 4 != 0:
            raise UnsupportedOpError(
                "DynamicQuantizeLSTM W last dim is not divisible by 4"
            )
        hidden_size = w_shape[2] // 4
    else:
        hidden_size = int(hidden_size_attr)
    direction = _normalize_direction(node.attrs.get("direction", "forward"))
    _validate_direction(direction, num_directions)
    # W shape: [num_directions, input_size, 4*hidden_size] (transposed vs standard LSTM)
    expected_w_shape = (num_directions, input_size, 4 * hidden_size)
    _expect_shape(input_w, w_shape, expected_w_shape)
    # R shape: [num_directions, hidden_size, 4*hidden_size] (transposed vs standard LSTM)
    r_shape = value_shape(graph, input_r, node)
    expected_r_shape = (num_directions, hidden_size, 4 * hidden_size)
    _expect_shape(input_r, r_shape, expected_r_shape)

    # Scale and zero_point shapes: either (num_directions,) for per-tensor
    # or (num_directions, 4*hidden_size) for per-channel
    w_scale_shape = value_shape(graph, input_w_scale, node)
    if w_scale_shape == (num_directions,):
        per_channel_weights = False
    elif w_scale_shape == (num_directions, 4 * hidden_size):
        per_channel_weights = True
    else:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM W_scale must have shape ({num_directions},) or "
            f"({num_directions}, {4 * hidden_size}), got {w_scale_shape}"
        )
    for scale_name in (input_r_scale,):
        scale_shape = value_shape(graph, scale_name, node)
        expected = (num_directions, 4 * hidden_size) if per_channel_weights else (num_directions,)
        if scale_shape != expected:
            raise UnsupportedOpError(
                f"DynamicQuantizeLSTM scale {scale_name} must have shape "
                f"{expected}, got {scale_shape}"
            )
    for zp_name in (input_w_zero_point, input_r_zero_point):
        zp_shape = value_shape(graph, zp_name, node)
        expected = (num_directions, 4 * hidden_size) if per_channel_weights else (num_directions,)
        if zp_shape != expected:
            raise UnsupportedOpError(
                f"DynamicQuantizeLSTM zero_point {zp_name} must have shape "
                f"{expected}, got {zp_shape}"
            )

    if input_b is not None:
        b_shape = value_shape(graph, input_b, node)
        _expect_shape(input_b, b_shape, (num_directions, 8 * hidden_size))
    if input_sequence_lens is not None:
        seq_dtype = value_dtype(graph, input_sequence_lens, node)
        if seq_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise UnsupportedOpError(
                "DynamicQuantizeLSTM sequence_lens must be int32 or int64"
            )
        seq_shape = value_shape(graph, input_sequence_lens, node)
        if seq_shape != (batch_size,):
            raise UnsupportedOpError(
                "DynamicQuantizeLSTM sequence_lens must match batch size"
            )
    state_shape = (
        (num_directions, batch_size, hidden_size)
        if layout == 0
        else (batch_size, num_directions, hidden_size)
    )
    if input_initial_h is not None:
        _expect_shape(
            input_initial_h,
            value_shape(graph, input_initial_h, node),
            state_shape,
        )
    if input_initial_c is not None:
        _expect_shape(
            input_initial_c,
            value_shape(graph, input_initial_c, node),
            state_shape,
        )
    if input_p is not None:
        _expect_shape(
            input_p,
            value_shape(graph, input_p, node),
            (num_directions, 3 * hidden_size),
        )
    if output_y is not None:
        expected_y_shape = (
            (seq_length, num_directions, batch_size, hidden_size)
            if layout == 0
            else (batch_size, seq_length, num_directions, hidden_size)
        )
        _expect_shape(output_y, value_shape(graph, output_y, node), expected_y_shape)
    if output_y_h is not None:
        _expect_shape(
            output_y_h,
            value_shape(graph, output_y_h, node),
            state_shape,
        )
    if output_y_c is not None:
        _expect_shape(
            output_y_c,
            value_shape(graph, output_y_c, node),
            state_shape,
        )
    input_forget = int(node.attrs.get("input_forget", 0))
    if input_forget not in {0, 1}:
        raise UnsupportedOpError("DynamicQuantizeLSTM input_forget must be 0 or 1")
    clip = node.attrs.get("clip")
    if clip is not None:
        clip = float(clip)
        if clip < 0:
            raise UnsupportedOpError("DynamicQuantizeLSTM clip must be non-negative")
    activation_kinds, activation_alphas, activation_betas = _resolve_activations_dqlstm(
        direction, num_directions, node.attrs
    )
    sequence_lens_dtype = (
        value_dtype(graph, input_sequence_lens, node)
        if input_sequence_lens is not None
        else None
    )
    return DynamicQuantizeLstmSpec(
        input_x=input_x,
        input_w=input_w,
        input_r=input_r,
        input_b=input_b,
        input_sequence_lens=input_sequence_lens,
        input_initial_h=input_initial_h,
        input_initial_c=input_initial_c,
        input_p=input_p,
        input_w_scale=input_w_scale,
        input_w_zero_point=input_w_zero_point,
        input_r_scale=input_r_scale,
        input_r_zero_point=input_r_zero_point,
        output_y=output_y,
        output_y_h=output_y_h,
        output_y_c=output_y_c,
        seq_length=seq_length,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        num_directions=num_directions,
        direction=direction,
        layout=layout,
        input_forget=input_forget,
        clip=clip,
        activation_kinds=activation_kinds,
        activation_alphas=activation_alphas,
        activation_betas=activation_betas,
        dtype=op_dtype,
        weight_dtype=weight_dtype,
        sequence_lens_dtype=sequence_lens_dtype,
        per_channel_weights=per_channel_weights,
    )


@register_lowering("DynamicQuantizeLSTM")
def lower_dynamic_quantize_lstm(graph: Graph, node: Node) -> DynamicQuantizeLstmOp:
    from ..ir.ops import DynamicQuantizeLstmOp

    spec = resolve_dynamic_quantize_lstm_spec(graph, node)
    return DynamicQuantizeLstmOp(
        input_x=spec.input_x,
        input_w=spec.input_w,
        input_r=spec.input_r,
        input_b=spec.input_b,
        input_sequence_lens=spec.input_sequence_lens,
        input_initial_h=spec.input_initial_h,
        input_initial_c=spec.input_initial_c,
        input_p=spec.input_p,
        input_w_scale=spec.input_w_scale,
        input_w_zero_point=spec.input_w_zero_point,
        input_r_scale=spec.input_r_scale,
        input_r_zero_point=spec.input_r_zero_point,
        output_y=spec.output_y,
        output_y_h=spec.output_y_h,
        output_y_c=spec.output_y_c,
        seq_length=spec.seq_length,
        batch_size=spec.batch_size,
        input_size=spec.input_size,
        hidden_size=spec.hidden_size,
        num_directions=spec.num_directions,
        direction=spec.direction,
        layout=spec.layout,
        input_forget=spec.input_forget,
        clip=spec.clip,
        activation_kinds=spec.activation_kinds,
        activation_alphas=spec.activation_alphas,
        activation_betas=spec.activation_betas,
        dtype=spec.dtype,
        weight_dtype=spec.weight_dtype,
        sequence_lens_dtype=spec.sequence_lens_dtype,
        per_channel_weights=spec.per_channel_weights,
    )
