from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype, optional_name, value_dtype, value_shape
from .lstm import (
    ACTIVATION_KIND_BY_NAME,
    LstmSpec,
    _resolve_activations,
    _validate_direction,
    _expect_shape,
    _normalize_direction,
)
from .registry import register_lowering

if TYPE_CHECKING:
    from ..ir.ops import DynamicQuantizeLstmOp

_ACTIVATION_NAME_LOWER = {name.lower(): name for name in ACTIVATION_KIND_BY_NAME}


def _normalize_contrib_activations(attrs: dict[str, object]) -> dict[str, object]:
    """Normalize lowercase activation names from contrib ops to Title Case."""
    activations = attrs.get("activations")
    if activations is None:
        return attrs
    normalized: list[str] = []
    for name in activations:
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if isinstance(name, str):
            canonical = _ACTIVATION_NAME_LOWER.get(name.lower())
            if canonical is not None:
                name = canonical
        normalized.append(name)
    result = dict(attrs)
    result["activations"] = normalized
    return result


@dataclass(frozen=True)
class DynamicQuantizeLstmSpec:
    lstm: LstmSpec
    w_scale: str
    w_zero_point: str
    r_scale: str
    r_zero_point: str
    w_dtype: ScalarType
    r_dtype: ScalarType
    w_scale_shape: tuple[int, ...]
    w_zero_point_shape: tuple[int, ...]
    r_scale_shape: tuple[int, ...]
    r_zero_point_shape: tuple[int, ...]


def resolve_dynamic_quantize_lstm_spec(
    graph: Graph,
    node: Node,
) -> DynamicQuantizeLstmSpec:
    if len(node.inputs) != 12:
        raise UnsupportedOpError("DynamicQuantizeLSTM expects exactly 12 inputs")
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise UnsupportedOpError("DynamicQuantizeLSTM expects between 1 and 3 outputs")
    input_x = node.inputs[0]
    input_w = node.inputs[1]
    input_r = node.inputs[2]
    input_b = optional_name(node.inputs, 3)
    input_sequence_lens = optional_name(node.inputs, 4)
    input_initial_h = optional_name(node.inputs, 5)
    input_initial_c = optional_name(node.inputs, 6)
    input_p = optional_name(node.inputs, 7)
    w_scale = node.inputs[8]
    w_zero_point = node.inputs[9]
    r_scale = node.inputs[10]
    r_zero_point = node.inputs[11]

    output_y = optional_name(node.outputs, 0)
    output_y_h = optional_name(node.outputs, 1)
    output_y_c = optional_name(node.outputs, 2)
    if output_y is None and output_y_h is None and output_y_c is None:
        raise UnsupportedOpError("DynamicQuantizeLSTM expects at least one output")

    # X, B, initial_h, initial_c, P, and outputs are float
    float_names = [input_x]
    float_names.extend(
        name for name in (input_b, input_initial_h, input_initial_c, input_p) if name
    )
    float_names.extend(name for name in (output_y, output_y_h, output_y_c) if name)
    op_dtype = node_dtype(graph, node, *float_names)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "DynamicQuantizeLSTM float inputs/outputs must be float type"
        )

    # W and R are quantized (INT8 or UINT8)
    w_dtype = value_dtype(graph, input_w, node)
    if w_dtype not in {ScalarType.I8, ScalarType.U8}:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM W must be INT8 or UINT8, got {w_dtype}"
        )
    r_dtype = value_dtype(graph, input_r, node)
    if r_dtype not in {ScalarType.I8, ScalarType.U8}:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM R must be INT8 or UINT8, got {r_dtype}"
        )

    x_shape = value_shape(graph, input_x, node)
    if len(x_shape) != 3:
        raise UnsupportedOpError("DynamicQuantizeLSTM input X must be rank 3")

    # DynamicQuantizeLSTM does not have a layout attribute
    layout = 0
    seq_length, batch_size, input_size = x_shape

    w_shape = value_shape(graph, input_w, node)
    if len(w_shape) != 3:
        raise UnsupportedOpError("DynamicQuantizeLSTM input W must be rank 3")
    num_directions = w_shape[0]
    hidden_size_attr = node.attrs.get("hidden_size")
    if hidden_size_attr is None:
        if w_shape[1] % 4 != 0:
            raise UnsupportedOpError(
                "DynamicQuantizeLSTM W shape is not divisible by 4"
            )
        hidden_size = w_shape[1] // 4
    else:
        hidden_size = int(hidden_size_attr)

    direction = _normalize_direction(node.attrs.get("direction", "forward"))
    _validate_direction(direction, num_directions)

    expected_w_shape = (num_directions, input_size, 4 * hidden_size)
    _expect_shape(input_w, w_shape, expected_w_shape)

    r_shape = value_shape(graph, input_r, node)
    expected_r_shape = (num_directions, hidden_size, 4 * hidden_size)
    _expect_shape(input_r, r_shape, expected_r_shape)

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

    state_shape = (num_directions, batch_size, hidden_size)
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
            seq_length,
            num_directions,
            batch_size,
            hidden_size,
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

    activation_kinds, activation_alphas, activation_betas = _resolve_activations(
        direction, num_directions, _normalize_contrib_activations(node.attrs)
    )
    sequence_lens_dtype = (
        value_dtype(graph, input_sequence_lens, node)
        if input_sequence_lens is not None
        else None
    )

    # Validate scale/zero_point shapes
    w_scale_shape = value_shape(graph, w_scale, node)
    w_zero_point_shape = value_shape(graph, w_zero_point, node)
    r_scale_shape = value_shape(graph, r_scale, node)
    r_zero_point_shape = value_shape(graph, r_zero_point, node)

    # Scales are float
    w_scale_dtype = value_dtype(graph, w_scale, node)
    if w_scale_dtype != op_dtype:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM W_scale dtype must match float dtype, "
            f"got {w_scale_dtype}"
        )
    r_scale_dtype = value_dtype(graph, r_scale, node)
    if r_scale_dtype != op_dtype:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM R_scale dtype must match float dtype, "
            f"got {r_scale_dtype}"
        )

    # Zero points must match weight dtype
    w_zp_dtype = value_dtype(graph, w_zero_point, node)
    if w_zp_dtype != w_dtype:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM W_zero_point dtype must match W dtype, "
            f"got {w_zp_dtype}"
        )
    r_zp_dtype = value_dtype(graph, r_zero_point, node)
    if r_zp_dtype != r_dtype:
        raise UnsupportedOpError(
            f"DynamicQuantizeLSTM R_zero_point dtype must match R dtype, "
            f"got {r_zp_dtype}"
        )

    lstm = LstmSpec(
        input_x=input_x,
        input_w=input_w,
        input_r=input_r,
        input_b=input_b,
        input_sequence_lens=input_sequence_lens,
        input_initial_h=input_initial_h,
        input_initial_c=input_initial_c,
        input_p=input_p,
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
        sequence_lens_dtype=sequence_lens_dtype,
    )

    return DynamicQuantizeLstmSpec(
        lstm=lstm,
        w_scale=w_scale,
        w_zero_point=w_zero_point,
        r_scale=r_scale,
        r_zero_point=r_zero_point,
        w_dtype=w_dtype,
        r_dtype=r_dtype,
        w_scale_shape=w_scale_shape,
        w_zero_point_shape=w_zero_point_shape,
        r_scale_shape=r_scale_shape,
        r_zero_point_shape=r_zero_point_shape,
    )


@register_lowering("DynamicQuantizeLSTM")
def lower_dynamic_quantize_lstm(
    graph: Graph,
    node: Node,
) -> DynamicQuantizeLstmOp:
    from ..ir.ops import DynamicQuantizeLstmOp

    spec = resolve_dynamic_quantize_lstm_spec(graph, node)
    lstm = spec.lstm
    return DynamicQuantizeLstmOp(
        input_x=lstm.input_x,
        input_w=lstm.input_w,
        input_r=lstm.input_r,
        input_b=lstm.input_b,
        input_sequence_lens=lstm.input_sequence_lens,
        input_initial_h=lstm.input_initial_h,
        input_initial_c=lstm.input_initial_c,
        input_p=lstm.input_p,
        w_scale=spec.w_scale,
        w_zero_point=spec.w_zero_point,
        r_scale=spec.r_scale,
        r_zero_point=spec.r_zero_point,
        output_y=lstm.output_y,
        output_y_h=lstm.output_y_h,
        output_y_c=lstm.output_y_c,
        seq_length=lstm.seq_length,
        batch_size=lstm.batch_size,
        input_size=lstm.input_size,
        hidden_size=lstm.hidden_size,
        num_directions=lstm.num_directions,
        direction=lstm.direction,
        input_forget=lstm.input_forget,
        clip=lstm.clip,
        activation_kinds=lstm.activation_kinds,
        activation_alphas=lstm.activation_alphas,
        activation_betas=lstm.activation_betas,
        dtype=lstm.dtype,
        w_dtype=spec.w_dtype,
        r_dtype=spec.r_dtype,
        sequence_lens_dtype=lstm.sequence_lens_dtype,
        w_scale_shape=spec.w_scale_shape,
        w_zero_point_shape=spec.w_zero_point_shape,
        r_scale_shape=spec.r_scale_shape,
        r_zero_point_shape=spec.r_zero_point_shape,
    )
