from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype, optional_name, value_dtype, value_shape
from .registry import register_lowering

if TYPE_CHECKING:
    from ..ir.ops import RnnOp


ACTIVATION_KIND_BY_NAME = {
    "Relu": 0,
    "Tanh": 1,
    "Sigmoid": 2,
    "Affine": 3,
    "LeakyRelu": 4,
    "ThresholdedRelu": 5,
    "ScaledTanh": 6,
    "HardSigmoid": 7,
    "Elu": 8,
    "Softsign": 9,
    "Softplus": 10,
}

DEFAULT_ACTIVATIONS = ("Tanh",)

DEFAULT_ALPHA_BY_NAME = {
    "Affine": 1.0,
    "LeakyRelu": 0.01,
    "ThresholdedRelu": 1.0,
    "ScaledTanh": 1.0,
    "HardSigmoid": 0.2,
    "Elu": 1.0,
}

DEFAULT_BETA_BY_NAME = {
    "Affine": 0.0,
    "ScaledTanh": 1.0,
    "HardSigmoid": 0.5,
}


@dataclass(frozen=True)
class RnnSpec:
    input_x: str
    input_w: str
    input_r: str
    input_b: str | None
    input_sequence_lens: str | None
    input_initial_h: str | None
    output_y: str | None
    output_y_h: str | None
    seq_length: int
    batch_size: int
    input_size: int
    hidden_size: int
    num_directions: int
    direction: str
    layout: int
    clip: float | None
    activation_kinds: tuple[int, ...]
    activation_alphas: tuple[float, ...]
    activation_betas: tuple[float, ...]
    dtype: ScalarType
    sequence_lens_dtype: ScalarType | None


def _normalize_activation_names(values: Iterable[object]) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise UnsupportedOpError("RNN activations must be strings")
        names.append(value)
    return names


def _resolve_activation_params(
    activations: Sequence[str],
    activation_alpha: Sequence[float] | None,
    activation_beta: Sequence[float] | None,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    if activation_alpha is None:
        activation_alpha = []
    if activation_beta is None:
        activation_beta = []
    if activation_alpha and len(activation_alpha) != len(activations):
        raise UnsupportedOpError("RNN activation_alpha must match activations")
    if activation_beta and len(activation_beta) != len(activations):
        raise UnsupportedOpError("RNN activation_beta must match activations")
    activation_kinds: list[int] = []
    alphas: list[float] = []
    betas: list[float] = []
    for idx, name in enumerate(activations):
        kind = ACTIVATION_KIND_BY_NAME.get(name)
        if kind is None:
            raise UnsupportedOpError(f"Unsupported RNN activation {name}")
        activation_kinds.append(kind)
        if activation_alpha:
            alpha = float(activation_alpha[idx])
        else:
            alpha = DEFAULT_ALPHA_BY_NAME.get(name, 1.0)
        if activation_beta:
            beta = float(activation_beta[idx])
        else:
            beta = DEFAULT_BETA_BY_NAME.get(name, 0.0)
        alphas.append(alpha)
        betas.append(beta)
    return tuple(activation_kinds), tuple(alphas), tuple(betas)


def _resolve_activations(
    num_directions: int, attrs: dict[str, object]
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    activations_attr = attrs.get("activations")
    if activations_attr is None:
        activations = list(DEFAULT_ACTIVATIONS)
    else:
        activations = _normalize_activation_names(activations_attr)
    if num_directions == 1:
        if len(activations) != 1:
            raise UnsupportedOpError("RNN activations must have length 1")
    else:
        if len(activations) == 1:
            activations = activations * 2
        elif len(activations) != 2:
            raise UnsupportedOpError("Bidirectional RNN activations must be length 2")
    return _resolve_activation_params(
        activations,
        attrs.get("activation_alpha"),
        attrs.get("activation_beta"),
    )


def _expect_shape(name: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if shape != expected:
        raise UnsupportedOpError(
            f"RNN input {name} must have shape {expected}, got {shape}"
        )


def _validate_direction(direction: str, num_directions: int) -> None:
    if direction == "bidirectional" and num_directions != 2:
        raise UnsupportedOpError(
            "RNN expects num_directions=2 for bidirectional models"
        )
    if direction in {"forward", "reverse"} and num_directions != 1:
        raise UnsupportedOpError(
            "RNN expects num_directions=1 for forward/reverse models"
        )
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise UnsupportedOpError(f"Unsupported RNN direction {direction}")


def resolve_rnn_spec(graph: Graph, node: Node) -> RnnSpec:
    if len(node.inputs) < 3 or len(node.inputs) > 6:
        raise UnsupportedOpError("RNN expects between 3 and 6 inputs")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise UnsupportedOpError("RNN expects between 1 and 2 outputs")
    input_x = node.inputs[0]
    input_w = node.inputs[1]
    input_r = node.inputs[2]
    input_b = optional_name(node.inputs, 3)
    input_sequence_lens = optional_name(node.inputs, 4)
    input_initial_h = optional_name(node.inputs, 5)
    output_y = optional_name(node.outputs, 0)
    output_y_h = optional_name(node.outputs, 1)
    if output_y is None and output_y_h is None:
        raise UnsupportedOpError("RNN expects at least one output")
    op_dtype = node_dtype(
        graph,
        node,
        input_x,
        input_w,
        input_r,
        *(name for name in (input_b, input_initial_h) if name),
        *(name for name in (output_y, output_y_h) if name),
    )
    if not op_dtype.is_float:
        raise UnsupportedOpError("RNN supports float16, float, and double inputs only")
    x_shape = value_shape(graph, input_x, node)
    if len(x_shape) != 3:
        raise UnsupportedOpError("RNN input X must be rank 3")
    layout = int(node.attrs.get("layout", 0))
    if layout not in {0, 1}:
        raise UnsupportedOpError("RNN layout must be 0 or 1")
    if layout == 0:
        seq_length, batch_size, input_size = x_shape
    else:
        batch_size, seq_length, input_size = x_shape
    w_shape = value_shape(graph, input_w, node)
    if len(w_shape) != 3:
        raise UnsupportedOpError("RNN input W must be rank 3")
    num_directions = w_shape[0]
    hidden_size_attr = node.attrs.get("hidden_size")
    if hidden_size_attr is None:
        hidden_size = w_shape[1]
    else:
        hidden_size = int(hidden_size_attr)
    direction = str(node.attrs.get("direction", "forward"))
    _validate_direction(direction, num_directions)
    _expect_shape(input_w, w_shape, (num_directions, hidden_size, input_size))
    _expect_shape(
        input_r,
        value_shape(graph, input_r, node),
        (num_directions, hidden_size, hidden_size),
    )
    if input_b is not None:
        _expect_shape(
            input_b,
            value_shape(graph, input_b, node),
            (num_directions, 2 * hidden_size),
        )
    if input_sequence_lens is not None:
        seq_dtype = value_dtype(graph, input_sequence_lens, node)
        if seq_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise UnsupportedOpError("RNN sequence_lens must be int32 or int64")
        _expect_shape(
            input_sequence_lens,
            value_shape(graph, input_sequence_lens, node),
            (batch_size,),
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
    if output_y is not None:
        expected_y_shape = (
            (seq_length, num_directions, batch_size, hidden_size)
            if layout == 0
            else (batch_size, seq_length, num_directions, hidden_size)
        )
        _expect_shape(output_y, value_shape(graph, output_y, node), expected_y_shape)
    if output_y_h is not None:
        _expect_shape(output_y_h, value_shape(graph, output_y_h, node), state_shape)
    clip = node.attrs.get("clip")
    if clip is not None:
        clip = float(clip)
        if clip < 0:
            raise UnsupportedOpError("RNN clip must be non-negative")
    activation_kinds, activation_alphas, activation_betas = _resolve_activations(
        num_directions, node.attrs
    )
    sequence_lens_dtype = (
        value_dtype(graph, input_sequence_lens, node)
        if input_sequence_lens is not None
        else None
    )
    return RnnSpec(
        input_x=input_x,
        input_w=input_w,
        input_r=input_r,
        input_b=input_b,
        input_sequence_lens=input_sequence_lens,
        input_initial_h=input_initial_h,
        output_y=output_y,
        output_y_h=output_y_h,
        seq_length=seq_length,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        num_directions=num_directions,
        direction=direction,
        layout=layout,
        clip=clip,
        activation_kinds=activation_kinds,
        activation_alphas=activation_alphas,
        activation_betas=activation_betas,
        dtype=op_dtype,
        sequence_lens_dtype=sequence_lens_dtype,
    )


@register_lowering("RNN")
def lower_rnn(graph: Graph, node: Node) -> RnnOp:
    from ..ir.ops import RnnOp

    spec = resolve_rnn_spec(graph, node)
    return RnnOp(
        input_x=spec.input_x,
        input_w=spec.input_w,
        input_r=spec.input_r,
        input_b=spec.input_b,
        input_sequence_lens=spec.input_sequence_lens,
        input_initial_h=spec.input_initial_h,
        output_y=spec.output_y,
        output_y_h=spec.output_y_h,
        seq_length=spec.seq_length,
        batch_size=spec.batch_size,
        input_size=spec.input_size,
        hidden_size=spec.hidden_size,
        num_directions=spec.num_directions,
        direction=spec.direction,
        layout=spec.layout,
        clip=spec.clip,
        activation_kinds=spec.activation_kinds,
        activation_alphas=spec.activation_alphas,
        activation_betas=spec.activation_betas,
        dtype=spec.dtype,
        sequence_lens_dtype=spec.sequence_lens_dtype,
    )
