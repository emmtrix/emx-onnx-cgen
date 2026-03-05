from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import STFTOp
from ..lowering.common import optional_name, value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_STFT_DTYPES = {
    ScalarType.F32,
    ScalarType.F64,
}


def _find_initializer(graph: Graph | GraphContext, name: str) -> Initializer | None:
    if isinstance(graph, GraphContext):
        return graph.initializer(name)
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


def _read_scalar_int_initializer(
    graph: Graph | GraphContext,
    name: str,
    node: Node,
    label: str,
) -> int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be int32 or int64, "
            f"got {initializer.type.dtype.onnx_name}"
        )
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if values.size != 1:
        raise UnsupportedOpError(f"{node.op_type} {label} input must be a scalar")
    return int(values[0])


def _require_scalar_int_tensor(
    graph: Graph | GraphContext,
    name: str,
    node: Node,
    label: str,
) -> None:
    shape = value_shape(graph, name, node)
    if not _is_scalar_shape(shape):
        raise UnsupportedOpError(f"{node.op_type} {label} input must be a scalar")
    dtype = value_dtype(graph, name, node)
    if dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be int32 or int64, got {dtype.onnx_name}"
        )


@register_lowering("STFT")
def lower_stft(graph: Graph | GraphContext, node: Node) -> STFTOp:
    if len(node.inputs) < 2 or len(node.inputs) > 4 or len(node.outputs) != 1:
        raise UnsupportedOpError("STFT must have 2 to 4 inputs and 1 output")

    signal_name = node.inputs[0]
    frame_step_name = node.inputs[1]
    if not signal_name or not frame_step_name:
        raise UnsupportedOpError("STFT requires signal and frame_step inputs")
    window_name = optional_name(node.inputs, 2)
    frame_length_name = optional_name(node.inputs, 3)
    output_name = node.outputs[0]

    signal_shape = value_shape(graph, signal_name, node)
    output_shape = value_shape(graph, output_name, node)
    if any(dim < 0 for dim in signal_shape + output_shape):
        raise ShapeInferenceError("STFT does not support dynamic dimensions")
    if len(signal_shape) != 3:
        raise ShapeInferenceError(
            f"STFT signal rank must be 3, got {len(signal_shape)}"
        )
    if len(output_shape) != 4:
        raise ShapeInferenceError(
            f"STFT output rank must be 4, got {len(output_shape)}"
        )
    if signal_shape[2] not in {1, 2}:
        raise ShapeInferenceError(
            f"STFT signal last dimension must be 1 or 2, got {signal_shape[2]}"
        )
    if output_shape[3] != 2:
        raise ShapeInferenceError(
            f"STFT output last dimension must be 2, got {output_shape[3]}"
        )
    if output_shape[0] != signal_shape[0]:
        raise ShapeInferenceError(
            "STFT output batch dimension must match signal batch dimension"
        )

    signal_dtype = value_dtype(graph, signal_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if signal_dtype != output_dtype:
        raise UnsupportedOpError(
            "STFT expects matching signal/output dtypes, "
            f"got {signal_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if signal_dtype not in _SUPPORTED_STFT_DTYPES:
        raise UnsupportedOpError(
            f"STFT supports only float and double, got {signal_dtype.onnx_name}"
        )

    onesided = bool(int(node.attrs.get("onesided", 1)))
    if signal_shape[2] == 2 and onesided:
        raise UnsupportedOpError(
            "STFT onesided output is not supported for complex input"
        )

    frame_step_const = _read_scalar_int_initializer(
        graph, frame_step_name, node, "frame_step"
    )
    if frame_step_const is None:
        _require_scalar_int_tensor(graph, frame_step_name, node, "frame_step")
    elif frame_step_const <= 0:
        raise ShapeInferenceError(
            f"STFT frame_step must be > 0, got {frame_step_const}"
        )

    window_length: int | None = None
    if window_name is not None:
        window_shape = value_shape(graph, window_name, node)
        if len(window_shape) != 1 or window_shape[0] <= 0:
            raise ShapeInferenceError(
                f"STFT window must have shape [N] with N > 0, got {window_shape}"
            )
        window_dtype = value_dtype(graph, window_name, node)
        if window_dtype != signal_dtype:
            raise UnsupportedOpError(
                "STFT window dtype must match signal dtype, "
                f"got {window_dtype.onnx_name} and {signal_dtype.onnx_name}"
            )
        window_length = window_shape[0]

    frame_length_const: int | None = None
    if frame_length_name is not None:
        frame_length_const = _read_scalar_int_initializer(
            graph, frame_length_name, node, "frame_length"
        )
        if frame_length_const is None:
            _require_scalar_int_tensor(graph, frame_length_name, node, "frame_length")
        elif frame_length_const <= 0:
            raise ShapeInferenceError(
                f"STFT frame_length must be > 0, got {frame_length_const}"
            )

    output_bins = output_shape[2]
    signal_length = signal_shape[1]
    if frame_length_const is not None:
        fft_length = frame_length_const
    elif frame_length_name is None:
        fft_length = window_length if window_length is not None else signal_length
    elif window_length is not None:
        fft_length = window_length
    elif onesided:
        fft_length = (output_bins - 1) * 2
    else:
        fft_length = output_bins
    if fft_length <= 0:
        raise ShapeInferenceError(f"STFT inferred invalid frame_length {fft_length}")

    expected_bins = fft_length // 2 + 1 if onesided else fft_length
    if output_bins != expected_bins:
        raise ShapeInferenceError(
            f"STFT output frequency bins must be {expected_bins}, got {output_bins}"
        )

    frame_length_for_count: int | None
    if frame_length_const is not None:
        frame_length_for_count = frame_length_const
    elif frame_length_name is None:
        frame_length_for_count = (
            window_length if window_length is not None else signal_length
        )
    else:
        frame_length_for_count = None
    if frame_step_const is not None and frame_length_for_count is not None:
        expected_frames = (
            1 + (signal_length - frame_length_for_count) // frame_step_const
        )
        if expected_frames <= 0:
            raise ShapeInferenceError(
                "STFT inferred a non-positive frame count; check frame_step and frame_length"
            )
        if output_shape[1] != expected_frames:
            raise ShapeInferenceError(
                f"STFT output frame dimension must be {expected_frames}, got {output_shape[1]}"
            )

    frame_length_literal: int
    use_runtime_frame_length = (
        frame_length_name is not None and frame_length_const is None
    )
    if frame_length_const is not None:
        frame_length_literal = frame_length_const
    elif frame_length_name is None:
        frame_length_literal = (
            window_length if window_length is not None else signal_length
        )
    else:
        frame_length_literal = fft_length

    return STFTOp(
        signal=signal_name,
        frame_step=frame_step_name,
        window=window_name,
        frame_length_input=frame_length_name,
        output=output_name,
        onesided=onesided,
        input_is_complex=signal_shape[2] == 2,
        fft_length=fft_length,
        window_length=window_length if window_length is not None else fft_length,
        frame_length_literal=frame_length_literal,
        use_runtime_frame_length=use_runtime_frame_length,
    )
