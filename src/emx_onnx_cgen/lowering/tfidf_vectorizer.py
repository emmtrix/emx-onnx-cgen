from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import TfIdfVectorizerOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_INPUT_DTYPES = {ScalarType.I32, ScalarType.I64}
_SUPPORTED_OUTPUT_DTYPES = {ScalarType.F32}
_SUPPORTED_MODES = {"TF", "IDF", "TFIDF"}


def _decode_mode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _ensure_int_list(
    values: object | None, *, name: str, node: Node
) -> tuple[int, ...]:
    if values is None:
        raise UnsupportedOpError(f"{node.op_type} requires {name} attribute")
    try:
        return tuple(int(value) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise UnsupportedOpError(
            f"{node.op_type} {name} attribute must be a list of integers"
        ) from exc


def _ensure_float_list(
    values: object | None, *, name: str, node: Node
) -> tuple[float, ...] | None:
    if values is None:
        return None
    try:
        return tuple(float(value) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise UnsupportedOpError(
            f"{node.op_type} {name} attribute must be a list of floats"
        ) from exc


def _validate_output_shape(
    node: Node,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    output_dim: int,
) -> None:
    if len(input_shape) == 1:
        expected = (output_dim,)
    else:
        expected = (input_shape[0], output_dim)
    if output_shape != expected:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {expected}, got {output_shape}"
        )


@register_lowering("TfIdfVectorizer")
def lower_tfidf_vectorizer(graph: Graph, node: Node) -> TfIdfVectorizerOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} expects 1 input and 1 output"
        )
    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if input_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise UnsupportedOpError(
            f"{node.op_type} input dtype must be int32 or int64, "
            f"got {input_dtype.onnx_name}"
        )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            f"{node.op_type} output dtype must be float, "
            f"got {output_dtype.onnx_name}"
        )
    if len(input_shape) not in {1, 2}:
        raise UnsupportedOpError(
            f"{node.op_type} input rank must be 1 or 2, got {len(input_shape)}"
        )
    mode_value = node.attrs.get("mode")
    if mode_value is None:
        raise UnsupportedOpError(
            f"{node.op_type} requires mode attribute"
        )
    mode = _decode_mode(mode_value)
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(
            f"{node.op_type} mode must be one of {sorted(_SUPPORTED_MODES)}, "
            f"got {mode}"
        )
    min_gram_length = int(node.attrs.get("min_gram_length", 0))
    max_gram_length = int(node.attrs.get("max_gram_length", 0))
    max_skip_count = int(node.attrs.get("max_skip_count", 0))
    if min_gram_length <= 0 or max_gram_length <= 0:
        raise UnsupportedOpError(
            f"{node.op_type} requires positive min/max gram lengths"
        )
    if min_gram_length > max_gram_length:
        raise UnsupportedOpError(
            f"{node.op_type} min_gram_length {min_gram_length} exceeds "
            f"max_gram_length {max_gram_length}"
        )
    if max_skip_count < 0:
        raise UnsupportedOpError(
            f"{node.op_type} max_skip_count must be non-negative"
        )
    ngram_counts = _ensure_int_list(
        node.attrs.get("ngram_counts"), name="ngram_counts", node=node
    )
    ngram_indexes = _ensure_int_list(
        node.attrs.get("ngram_indexes"), name="ngram_indexes", node=node
    )
    if "pool_strings" in node.attrs:
        raise UnsupportedOpError(
            f"{node.op_type} string pools are not supported"
        )
    pool_int64s = _ensure_int_list(
        node.attrs.get("pool_int64s"), name="pool_int64s", node=node
    )
    weights = _ensure_float_list(
        node.attrs.get("weights"), name="weights", node=node
    )
    if len(ngram_counts) < max_gram_length:
        raise UnsupportedOpError(
            f"{node.op_type} ngram_counts length must be >= max_gram_length"
        )
    if ngram_counts and ngram_counts[0] != 0:
        raise UnsupportedOpError(
            f"{node.op_type} ngram_counts must start with 0"
        )
    if any(value < 0 for value in ngram_counts):
        raise UnsupportedOpError(
            f"{node.op_type} ngram_counts must be non-negative"
        )
    if any(
        later < earlier
        for earlier, later in zip(ngram_counts, ngram_counts[1:])
    ):
        raise UnsupportedOpError(
            f"{node.op_type} ngram_counts must be non-decreasing"
        )
    pool_size = len(pool_int64s)
    if ngram_counts and ngram_counts[-1] > pool_size:
        raise UnsupportedOpError(
            f"{node.op_type} ngram_counts exceeds pool_int64s length"
        )
    total_ngrams = 0
    for gram_length in range(1, max_gram_length + 1):
        start = ngram_counts[gram_length - 1]
        end = (
            ngram_counts[gram_length]
            if gram_length < len(ngram_counts)
            else pool_size
        )
        count = end - start
        if count < 0 or count % gram_length != 0:
            raise UnsupportedOpError(
                f"{node.op_type} pool size for {gram_length}-grams "
                "must be divisible by gram length"
            )
        total_ngrams += count // gram_length
    if total_ngrams != len(ngram_indexes):
        raise UnsupportedOpError(
            f"{node.op_type} ngram_indexes length {len(ngram_indexes)} "
            f"does not match pool ngram count {total_ngrams}"
        )
    if weights is not None and len(weights) != len(ngram_indexes):
        raise UnsupportedOpError(
            f"{node.op_type} weights length {len(weights)} does not match "
            f"ngram_indexes length {len(ngram_indexes)}"
        )
    output_dim = max(ngram_indexes, default=-1) + 1
    _validate_output_shape(node, input_shape, output_shape, output_dim)
    return TfIdfVectorizerOp(
        input0=input_name,
        output=output_name,
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        min_gram_length=min_gram_length,
        max_gram_length=max_gram_length,
        max_skip_count=max_skip_count,
        mode=mode,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_int64s=pool_int64s,
        weights=weights,
    )
