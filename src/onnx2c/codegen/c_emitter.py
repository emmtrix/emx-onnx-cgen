from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ..errors import CodegenError
from ..ops import (
    COMPARE_FUNCTIONS,
    OperatorKind,
    binary_op_symbol,
    unary_op_symbol,
)
from shared.scalar_functions import (
    ScalarFunction,
    ScalarFunctionKey,
    ScalarFunctionRegistry,
)
from shared.scalar_types import ScalarFunctionError, ScalarType


def _format_c_indentation(source: str, *, indent: str = "    ") -> str:
    formatted_lines: list[str] = []
    indent_level = 0
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped:
            formatted_lines.append("")
            continue
        if stripped.startswith("}"):
            indent_level = max(indent_level - 1, 0)
        formatted_lines.append(f"{indent * indent_level}{stripped}")
        open_count = stripped.count("{")
        close_count = stripped.count("}")
        if stripped.startswith("}"):
            close_count = max(close_count - 1, 0)
        indent_level += open_count - close_count
        indent_level = max(indent_level, 0)
    return "\n".join(formatted_lines)


_SCALAR_TYPE_BY_DTYPE: dict[str, ScalarType] = {
    "float": ScalarType.F32,
    "double": ScalarType.F64,
    "int8": ScalarType.I8,
    "int16": ScalarType.I16,
    "int32": ScalarType.I32,
    "int64": ScalarType.I64,
    "uint8": ScalarType.U8,
    "uint16": ScalarType.U16,
    "uint32": ScalarType.U32,
    "uint64": ScalarType.U64,
    "bool": ScalarType.BOOL,
}

_C_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_]")
_C_KEYWORDS = {
    "_Bool",
    "_Complex",
    "_Imaginary",
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
}

@dataclass(frozen=True)
class BinaryOp:
    input0: str
    input1: str
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class MultiInputBinaryOp:
    inputs: tuple[str, ...]
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class WhereOp:
    condition: str
    input_x: str
    input_y: str
    output: str
    condition_shape: tuple[int, ...]
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class NodeInfo:
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object]


@dataclass(frozen=True)
class UnaryOp:
    input0: str
    output: str
    function: ScalarFunction
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class ClipOp:
    input0: str
    input_min: str | None
    input_max: str | None
    output: str
    input_shape: tuple[int, ...]
    min_shape: tuple[int, ...] | None
    max_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...]
    dtype: ScalarType




@dataclass(frozen=True)
class CastOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    input_dtype: ScalarType
    dtype: ScalarType


@dataclass(frozen=True)
class MatMulOp:
    input0: str
    input1: str
    output: str
    input0_shape: tuple[int, ...]
    input1_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    input0_batch_shape: tuple[int, ...]
    input1_batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    left_vector: bool
    right_vector: bool
    dtype: ScalarType


@dataclass(frozen=True)
class GemmOp:
    input_a: str
    input_b: str
    input_c: str | None
    output: str
    m: int
    n: int
    k: int
    trans_a: bool
    trans_b: bool
    alpha: float | int
    beta: float | int
    c_shape: tuple[int, ...] | None
    dtype: ScalarType


@dataclass(frozen=True)
class AttentionOp:
    input_q: str
    input_k: str
    input_v: str
    input_attn_mask: str | None
    input_past_key: str | None
    input_past_value: str | None
    input_nonpad_kv_seqlen: str | None
    output: str
    output_present_key: str | None
    output_present_value: str | None
    output_qk_matmul: str | None
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    total_seq: int
    past_seq: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int | None
    k_hidden_size: int | None
    v_hidden_size: int | None
    scale: float
    is_causal: bool
    softcap: float
    qk_matmul_output_mode: int
    q_rank: int
    k_rank: int
    v_rank: int
    output_rank: int
    mask_shape: tuple[int, ...] | None
    mask_is_bool: bool
    mask_rank: int | None
    mask_broadcast_batch: bool
    mask_broadcast_heads: bool
    mask_broadcast_q_seq: bool
    mask_q_seq: int | None
    mask_kv_seq: int | None
    head_group_size: int
    dtype: ScalarType


@dataclass(frozen=True)
class ConvOp:
    input0: str
    weights: str
    bias: str | None
    output: str
    batch: int
    in_channels: int
    out_channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    group: int
    dtype: ScalarType

    @property
    def out_h(self) -> int:
        if self.spatial_rank < 1:
            raise ValueError("Conv output height is undefined for spatial_rank < 1")
        return self.out_spatial[0]

    @property
    def out_w(self) -> int:
        if self.spatial_rank < 2:
            raise ValueError("Conv output width is undefined for spatial_rank < 2")
        return self.out_spatial[1]


@dataclass(frozen=True)
class AveragePoolOp:
    input0: str
    output: str
    batch: int
    channels: int
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool
    dtype: ScalarType


@dataclass(frozen=True)
class SoftmaxOp:
    input0: str
    output: str
    outer: int
    axis_size: int
    inner: int
    axis: int
    shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class LogSoftmaxOp:
    input0: str
    output: str
    outer: int
    axis_size: int
    inner: int
    axis: int
    shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class NegativeLogLikelihoodLossOp:
    input0: str
    target: str
    weight: str | None
    output: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    n: int
    c: int
    d: int
    reduction: str
    ignore_index: int
    input_dtype: ScalarType
    weight_dtype: ScalarType | None
    weight_shape: tuple[int, ...] | None
    dtype: ScalarType
    target_dtype: ScalarType


@dataclass(frozen=True)
class SoftmaxCrossEntropyLossOp:
    input0: str
    target: str
    weight: str | None
    output: str
    log_prob: str | None
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    log_prob_shape: tuple[int, ...] | None
    n: int
    c: int
    d: int
    reduction: str
    ignore_index: int | None
    input_dtype: ScalarType
    weight_dtype: ScalarType | None
    weight_shape: tuple[int, ...] | None
    dtype: ScalarType
    target_dtype: ScalarType


@dataclass(frozen=True)
class BatchNormOp:
    input0: str
    scale: str
    bias: str
    mean: str
    variance: str
    output: str
    shape: tuple[int, ...]
    channels: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class LpNormalizationOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    axis: int
    p: int
    outer: int
    axis_size: int
    inner: int
    dtype: ScalarType


@dataclass(frozen=True)
class InstanceNormalizationOp:
    input0: str
    scale: str
    bias: str
    output: str
    shape: tuple[int, ...]
    channels: int
    spatial_size: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class GroupNormalizationOp:
    input0: str
    scale: str
    bias: str
    output: str
    shape: tuple[int, ...]
    channels: int
    num_groups: int
    group_size: int
    spatial_size: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class LayerNormalizationOp:
    input0: str
    scale: str
    bias: str | None
    output: str
    mean_output: str | None
    invstd_output: str | None
    shape: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    bias_shape: tuple[int, ...] | None
    outer: int
    inner: int
    axis: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class MeanVarianceNormalizationOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    non_axes: tuple[int, ...]
    reduce_count: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class RMSNormalizationOp:
    input0: str
    scale: str
    output: str
    shape: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    outer: int
    inner: int
    axis: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class LrnOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    channels: int
    size: int
    half: int
    alpha: float
    beta: float
    bias: float
    dtype: ScalarType


@dataclass(frozen=True)
class LstmOp:
    input_x: str
    input_w: str
    input_r: str
    input_b: str | None
    input_sequence_lens: str | None
    input_initial_h: str | None
    input_initial_c: str | None
    input_p: str | None
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
    sequence_lens_dtype: ScalarType | None


@dataclass(frozen=True)
class MaxPoolOp:
    input0: str
    output: str
    indices: str | None
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    ceil_mode: bool
    storage_order: int
    dtype: ScalarType
    indices_dtype: ScalarType | None


@dataclass(frozen=True)
class ConcatOp:
    inputs: tuple[str, ...]
    output: str
    axis: int
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class GatherElementsOp:
    data: str
    indices: str
    output: str
    axis: int
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    indices_dtype: ScalarType


@dataclass(frozen=True)
class GatherOp:
    data: str
    indices: str
    output: str
    axis: int
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    indices_dtype: ScalarType


@dataclass(frozen=True)
class TransposeOp:
    input0: str
    output: str
    perm: tuple[int, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class ReshapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class IdentityOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class EyeLikeOp:
    input0: str
    output: str
    output_shape: tuple[int, ...]
    k: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class TileOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    repeats: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class DepthToSpaceOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    blocksize: int
    mode: str
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class SpaceToDepthOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    blocksize: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class SliceOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    starts: tuple[int, ...] | None
    steps: tuple[int, ...] | None
    axes: tuple[int, ...] | None
    starts_input: str | None
    ends_input: str | None
    axes_input: str | None
    steps_input: str | None
    starts_shape: tuple[int, ...] | None
    ends_shape: tuple[int, ...] | None
    axes_shape: tuple[int, ...] | None
    steps_shape: tuple[int, ...] | None
    starts_dtype: ScalarType | None
    ends_dtype: ScalarType | None
    axes_dtype: ScalarType | None
    steps_dtype: ScalarType | None
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class ResizeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    scales: tuple[float, ...]
    scales_input: str | None
    sizes_input: str | None
    roi_input: str | None
    axes: tuple[int, ...]
    scales_shape: tuple[int, ...] | None
    sizes_shape: tuple[int, ...] | None
    roi_shape: tuple[int, ...] | None
    scales_dtype: ScalarType | None
    sizes_dtype: ScalarType | None
    roi_dtype: ScalarType | None
    scales_axes: tuple[int, ...] | None
    sizes_axes: tuple[int, ...] | None
    roi_axes: tuple[int, ...] | None
    mode: str
    coordinate_transformation_mode: str
    nearest_mode: str
    cubic_coeff_a: float
    exclude_outside: bool
    extrapolation_value: float
    antialias: bool
    keep_aspect_ratio_policy: str
    dtype: ScalarType


@dataclass(frozen=True)
class ReduceOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axes: tuple[int, ...]
    axes_input: str | None
    axes_input_shape: tuple[int, ...] | None
    axes_input_dtype: ScalarType | None
    keepdims: bool
    noop_with_empty_axes: bool
    reduce_kind: str
    reduce_count: int | None
    dtype: ScalarType


@dataclass(frozen=True)
class ArgReduceOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: int
    keepdims: bool
    select_last_index: bool
    reduce_kind: str
    input_dtype: ScalarType
    output_dtype: ScalarType


@dataclass(frozen=True)
class ConstantOfShapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    shape: tuple[int, ...]
    value: float | int | bool
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class ShapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    values: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class SizeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    value: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class ExpandOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_shape_padded: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class RangeOp:
    start: str
    limit: str
    delta: str
    output: str
    output_shape: tuple[int, ...]
    length: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class SplitOp:
    input0: str
    outputs: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    axis: int
    split_sizes: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class ConstTensor:
    name: str
    shape: tuple[int, ...]
    data: tuple[float | int | bool, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class TempBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class ModelHeader:
    generator: str
    command_line: str | None
    model_checksum: str | None
    model_name: str | None
    graph_name: str | None
    description: str | None
    graph_description: str | None
    producer_name: str | None
    producer_version: str | None
    domain: str | None
    model_version: int | None
    ir_version: int | None
    opset_imports: tuple[tuple[str, int], ...]
    metadata_props: tuple[tuple[str, str], ...]
    input_count: int
    output_count: int
    node_count: int
    initializer_count: int


@dataclass(frozen=True)
class LoweredModel:
    name: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[ScalarType, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[ScalarType, ...]
    constants: tuple[ConstTensor, ...]
    ops: tuple[
        BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
        ...,
    ]
    node_infos: tuple[NodeInfo, ...]
    header: ModelHeader


class CEmitter:
    def __init__(self, template_dir: Path, *, restrict_arrays: bool = True) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._restrict_arrays = restrict_arrays

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        sanitized = _C_IDENTIFIER_RE.sub("_", name)
        if not sanitized:
            sanitized = "v"
        if sanitized[0].isdigit():
            sanitized = f"v_{sanitized}"
        return sanitized

    @staticmethod
    def _ensure_unique_identifier(base: str, used: set[str]) -> str:
        if base not in used and base not in _C_KEYWORDS:
            return base
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in used and candidate not in _C_KEYWORDS:
                return candidate
            index += 1

    @staticmethod
    def _op_names(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> tuple[str, ...]:
        if isinstance(op, BinaryOp):
            return (op.input0, op.input1, op.output)
        if isinstance(op, MultiInputBinaryOp):
            return (*op.inputs, op.output)
        if isinstance(op, WhereOp):
            return (op.condition, op.input_x, op.input_y, op.output)
        if isinstance(op, UnaryOp):
            return (op.input0, op.output)
        if isinstance(op, ClipOp):
            names = [op.input0]
            if op.input_min is not None:
                names.append(op.input_min)
            if op.input_max is not None:
                names.append(op.input_max)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, CastOp):
            return (op.input0, op.output)
        if isinstance(op, MatMulOp):
            return (op.input0, op.input1, op.output)
        if isinstance(op, GemmOp):
            names = [op.input_a, op.input_b]
            if op.input_c is not None:
                names.append(op.input_c)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, AttentionOp):
            names = [op.input_q, op.input_k, op.input_v]
            if op.input_attn_mask is not None:
                names.append(op.input_attn_mask)
            if op.input_past_key is not None:
                names.append(op.input_past_key)
            if op.input_past_value is not None:
                names.append(op.input_past_value)
            if op.input_nonpad_kv_seqlen is not None:
                names.append(op.input_nonpad_kv_seqlen)
            names.append(op.output)
            if op.output_present_key is not None:
                names.append(op.output_present_key)
            if op.output_present_value is not None:
                names.append(op.output_present_value)
            if op.output_qk_matmul is not None:
                names.append(op.output_qk_matmul)
            return tuple(names)
        if isinstance(op, ConvOp):
            names = [op.input0, op.weights]
            if op.bias is not None:
                names.append(op.bias)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, AveragePoolOp):
            return (op.input0, op.output)
        if isinstance(op, BatchNormOp):
            return (op.input0, op.scale, op.bias, op.mean, op.variance, op.output)
        if isinstance(op, LpNormalizationOp):
            return (op.input0, op.output)
        if isinstance(op, InstanceNormalizationOp):
            return (op.input0, op.scale, op.bias, op.output)
        if isinstance(op, GroupNormalizationOp):
            return (op.input0, op.scale, op.bias, op.output)
        if isinstance(op, LayerNormalizationOp):
            names = [op.input0, op.scale]
            if op.bias is not None:
                names.append(op.bias)
            names.append(op.output)
            if op.mean_output is not None:
                names.append(op.mean_output)
            if op.invstd_output is not None:
                names.append(op.invstd_output)
            return tuple(names)
        if isinstance(op, MeanVarianceNormalizationOp):
            return (op.input0, op.output)
        if isinstance(op, RMSNormalizationOp):
            return (op.input0, op.scale, op.output)
        if isinstance(op, LstmOp):
            names = [op.input_x, op.input_w, op.input_r]
            if op.input_b is not None:
                names.append(op.input_b)
            if op.input_sequence_lens is not None:
                names.append(op.input_sequence_lens)
            if op.input_initial_h is not None:
                names.append(op.input_initial_h)
            if op.input_initial_c is not None:
                names.append(op.input_initial_c)
            if op.input_p is not None:
                names.append(op.input_p)
            if op.output_y is not None:
                names.append(op.output_y)
            if op.output_y_h is not None:
                names.append(op.output_y_h)
            if op.output_y_c is not None:
                names.append(op.output_y_c)
            return tuple(names)
        if isinstance(op, (SoftmaxOp, LogSoftmaxOp)):
            return (op.input0, op.output)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            names = [op.input0, op.target]
            if op.weight is not None:
                names.append(op.weight)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            names = [op.input0, op.target]
            if op.weight is not None:
                names.append(op.weight)
            names.append(op.output)
            if op.log_prob is not None:
                names.append(op.log_prob)
            return tuple(names)
        if isinstance(op, MaxPoolOp):
            names = [op.input0, op.output]
            if op.indices is not None:
                names.append(op.indices)
            return tuple(names)
        if isinstance(op, GatherElementsOp):
            return (op.data, op.indices, op.output)
        if isinstance(op, GatherOp):
            return (op.data, op.indices, op.output)
        if isinstance(op, ConcatOp):
            return (*op.inputs, op.output)
        if isinstance(op, ConstantOfShapeOp):
            return (op.input0, op.output)
        if isinstance(op, ShapeOp):
            return (op.input0, op.output)
        if isinstance(op, SizeOp):
            return (op.input0, op.output)
        if isinstance(op, ExpandOp):
            return (op.input0, op.output)
        if isinstance(op, RangeOp):
            return (op.start, op.limit, op.delta, op.output)
        if isinstance(op, SplitOp):
            return (op.input0, *op.outputs)
        if isinstance(op, ReshapeOp):
            return (op.input0, op.output)
        if isinstance(op, IdentityOp):
            return (op.input0, op.output)
        if isinstance(op, EyeLikeOp):
            return (op.input0, op.output)
        if isinstance(op, TileOp):
            return (op.input0, op.output)
        if isinstance(op, DepthToSpaceOp):
            return (op.input0, op.output)
        if isinstance(op, SpaceToDepthOp):
            return (op.input0, op.output)
        if isinstance(op, SliceOp):
            names = [op.input0]
            if op.starts_input is not None:
                names.append(op.starts_input)
            if op.ends_input is not None:
                names.append(op.ends_input)
            if op.axes_input is not None:
                names.append(op.axes_input)
            if op.steps_input is not None:
                names.append(op.steps_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ResizeOp):
            names = [op.input0]
            if op.roi_input is not None:
                names.append(op.roi_input)
            if op.scales_input is not None:
                names.append(op.scales_input)
            if op.sizes_input is not None:
                names.append(op.sizes_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ReduceOp):
            names = [op.input0]
            if op.axes_input is not None:
                names.append(op.axes_input)
            names.append(op.output)
            return tuple(names)
        return (op.input0, op.output)

    def _build_name_map(self, model: LoweredModel) -> dict[str, str]:
        used: set[str] = set()
        name_map: dict[str, str] = {}
        names = [model.name]
        names.extend(model.input_names)
        names.extend(model.output_names)
        names.extend(const.name for const in model.constants)
        for op in model.ops:
            names.extend(self._op_names(op))
        for name in names:
            if name in name_map:
                continue
            sanitized = self._sanitize_identifier(name)
            unique = self._ensure_unique_identifier(sanitized, used)
            name_map[name] = unique
            used.add(unique)
        return name_map

    @staticmethod
    def _map_optional_name(
        name_map: dict[str, str], name: str | None
    ) -> str | None:
        if name is None:
            return None
        return name_map.get(name, name)

    def _map_op_names(
        self,
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
        name_map: dict[str, str],
    ) -> (
        BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp
    ):
        if isinstance(op, BinaryOp):
            return BinaryOp(
                input0=name_map.get(op.input0, op.input0),
                input1=name_map.get(op.input1, op.input1),
                output=name_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, MultiInputBinaryOp):
            return MultiInputBinaryOp(
                inputs=tuple(name_map.get(name, name) for name in op.inputs),
                output=name_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, WhereOp):
            return WhereOp(
                condition=name_map.get(op.condition, op.condition),
                input_x=name_map.get(op.input_x, op.input_x),
                input_y=name_map.get(op.input_y, op.input_y),
                output=name_map.get(op.output, op.output),
                condition_shape=op.condition_shape,
                x_shape=op.x_shape,
                y_shape=op.y_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, UnaryOp):
            return UnaryOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                function=op.function,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                params=op.params,
            )
        if isinstance(op, ClipOp):
            return ClipOp(
                input0=name_map.get(op.input0, op.input0),
                input_min=self._map_optional_name(name_map, op.input_min),
                input_max=self._map_optional_name(name_map, op.input_max),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                min_shape=op.min_shape,
                max_shape=op.max_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, CastOp):
            return CastOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                input_dtype=op.input_dtype,
                dtype=op.dtype,
            )
        if isinstance(op, MatMulOp):
            return MatMulOp(
                input0=name_map.get(op.input0, op.input0),
                input1=name_map.get(op.input1, op.input1),
                output=name_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                dtype=op.dtype,
            )
        if isinstance(op, GemmOp):
            return GemmOp(
                input_a=name_map.get(op.input_a, op.input_a),
                input_b=name_map.get(op.input_b, op.input_b),
                input_c=self._map_optional_name(name_map, op.input_c),
                output=name_map.get(op.output, op.output),
                m=op.m,
                n=op.n,
                k=op.k,
                trans_a=op.trans_a,
                trans_b=op.trans_b,
                alpha=op.alpha,
                beta=op.beta,
                c_shape=op.c_shape,
                dtype=op.dtype,
            )
        if isinstance(op, AttentionOp):
            return AttentionOp(
                input_q=name_map.get(op.input_q, op.input_q),
                input_k=name_map.get(op.input_k, op.input_k),
                input_v=name_map.get(op.input_v, op.input_v),
                input_attn_mask=self._map_optional_name(
                    name_map, op.input_attn_mask
                ),
                input_past_key=self._map_optional_name(
                    name_map, op.input_past_key
                ),
                input_past_value=self._map_optional_name(
                    name_map, op.input_past_value
                ),
                input_nonpad_kv_seqlen=self._map_optional_name(
                    name_map, op.input_nonpad_kv_seqlen
                ),
                output_present_key=self._map_optional_name(
                    name_map, op.output_present_key
                ),
                output_present_value=self._map_optional_name(
                    name_map, op.output_present_value
                ),
                output_qk_matmul=self._map_optional_name(
                    name_map, op.output_qk_matmul
                ),
                output=name_map.get(op.output, op.output),
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                scale=op.scale,
                is_causal=op.is_causal,
                softcap=op.softcap,
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                mask_shape=op.mask_shape,
                mask_is_bool=op.mask_is_bool,
                mask_rank=op.mask_rank,
                mask_broadcast_batch=op.mask_broadcast_batch,
                mask_broadcast_heads=op.mask_broadcast_heads,
                mask_broadcast_q_seq=op.mask_broadcast_q_seq,
                mask_q_seq=op.mask_q_seq,
                mask_kv_seq=op.mask_kv_seq,
                head_group_size=op.head_group_size,
                dtype=op.dtype,
            )
        if isinstance(op, ConvOp):
            return ConvOp(
                input0=name_map.get(op.input0, op.input0),
                weights=name_map.get(op.weights, op.weights),
                bias=self._map_optional_name(name_map, op.bias),
                output=name_map.get(op.output, op.output),
                batch=op.batch,
                in_channels=op.in_channels,
                out_channels=op.out_channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                group=op.group,
                dtype=op.dtype,
            )
        if isinstance(op, AveragePoolOp):
            return AveragePoolOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                batch=op.batch,
                channels=op.channels,
                in_h=op.in_h,
                in_w=op.in_w,
                out_h=op.out_h,
                out_w=op.out_w,
                kernel_h=op.kernel_h,
                kernel_w=op.kernel_w,
                stride_h=op.stride_h,
                stride_w=op.stride_w,
                pad_top=op.pad_top,
                pad_left=op.pad_left,
                pad_bottom=op.pad_bottom,
                pad_right=op.pad_right,
                count_include_pad=op.count_include_pad,
                dtype=op.dtype,
            )
        if isinstance(op, BatchNormOp):
            return BatchNormOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                mean=name_map.get(op.mean, op.mean),
                variance=name_map.get(op.variance, op.variance),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LpNormalizationOp):
            return LpNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                axis=op.axis,
                p=op.p,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                dtype=op.dtype,
            )
        if isinstance(op, InstanceNormalizationOp):
            return InstanceNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, GroupNormalizationOp):
            return GroupNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LayerNormalizationOp):
            return LayerNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=self._map_optional_name(name_map, op.bias),
                output=name_map.get(op.output, op.output),
                mean_output=self._map_optional_name(name_map, op.mean_output),
                invstd_output=self._map_optional_name(name_map, op.invstd_output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                bias_shape=op.bias_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, MeanVarianceNormalizationOp):
            return MeanVarianceNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, RMSNormalizationOp):
            return RMSNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LrnOp):
            return LrnOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                size=op.size,
                half=op.half,
                alpha=op.alpha,
                beta=op.beta,
                bias=op.bias,
                dtype=op.dtype,
            )
        if isinstance(op, LstmOp):
            return LstmOp(
                input_x=name_map.get(op.input_x, op.input_x),
                input_w=name_map.get(op.input_w, op.input_w),
                input_r=name_map.get(op.input_r, op.input_r),
                input_b=self._map_optional_name(name_map, op.input_b),
                input_sequence_lens=self._map_optional_name(
                    name_map, op.input_sequence_lens
                ),
                input_initial_h=self._map_optional_name(
                    name_map, op.input_initial_h
                ),
                input_initial_c=self._map_optional_name(
                    name_map, op.input_initial_c
                ),
                input_p=self._map_optional_name(name_map, op.input_p),
                output_y=self._map_optional_name(name_map, op.output_y),
                output_y_h=self._map_optional_name(name_map, op.output_y_h),
                output_y_c=self._map_optional_name(name_map, op.output_y_c),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                direction=op.direction,
                layout=op.layout,
                input_forget=op.input_forget,
                clip=op.clip,
                activation_kinds=op.activation_kinds,
                activation_alphas=op.activation_alphas,
                activation_betas=op.activation_betas,
                dtype=op.dtype,
                sequence_lens_dtype=op.sequence_lens_dtype,
            )
        if isinstance(op, SoftmaxOp):
            return SoftmaxOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, LogSoftmaxOp):
            return LogSoftmaxOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return NegativeLogLikelihoodLossOp(
                input0=name_map.get(op.input0, op.input0),
                target=name_map.get(op.target, op.target),
                weight=self._map_optional_name(name_map, op.weight),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                ignore_index=op.ignore_index,
                reduction=op.reduction,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return SoftmaxCrossEntropyLossOp(
                input0=name_map.get(op.input0, op.input0),
                target=name_map.get(op.target, op.target),
                weight=self._map_optional_name(name_map, op.weight),
                output=name_map.get(op.output, op.output),
                log_prob=self._map_optional_name(name_map, op.log_prob),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                log_prob_shape=op.log_prob_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                ignore_index=op.ignore_index,
                reduction=op.reduction,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, MaxPoolOp):
            return MaxPoolOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                indices=self._map_optional_name(name_map, op.indices),
                batch=op.batch,
                channels=op.channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                ceil_mode=op.ceil_mode,
                storage_order=op.storage_order,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, ConcatOp):
            return ConcatOp(
                inputs=tuple(
                    name_map.get(name, name) for name in op.inputs
                ),
                output=name_map.get(op.output, op.output),
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
            )
        if isinstance(op, GatherElementsOp):
            return GatherElementsOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                output=name_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherOp):
            return GatherOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                output=name_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, TransposeOp):
            return TransposeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                perm=op.perm,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ReshapeOp):
            return ReshapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, IdentityOp):
            return IdentityOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, EyeLikeOp):
            return EyeLikeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                output_shape=op.output_shape,
                k=op.k,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TileOp):
            return TileOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                repeats=op.repeats,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, DepthToSpaceOp):
            return DepthToSpaceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                mode=op.mode,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SpaceToDepthOp):
            return SpaceToDepthOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SliceOp):
            return SliceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                starts=op.starts,
                steps=op.steps,
                axes=op.axes,
                starts_input=self._map_optional_name(name_map, op.starts_input),
                ends_input=self._map_optional_name(name_map, op.ends_input),
                axes_input=self._map_optional_name(name_map, op.axes_input),
                steps_input=self._map_optional_name(name_map, op.steps_input),
                starts_shape=op.starts_shape,
                ends_shape=op.ends_shape,
                axes_shape=op.axes_shape,
                steps_shape=op.steps_shape,
                starts_dtype=op.starts_dtype,
                ends_dtype=op.ends_dtype,
                axes_dtype=op.axes_dtype,
                steps_dtype=op.steps_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ResizeOp):
            return ResizeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                roi_input=self._map_optional_name(name_map, op.roi_input),
                scales_input=self._map_optional_name(name_map, op.scales_input),
                sizes_input=self._map_optional_name(name_map, op.sizes_input),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                scales=op.scales,
                axes=op.axes,
                scales_shape=op.scales_shape,
                sizes_shape=op.sizes_shape,
                roi_shape=op.roi_shape,
                scales_dtype=op.scales_dtype,
                sizes_dtype=op.sizes_dtype,
                roi_dtype=op.roi_dtype,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                mode=op.mode,
                nearest_mode=op.nearest_mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                cubic_coeff_a=op.cubic_coeff_a,
                extrapolation_value=op.extrapolation_value,
                exclude_outside=op.exclude_outside,
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
                dtype=op.dtype,
            )
        if isinstance(op, ReduceOp):
            return ReduceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axes=op.axes,
                axes_input=self._map_optional_name(name_map, op.axes_input),
                axes_input_shape=op.axes_input_shape,
                axes_input_dtype=op.axes_input_dtype,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                reduce_kind=op.reduce_kind,
                reduce_count=op.reduce_count,
                dtype=op.dtype,
            )
        if isinstance(op, ArgReduceOp):
            return ArgReduceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                keepdims=op.keepdims,
                select_last_index=op.select_last_index,
                reduce_kind=op.reduce_kind,
                input_dtype=op.input_dtype,
                output_dtype=op.output_dtype,
            )
        if isinstance(op, ConstantOfShapeOp):
            return ConstantOfShapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                shape=op.shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ShapeOp):
            return ShapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                values=op.values,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SizeOp):
            return SizeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ExpandOp):
            return ExpandOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                input_shape_padded=op.input_shape_padded,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, RangeOp):
            return RangeOp(
                start=name_map.get(op.start, op.start),
                limit=name_map.get(op.limit, op.limit),
                delta=name_map.get(op.delta, op.delta),
                output=name_map.get(op.output, op.output),
                output_shape=op.output_shape,
                length=op.length,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SplitOp):
            return SplitOp(
                input0=name_map.get(op.input0, op.input0),
                outputs=tuple(
                    name_map.get(name, name) for name in op.outputs
                ),
                input_shape=op.input_shape,
                output_shapes=op.output_shapes,
                axis=op.axis,
                split_sizes=op.split_sizes,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        return UnaryOp(
            input0=name_map.get(op.input0, op.input0),
            output=name_map.get(op.output, op.output),
            function=op.function,
            shape=op.shape,
            dtype=op.dtype,
        )

    def _sanitize_model_names_with_map(
        self, model: LoweredModel
    ) -> tuple[LoweredModel, dict[str, str]]:
        name_map = self._build_name_map(model)
        constants = tuple(
            ConstTensor(
                name=name_map.get(const.name, const.name),
                shape=const.shape,
                data=const.data,
                dtype=const.dtype,
            )
            for const in model.constants
        )
        ops = tuple(self._map_op_names(op, name_map) for op in model.ops)
        sanitized = LoweredModel(
            name=name_map.get(model.name, model.name),
            input_names=tuple(
                name_map.get(name, name) for name in model.input_names
            ),
            input_shapes=model.input_shapes,
            input_dtypes=model.input_dtypes,
            output_names=tuple(
                name_map.get(name, name) for name in model.output_names
            ),
            output_shapes=model.output_shapes,
            output_dtypes=model.output_dtypes,
            constants=constants,
            ops=ops,
            node_infos=model.node_infos,
            header=model.header,
        )
        return sanitized, name_map

    def _sanitize_model_names(self, model: LoweredModel) -> LoweredModel:
        return self._sanitize_model_names_with_map(model)[0]

    @staticmethod
    def _sanitize_testbench_inputs(
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None,
        name_map: Mapping[str, str],
    ) -> Mapping[str, tuple[float | int | bool, ...]] | None:
        if not testbench_inputs:
            return None
        return {
            name_map.get(name, name): values
            for name, values in testbench_inputs.items()
        }

    def _load_templates(self, emit_testbench: bool) -> dict[str, Template]:
        try:
            templates = {
                "binary": self._env.get_template("binary_op.c.j2"),
                "multi_input": self._env.get_template(
                    "multi_input_op.c.j2"
                ),
                "where": self._env.get_template("where_op.c.j2"),
                "unary": self._env.get_template("unary_op.c.j2"),
                "clip": self._env.get_template("clip_op.c.j2"),
                "cast": self._env.get_template("cast_op.c.j2"),
                "matmul": self._env.get_template("matmul_op.c.j2"),
                "gemm": self._env.get_template("gemm_op.c.j2"),
                "attention": self._env.get_template("attention_op.c.j2"),
                "conv": self._env.get_template("conv_op.c.j2"),
                "avg_pool": self._env.get_template("average_pool_op.c.j2"),
                "batch_norm": self._env.get_template("batch_norm_op.c.j2"),
                "lp_norm": self._env.get_template("lp_normalization_op.c.j2"),
                "instance_norm": self._env.get_template(
                    "instance_normalization_op.c.j2"
                ),
                "group_norm": self._env.get_template(
                    "group_normalization_op.c.j2"
                ),
                "layer_norm": self._env.get_template(
                    "layer_normalization_op.c.j2"
                ),
                "mean_variance_norm": self._env.get_template(
                    "mean_variance_normalization_op.c.j2"
                ),
                "rms_norm": self._env.get_template("rms_normalization_op.c.j2"),
                "lrn": self._env.get_template("lrn_op.c.j2"),
                "lstm": self._env.get_template("lstm_op.c.j2"),
                "softmax": self._env.get_template("softmax_op.c.j2"),
                "logsoftmax": self._env.get_template("logsoftmax_op.c.j2"),
                "nllloss": self._env.get_template(
                    "negative_log_likelihood_loss_op.c.j2"
                ),
                "softmax_cross_entropy_loss": self._env.get_template(
                    "softmax_cross_entropy_loss_op.c.j2"
                ),
                "maxpool": self._env.get_template("maxpool_op.c.j2"),
                "concat": self._env.get_template("concat_op.c.j2"),
                "gather_elements": self._env.get_template("gather_elements_op.c.j2"),
                "gather": self._env.get_template("gather_op.c.j2"),
                "transpose": self._env.get_template("transpose_op.c.j2"),
                "reshape": self._env.get_template("reshape_op.c.j2"),
                "identity": self._env.get_template("identity_op.c.j2"),
                "eye_like": self._env.get_template("eye_like_op.c.j2"),
                "tile": self._env.get_template("tile_op.c.j2"),
                "depth_to_space": self._env.get_template("depth_to_space_op.c.j2"),
                "space_to_depth": self._env.get_template("space_to_depth_op.c.j2"),
                "slice": self._env.get_template("slice_op.c.j2"),
                "slice_dynamic": self._env.get_template(
                    "slice_op_dynamic.c.j2"
                ),
                "resize": self._env.get_template("resize_op.c.j2"),
                "reduce": self._env.get_template("reduce_op.c.j2"),
                "reduce_dynamic": self._env.get_template(
                    "reduce_op_dynamic.c.j2"
                ),
                "arg_reduce": self._env.get_template("arg_reduce_op.c.j2"),
                "constant_of_shape": self._env.get_template(
                    "constant_of_shape_op.c.j2"
                ),
                "shape": self._env.get_template("shape_op.c.j2"),
                "size": self._env.get_template("size_op.c.j2"),
                "expand": self._env.get_template("expand_op.c.j2"),
                "range": self._env.get_template("range_op.c.j2"),
                "split": self._env.get_template("split_op.c.j2"),
            }
            if emit_testbench:
                templates["testbench"] = self._env.get_template("testbench.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        return templates

    def emit_model(
        self,
        model: LoweredModel,
        *,
        emit_testbench: bool = False,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> str:
        model, name_map = self._sanitize_model_names_with_map(model)
        testbench_inputs = self._sanitize_testbench_inputs(
            testbench_inputs, name_map
        )
        (
            dim_order,
            input_dim_names,
            output_dim_names,
            dim_values,
        ) = self._build_variable_dim_names(
            model,
            variable_dim_inputs,
            variable_dim_outputs,
        )
        tensor_dim_names = self._build_tensor_dim_names(
            model,
            input_dim_names,
            output_dim_names,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        binary_template = templates["binary"]
        multi_input_template = templates["multi_input"]
        where_template = templates["where"]
        unary_template = templates["unary"]
        clip_template = templates["clip"]
        cast_template = templates["cast"]
        matmul_template = templates["matmul"]
        gemm_template = templates["gemm"]
        attention_template = templates["attention"]
        conv_template = templates["conv"]
        avg_pool_template = templates["avg_pool"]
        batch_norm_template = templates["batch_norm"]
        lp_norm_template = templates["lp_norm"]
        instance_norm_template = templates["instance_norm"]
        group_norm_template = templates["group_norm"]
        layer_norm_template = templates["layer_norm"]
        mean_variance_norm_template = templates["mean_variance_norm"]
        rms_norm_template = templates["rms_norm"]
        lrn_template = templates["lrn"]
        lstm_template = templates["lstm"]
        softmax_template = templates["softmax"]
        logsoftmax_template = templates["logsoftmax"]
        nllloss_template = templates["nllloss"]
        softmax_cross_entropy_loss_template = templates["softmax_cross_entropy_loss"]
        maxpool_template = templates["maxpool"]
        concat_template = templates["concat"]
        gather_elements_template = templates["gather_elements"]
        gather_template = templates["gather"]
        transpose_template = templates["transpose"]
        reshape_template = templates["reshape"]
        identity_template = templates["identity"]
        eye_like_template = templates["eye_like"]
        tile_template = templates["tile"]
        depth_to_space_template = templates["depth_to_space"]
        space_to_depth_template = templates["space_to_depth"]
        slice_template = templates["slice"]
        slice_dynamic_template = templates["slice_dynamic"]
        resize_template = templates["resize"]
        reduce_template = templates["reduce"]
        reduce_dynamic_template = templates["reduce_dynamic"]
        arg_reduce_template = templates["arg_reduce"]
        constant_of_shape_template = templates["constant_of_shape"]
        shape_template = templates["shape"]
        size_template = templates["size"]
        expand_template = templates["expand"]
        range_template = templates["range"]
        split_template = templates["split"]
        testbench_template = templates.get("testbench")
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
        operator_fns = "\n\n".join(
            self._render_op(
                model,
                op,
                index,
                array_suffix="",
                loop_vars=(),
                c_type=self._op_output_dtype(op).c_type,
                zero_literal=self._op_output_dtype(op).zero_literal,
                min_literal=self._op_output_dtype(op).min_literal,
                max_literal=self._op_output_dtype(op).max_literal,
                binary_template=binary_template,
                multi_input_template=multi_input_template,
                where_template=where_template,
                unary_template=unary_template,
                clip_template=clip_template,
                cast_template=cast_template,
                matmul_template=matmul_template,
                gemm_template=gemm_template,
                attention_template=attention_template,
                conv_template=conv_template,
                avg_pool_template=avg_pool_template,
                batch_norm_template=batch_norm_template,
                lp_norm_template=lp_norm_template,
                instance_norm_template=instance_norm_template,
                group_norm_template=group_norm_template,
                layer_norm_template=layer_norm_template,
                mean_variance_norm_template=mean_variance_norm_template,
                rms_norm_template=rms_norm_template,
                lrn_template=lrn_template,
                lstm_template=lstm_template,
                softmax_template=softmax_template,
                logsoftmax_template=logsoftmax_template,
                nllloss_template=nllloss_template,
                softmax_cross_entropy_loss_template=softmax_cross_entropy_loss_template,
                maxpool_template=maxpool_template,
                concat_template=concat_template,
                gather_elements_template=gather_elements_template,
                gather_template=gather_template,
                transpose_template=transpose_template,
                reshape_template=reshape_template,
                identity_template=identity_template,
                eye_like_template=eye_like_template,
                tile_template=tile_template,
                depth_to_space_template=depth_to_space_template,
                space_to_depth_template=space_to_depth_template,
                slice_template=slice_template,
                slice_dynamic_template=slice_dynamic_template,
                resize_template=resize_template,
                reduce_template=reduce_template,
                reduce_dynamic_template=reduce_dynamic_template,
                arg_reduce_template=arg_reduce_template,
                constant_of_shape_template=constant_of_shape_template,
                shape_template=shape_template,
                size_template=size_template,
                expand_template=expand_template,
                range_template=range_template,
                split_template=split_template,
                scalar_registry=scalar_registry,
                dim_args=dim_args,
                tensor_dim_names=tensor_dim_names,
            )
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        scalar_functions = scalar_registry.render()
        scalar_include_lines = (
            scalar_registry.include_lines() if scalar_functions else []
        )
        scalar_includes = {
            line for line in scalar_include_lines if line.startswith("#include ")
        }
        scalar_preamble = [
            line for line in scalar_include_lines if not line.startswith("#include ")
        ]
        includes = self._collect_includes(
            model,
            resolved_ops,
            emit_testbench=emit_testbench,
            extra_includes=scalar_includes,
        )
        sections = [self._emit_header_comment(model.header), "", *includes]
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        sections.append("")
        constants_section = self._emit_constant_definitions(model.constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        sections.extend(
            (
                operator_fns.rstrip(),
                "",
                wrapper_fn,
            )
        )
        if emit_testbench and testbench_template is not None:
            sections.extend(
                (
                    "",
                    self._emit_testbench(
                        model,
                        testbench_template,
                        testbench_inputs=testbench_inputs,
                        dim_order=dim_order,
                        dim_values=dim_values,
                    ),
                )
            )
        sections.append("")
        rendered = "\n".join(sections)
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def emit_model_with_data_file(
        self,
        model: LoweredModel,
        *,
        emit_testbench: bool = False,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> tuple[str, str]:
        model, name_map = self._sanitize_model_names_with_map(model)
        testbench_inputs = self._sanitize_testbench_inputs(
            testbench_inputs, name_map
        )
        (
            dim_order,
            input_dim_names,
            output_dim_names,
            dim_values,
        ) = self._build_variable_dim_names(
            model,
            variable_dim_inputs,
            variable_dim_outputs,
        )
        tensor_dim_names = self._build_tensor_dim_names(
            model,
            input_dim_names,
            output_dim_names,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        binary_template = templates["binary"]
        multi_input_template = templates["multi_input"]
        where_template = templates["where"]
        unary_template = templates["unary"]
        clip_template = templates["clip"]
        cast_template = templates["cast"]
        matmul_template = templates["matmul"]
        gemm_template = templates["gemm"]
        attention_template = templates["attention"]
        conv_template = templates["conv"]
        avg_pool_template = templates["avg_pool"]
        batch_norm_template = templates["batch_norm"]
        lp_norm_template = templates["lp_norm"]
        instance_norm_template = templates["instance_norm"]
        group_norm_template = templates["group_norm"]
        layer_norm_template = templates["layer_norm"]
        mean_variance_norm_template = templates["mean_variance_norm"]
        rms_norm_template = templates["rms_norm"]
        lrn_template = templates["lrn"]
        lstm_template = templates["lstm"]
        softmax_template = templates["softmax"]
        logsoftmax_template = templates["logsoftmax"]
        nllloss_template = templates["nllloss"]
        softmax_cross_entropy_loss_template = templates["softmax_cross_entropy_loss"]
        maxpool_template = templates["maxpool"]
        concat_template = templates["concat"]
        gather_elements_template = templates["gather_elements"]
        gather_template = templates["gather"]
        transpose_template = templates["transpose"]
        reshape_template = templates["reshape"]
        identity_template = templates["identity"]
        eye_like_template = templates["eye_like"]
        tile_template = templates["tile"]
        depth_to_space_template = templates["depth_to_space"]
        space_to_depth_template = templates["space_to_depth"]
        slice_template = templates["slice"]
        slice_dynamic_template = templates["slice_dynamic"]
        resize_template = templates["resize"]
        reduce_template = templates["reduce"]
        reduce_dynamic_template = templates["reduce_dynamic"]
        arg_reduce_template = templates["arg_reduce"]
        constant_of_shape_template = templates["constant_of_shape"]
        shape_template = templates["shape"]
        size_template = templates["size"]
        expand_template = templates["expand"]
        range_template = templates["range"]
        split_template = templates["split"]
        testbench_template = templates.get("testbench")
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
        operator_fns = "\n\n".join(
            self._render_op(
                model,
                op,
                index,
                array_suffix="",
                loop_vars=(),
                c_type=self._op_output_dtype(op).c_type,
                zero_literal=self._op_output_dtype(op).zero_literal,
                min_literal=self._op_output_dtype(op).min_literal,
                max_literal=self._op_output_dtype(op).max_literal,
                binary_template=binary_template,
                multi_input_template=multi_input_template,
                where_template=where_template,
                unary_template=unary_template,
                clip_template=clip_template,
                cast_template=cast_template,
                matmul_template=matmul_template,
                gemm_template=gemm_template,
                attention_template=attention_template,
                conv_template=conv_template,
                avg_pool_template=avg_pool_template,
                batch_norm_template=batch_norm_template,
                lp_norm_template=lp_norm_template,
                instance_norm_template=instance_norm_template,
                group_norm_template=group_norm_template,
                layer_norm_template=layer_norm_template,
                mean_variance_norm_template=mean_variance_norm_template,
                rms_norm_template=rms_norm_template,
                lrn_template=lrn_template,
                lstm_template=lstm_template,
                softmax_template=softmax_template,
                logsoftmax_template=logsoftmax_template,
                nllloss_template=nllloss_template,
                softmax_cross_entropy_loss_template=softmax_cross_entropy_loss_template,
                maxpool_template=maxpool_template,
                concat_template=concat_template,
                gather_elements_template=gather_elements_template,
                gather_template=gather_template,
                transpose_template=transpose_template,
                reshape_template=reshape_template,
                identity_template=identity_template,
                eye_like_template=eye_like_template,
                tile_template=tile_template,
                depth_to_space_template=depth_to_space_template,
                space_to_depth_template=space_to_depth_template,
                slice_template=slice_template,
                slice_dynamic_template=slice_dynamic_template,
                resize_template=resize_template,
                reduce_template=reduce_template,
                reduce_dynamic_template=reduce_dynamic_template,
                arg_reduce_template=arg_reduce_template,
                constant_of_shape_template=constant_of_shape_template,
                shape_template=shape_template,
                size_template=size_template,
                expand_template=expand_template,
                range_template=range_template,
                split_template=split_template,
                scalar_registry=scalar_registry,
                dim_args=dim_args,
                tensor_dim_names=tensor_dim_names,
            )
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        scalar_functions = scalar_registry.render()
        scalar_include_lines = (
            scalar_registry.include_lines() if scalar_functions else []
        )
        scalar_includes = {
            line for line in scalar_include_lines if line.startswith("#include ")
        }
        scalar_preamble = [
            line for line in scalar_include_lines if not line.startswith("#include ")
        ]
        includes = self._collect_includes(
            model,
            resolved_ops,
            emit_testbench=emit_testbench,
            extra_includes=scalar_includes,
        )
        sections = [self._emit_header_comment(model.header), "", *includes]
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        sections.append("")
        constants_section = self._emit_constant_declarations(model.constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        sections.extend(
            (
                operator_fns.rstrip(),
                "",
                wrapper_fn,
            )
        )
        if emit_testbench and testbench_template is not None:
            sections.extend(
                (
                    "",
                    self._emit_testbench(
                        model,
                        testbench_template,
                        testbench_inputs=testbench_inputs,
                        dim_order=dim_order,
                        dim_values=dim_values,
                    ),
                )
            )
        sections.append("")
        main_rendered = "\n".join(sections)
        if not main_rendered.endswith("\n"):
            main_rendered += "\n"
        data_includes = self._collect_constant_includes(model.constants)
        data_sections = [self._emit_header_comment(model.header), ""]
        if data_includes:
            data_sections.extend((*data_includes, ""))
        else:
            data_sections.append("")
        data_constants = self._emit_constant_definitions(
            model.constants, storage_prefix="const"
        )
        if data_constants:
            data_sections.append(data_constants.rstrip())
        data_sections.append("")
        data_rendered = "\n".join(data_sections)
        if not data_rendered.endswith("\n"):
            data_rendered += "\n"
        return main_rendered, data_rendered

    @staticmethod
    def _emit_header_comment(header: ModelHeader) -> str:
        lines: list[str] = [header.generator, ""]
        lines.append(f"Command line: {header.command_line or 'n/a'}")
        lines.append(f"Model checksum (sha256): {header.model_checksum or 'n/a'}")
        lines.append(f"Model name: {header.model_name or 'n/a'}")
        lines.append(f"Graph name: {header.graph_name or 'n/a'}")
        lines.append(
            "Inputs: "
            f"{header.input_count} Outputs: {header.output_count} "
            f"Nodes: {header.node_count} Initializers: {header.initializer_count}"
        )
        lines.append(f"IR version: {header.ir_version or 'n/a'}")
        lines.append(f"Model version: {header.model_version or 'n/a'}")
        lines.append(f"Domain: {header.domain or 'n/a'}")
        producer = header.producer_name or "n/a"
        producer_version = header.producer_version or "n/a"
        lines.append(f"Producer: {producer} (version: {producer_version})")
        if header.opset_imports:
            opset_items = []
            for domain, version in header.opset_imports:
                opset_domain = domain or "ai.onnx"
                opset_items.append(f"{opset_domain}={version}")
            lines.append(f"Opset imports: {', '.join(opset_items)}")
        else:
            lines.append("Opset imports: n/a")
        lines.append("Description:")
        lines.extend(_format_multiline_value(header.description))
        lines.append("Graph description:")
        lines.extend(_format_multiline_value(header.graph_description))
        lines.append("Metadata:")
        if header.metadata_props:
            for key, value in header.metadata_props:
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  n/a")
        comment_lines = ["/*"]
        comment_lines.extend(
            f" * {line}" if line else " *" for line in lines
        )
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _format_node_attr_value(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(CEmitter._format_node_attr_value(item) for item in value)
            return f"[{rendered}]"
        if hasattr(value, "tolist"):
            try:
                return CEmitter._format_node_attr_value(value.tolist())
            except Exception:
                return repr(value)
        return repr(value)

    @staticmethod
    def _emit_node_comment(node_info: NodeInfo, index: int) -> str:
        lines = [
            f"Node {index}:",
            f"OpType: {node_info.op_type}",
            "Inputs: "
            + (", ".join(node_info.inputs) if node_info.inputs else "n/a"),
            "Outputs: "
            + (", ".join(node_info.outputs) if node_info.outputs else "n/a"),
        ]
        if node_info.attrs:
            lines.append("Attrs:")
            for key, value in sorted(node_info.attrs.items()):
                rendered = CEmitter._format_node_attr_value(value)
                lines.append(f"  {key}: {rendered}")
        else:
            lines.append("Attrs: n/a")
        comment_lines = ["/*"]
        comment_lines.extend(
            f" * {line}" if line else " *" for line in lines
        )
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _collect_constant_includes(constants: tuple[ConstTensor, ...]) -> list[str]:
        if not constants:
            return []
        includes: list[str] = []
        dtypes = {const.dtype for const in constants}
        if any(
            dtype
            in {
                ScalarType.I64,
                ScalarType.I32,
                ScalarType.I16,
                ScalarType.I8,
                ScalarType.U64,
                ScalarType.U32,
                ScalarType.U16,
                ScalarType.U8,
            }
            for dtype in dtypes
        ):
            includes.append("#include <stdint.h>")
        if ScalarType.BOOL in dtypes:
            includes.append("#include <stdbool.h>")
        return includes

    @staticmethod
    def _scalar_function_name(
        function: ScalarFunction,
        dtype: ScalarType,
        registry: ScalarFunctionRegistry,
        *,
        params: tuple[float, ...] = (),
    ) -> str | None:
        registry_functions = {
            ScalarFunction.ABS,
            ScalarFunction.ADD,
            ScalarFunction.ACOS,
            ScalarFunction.ACOSH,
            ScalarFunction.LOGICAL_AND,
            ScalarFunction.ASIN,
            ScalarFunction.ASINH,
            ScalarFunction.ATAN,
            ScalarFunction.ATANH,
            ScalarFunction.BITWISE_AND,
            ScalarFunction.BITWISE_NOT,
            ScalarFunction.BITWISE_OR,
            ScalarFunction.BITWISE_XOR,
            ScalarFunction.CEIL,
            ScalarFunction.CELU,
            ScalarFunction.COS,
            ScalarFunction.COSH,
            ScalarFunction.DIV,
            ScalarFunction.ELU,
            ScalarFunction.ERF,
            ScalarFunction.EXP,
            ScalarFunction.FLOOR,
            ScalarFunction.GELU,
            ScalarFunction.HARDSIGMOID,
            ScalarFunction.HARDSWISH,
            ScalarFunction.LOG,
            ScalarFunction.FMOD,
            ScalarFunction.REMAINDER,
            ScalarFunction.LEAKY_RELU,
            ScalarFunction.MUL,
            ScalarFunction.NEG,
            ScalarFunction.LOGICAL_NOT,
            ScalarFunction.LOGICAL_OR,
            ScalarFunction.POW,
            ScalarFunction.RECIPROCAL,
            ScalarFunction.RELU,
            ScalarFunction.ROUND,
            ScalarFunction.SELU,
            ScalarFunction.SHRINK,
            ScalarFunction.SIGMOID,
            ScalarFunction.SIGN,
            ScalarFunction.SIN,
            ScalarFunction.SINH,
            ScalarFunction.SOFTPLUS,
            ScalarFunction.SOFTSIGN,
            ScalarFunction.SQRT,
            ScalarFunction.SUB,
            ScalarFunction.SWISH,
            ScalarFunction.TAN,
            ScalarFunction.TANH,
            ScalarFunction.THRESHOLDED_RELU,
            ScalarFunction.LOGICAL_XOR,
        }
        if function in {ScalarFunction.MAXIMUM, ScalarFunction.MINIMUM}:
            if dtype in {ScalarType.F32, ScalarType.F64}:
                scalar_function = (
                    ScalarFunction.FMAX
                    if function == ScalarFunction.MAXIMUM
                    else ScalarFunction.FMIN
                )
            else:
                scalar_function = function
        elif function in registry_functions:
            scalar_function = function
        else:
            return None
        try:
            return registry.request(
                ScalarFunctionKey(
                    function=scalar_function, return_type=dtype, params=params
                )
            )
        except ScalarFunctionError:
            return None

    @staticmethod
    def _collect_includes(
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
        *,
        emit_testbench: bool,
        extra_includes: set[str] | None = None,
    ) -> list[str]:
        includes: set[str] = {"#include <stddef.h>"}
        if emit_testbench:
            includes.update({"#include <stdio.h>", "#include <stdint.h>"})
        if extra_includes:
            includes.update(extra_includes)
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}
            for op in resolved_ops
        ):
            includes.add("#include <math.h>")
        constant_of_shape_inputs = {
            op.input_dtype
            for op in resolved_ops
            if isinstance(op, ConstantOfShapeOp)
        }
        model_dtypes = {
            *model.input_dtypes,
            *model.output_dtypes,
            *(const.dtype for const in model.constants),
            *constant_of_shape_inputs,
        }
        model_dtypes.update(
            op.dtype for op in resolved_ops if not isinstance(op, ArgReduceOp)
        )
        arg_reduce_dtypes = {
            dtype
            for op in resolved_ops
            if isinstance(op, ArgReduceOp)
            for dtype in (op.input_dtype, op.output_dtype)
        }
        model_dtypes.update(arg_reduce_dtypes)
        slice_input_dtypes = {
            dtype
            for op in resolved_ops
            if isinstance(op, SliceOp)
            for dtype in (
                op.starts_dtype,
                op.ends_dtype,
                op.axes_dtype,
                op.steps_dtype,
            )
            if dtype is not None
        }
        maxpool_indices_dtypes = {
            op.indices_dtype
            for op in resolved_ops
            if isinstance(op, MaxPoolOp) and op.indices_dtype is not None
        }
        model_dtypes.update(maxpool_indices_dtypes)
        nll_target_dtypes = {
            op.target_dtype
            for op in resolved_ops
            if isinstance(op, NegativeLogLikelihoodLossOp)
        }
        sce_target_dtypes = {
            op.target_dtype
            for op in resolved_ops
            if isinstance(op, SoftmaxCrossEntropyLossOp)
        }
        if CEmitter._needs_stdint(
            model_dtypes,
            (nll_target_dtypes, sce_target_dtypes, slice_input_dtypes),
            has_resize=any(isinstance(op, ResizeOp) for op in resolved_ops),
        ):
            includes.add("#include <stdint.h>")
        if ScalarType.BOOL in model_dtypes:
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, ReduceOp) and op.axes_input is not None
            for op in resolved_ops
        ):
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, UnaryOp)
            and unary_op_symbol(op.function, dtype=op.dtype) in {"llabs", "abs"}
            for op in resolved_ops
        ):
            includes.add("#include <stdlib.h>")
        if CEmitter._needs_math(resolved_ops):
            includes.add("#include <math.h>")
        if CEmitter._needs_limits(resolved_ops):
            includes.add("#include <limits.h>")
        if any(
            isinstance(op, (ConcatOp, ReshapeOp, SplitOp, IdentityOp))
            for op in resolved_ops
        ):
            includes.add("#include <string.h>")
        ordered_includes = (
            "#include <stddef.h>",
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <float.h>",
            "#include <limits.h>",
            "#include <string.h>",
        )
        return [include for include in ordered_includes if include in includes]

    @staticmethod
    def _needs_stdint(
        model_dtypes: set[ScalarType],
        targets: tuple[set[ScalarType], ...],
        *,
        has_resize: bool,
    ) -> bool:
        integer_dtypes = {
            ScalarType.I64,
            ScalarType.I32,
            ScalarType.I16,
            ScalarType.I8,
            ScalarType.U64,
            ScalarType.U32,
            ScalarType.U16,
            ScalarType.U8,
        }
        if any(dtype in integer_dtypes for dtype in model_dtypes):
            return True
        if any(
            dtype in {ScalarType.I64, ScalarType.I32}
            for target_dtypes in targets
            for dtype in target_dtypes
        ):
            return True
        return has_resize

    @staticmethod
    def _needs_math(
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
    ) -> bool:
        math_ops = {
            "atanhf",
            "ceilf",
            "cosf",
            "expf",
            "fabsf",
            "floorf",
            "logf",
            "sinf",
            "sqrtf",
            "tanf",
            "tanhf",
        }
        binary_math_ops = {"fmaxf", "fminf", "fmodf", "powf"}

        def is_binary_math_op(op: BinaryOp) -> bool:
            op_spec = binary_op_symbol(
                op.function, dtype=op.input_dtype, validate_attrs=False
            )
            return op_spec is not None and op_spec.operator in binary_math_ops

        if any(
            isinstance(op, UnaryOp)
            and unary_op_symbol(op.function, dtype=op.dtype) in math_ops
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.CELU, ScalarFunction.SWISH}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ClipOp)
            and op.dtype.is_float
            and (op.input_min is None or op.input_max is None)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, BinaryOp) and is_binary_math_op(op)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(
                op,
                (
                    AttentionOp,
                    BatchNormOp,
                    LpNormalizationOp,
                    InstanceNormalizationOp,
                    GroupNormalizationOp,
                    LayerNormalizationOp,
                    MeanVarianceNormalizationOp,
                    RMSNormalizationOp,
                    LrnOp,
                    LstmOp,
                    SoftmaxOp,
                    LogSoftmaxOp,
                    SoftmaxCrossEntropyLossOp,
                    ResizeOp,
                ),
            )
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"l1", "l2", "logsum", "logsumexp"}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"min", "max"}
            and op.dtype.is_float
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp) and op.dtype.is_float
            for op in resolved_ops
        ):
            return True
        return False

    @staticmethod
    def _needs_limits(
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
    ) -> bool:
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"min", "max"}
            and op.dtype in {
                ScalarType.I64,
                ScalarType.I32,
                ScalarType.I16,
                ScalarType.I8,
            }
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ClipOp)
            and op.dtype.is_integer
            and (op.input_min is None or op.input_max is None)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp)
            and op.dtype
            in {ScalarType.I64, ScalarType.I32, ScalarType.I16, ScalarType.I8}
            for op in resolved_ops
        ):
            return True
        return False

    def _emit_model_wrapper(
        self,
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
        temp_buffers: tuple[TempBuffer, ...],
        *,
        dim_order: Sequence[str],
        input_dim_names: Mapping[int, Mapping[int, str]],
        output_dim_names: Mapping[int, Mapping[int, str]],
    ) -> str:
        params = []
        if dim_order:
            params.extend(self._format_dim_args(dim_order))
        for index, (name, shape, dtype) in enumerate(
            zip(model.input_names, model.input_shapes, model.input_dtypes)
        ):
            params.append(
                f"const {dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, input_dim_names.get(index))}"
            )
        for index, (name, shape, dtype) in enumerate(
            zip(model.output_names, model.output_shapes, model.output_dtypes)
        ):
            params.append(
                f"{dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, output_dim_names.get(index))}"
            )
        signature = ", ".join(params)
        lines = [f"void {model.name}({signature}) {{"]
        for temp in temp_buffers:
            c_type = temp.dtype.c_type
            lines.append(
                f"    {c_type} {temp.name}{self._array_suffix(temp.shape)};"
            )
        for index, op in enumerate(resolved_ops):
            call = self._build_op_call(op, dim_order)
            lines.append(f"    {model.name}_op{index}({call});")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _build_op_call(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
        dim_order: Sequence[str],
    ) -> str:
        args: list[str] = []
        if dim_order:
            args.extend(dim_order)
        if isinstance(op, BinaryOp):
            args.extend([op.input0, op.input1, op.output])
            return ", ".join(args)
        if isinstance(op, WhereOp):
            args.extend([op.condition, op.input_x, op.input_y, op.output])
            return ", ".join(args)
        if isinstance(op, MatMulOp):
            args.extend([op.input0, op.input1, op.output])
            return ", ".join(args)
        if isinstance(op, GemmOp):
            if op.input_c is None:
                args.extend([op.input_a, op.input_b, op.output])
                return ", ".join(args)
            args.extend([op.input_a, op.input_b, op.input_c, op.output])
            return ", ".join(args)
        if isinstance(op, UnaryOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ClipOp):
            call_parts = [op.input0]
            if op.input_min is not None:
                call_parts.append(op.input_min)
            if op.input_max is not None:
                call_parts.append(op.input_max)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, AttentionOp):
            call_parts = [op.input_q, op.input_k, op.input_v]
            if op.input_attn_mask is not None:
                call_parts.append(op.input_attn_mask)
            if op.input_past_key is not None:
                call_parts.append(op.input_past_key)
            if op.input_past_value is not None:
                call_parts.append(op.input_past_value)
            if op.input_nonpad_kv_seqlen is not None:
                call_parts.append(op.input_nonpad_kv_seqlen)
            call_parts.append(op.output)
            if op.output_present_key is not None:
                call_parts.append(op.output_present_key)
            if op.output_present_value is not None:
                call_parts.append(op.output_present_value)
            if op.output_qk_matmul is not None:
                call_parts.append(op.output_qk_matmul)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ConvOp):
            if op.bias is None:
                args.extend([op.input0, op.weights, op.output])
                return ", ".join(args)
            args.extend([op.input0, op.weights, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, AveragePoolOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, BatchNormOp):
            args.extend(
                [op.input0, op.scale, op.bias, op.mean, op.variance, op.output]
            )
            return ", ".join(args)
        if isinstance(op, LpNormalizationOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, InstanceNormalizationOp):
            args.extend([op.input0, op.scale, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, GroupNormalizationOp):
            args.extend([op.input0, op.scale, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, LayerNormalizationOp):
            call_parts = [op.input0, op.scale]
            if op.bias is not None:
                call_parts.append(op.bias)
            call_parts.append(op.output)
            if op.mean_output is not None:
                call_parts.append(op.mean_output)
            if op.invstd_output is not None:
                call_parts.append(op.invstd_output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, MeanVarianceNormalizationOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, RMSNormalizationOp):
            args.extend([op.input0, op.scale, op.output])
            return ", ".join(args)
        if isinstance(op, LstmOp):
            call_parts = [op.input_x, op.input_w, op.input_r]
            if op.input_b is not None:
                call_parts.append(op.input_b)
            if op.input_sequence_lens is not None:
                call_parts.append(op.input_sequence_lens)
            if op.input_initial_h is not None:
                call_parts.append(op.input_initial_h)
            if op.input_initial_c is not None:
                call_parts.append(op.input_initial_c)
            if op.input_p is not None:
                call_parts.append(op.input_p)
            if op.output_y is not None:
                call_parts.append(op.output_y)
            if op.output_y_h is not None:
                call_parts.append(op.output_y_h)
            if op.output_y_c is not None:
                call_parts.append(op.output_y_c)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, (SoftmaxOp, LogSoftmaxOp)):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            if op.log_prob is not None:
                call_parts.append(op.log_prob)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, MaxPoolOp):
            if op.indices is not None:
                args.extend([op.input0, op.output, op.indices])
                return ", ".join(args)
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, GatherElementsOp):
            args.extend([op.data, op.indices, op.output])
            return ", ".join(args)
        if isinstance(op, GatherOp):
            args.extend([op.data, op.indices, op.output])
            return ", ".join(args)
        if isinstance(op, ConcatOp):
            args.extend([*op.inputs, op.output])
            return ", ".join(args)
        if isinstance(op, ConstantOfShapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ShapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, SizeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ExpandOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, RangeOp):
            args.extend([op.start, op.limit, op.delta, op.output])
            return ", ".join(args)
        if isinstance(op, SplitOp):
            args.extend([op.input0, *op.outputs])
            return ", ".join(args)
        if isinstance(op, ReshapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, SliceOp):
            call_parts = [op.input0]
            if op.starts_input is not None:
                call_parts.append(op.starts_input)
            if op.ends_input is not None:
                call_parts.append(op.ends_input)
            if op.axes_input is not None:
                call_parts.append(op.axes_input)
            if op.steps_input is not None:
                call_parts.append(op.steps_input)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ResizeOp):
            call_parts = [op.input0]
            if op.roi_input is not None:
                call_parts.append(op.roi_input)
            if op.scales_input is not None:
                call_parts.append(op.scales_input)
            if op.sizes_input is not None:
                call_parts.append(op.sizes_input)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ReduceOp):
            if op.axes_input is not None:
                args.extend([op.input0, op.axes_input, op.output])
                return ", ".join(args)
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ArgReduceOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        args.extend([op.input0, op.output])
        return ", ".join(args)

    def _temp_buffers(
        self, model: LoweredModel, *, reserved_names: set[str] | None = None
    ) -> dict[str, TempBuffer]:
        output_names = set(model.output_names)
        used_names = set(reserved_names or ())

        def allocate_temp_name(base: str) -> str:
            if base not in used_names:
                used_names.add(base)
                return base
            index = 0
            while True:
                candidate = f"{base}{index}"
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate
                index += 1

        intermediates = [
            (name, shape, dtype)
            for op in model.ops
            for name, shape, dtype in self._op_outputs(op)
            if name not in output_names
        ]
        if not intermediates:
            return {}
        if len(intermediates) == 1:
            name, shape, dtype = intermediates[0]
            temp_name = allocate_temp_name("tmp")
            return {name: TempBuffer(name=temp_name, shape=shape, dtype=dtype)}
        return {
            name: TempBuffer(
                name=allocate_temp_name(f"tmp{index}"),
                shape=shape,
                dtype=dtype,
            )
            for index, (name, shape, dtype) in enumerate(intermediates)
        }

    @staticmethod
    def _resolve_op(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
        temp_map: dict[str, str],
    ) -> (
        BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp
    ):
        if isinstance(op, BinaryOp):
            return BinaryOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, MultiInputBinaryOp):
            return MultiInputBinaryOp(
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, WhereOp):
            return WhereOp(
                condition=temp_map.get(op.condition, op.condition),
                input_x=temp_map.get(op.input_x, op.input_x),
                input_y=temp_map.get(op.input_y, op.input_y),
                output=temp_map.get(op.output, op.output),
                condition_shape=op.condition_shape,
                x_shape=op.x_shape,
                y_shape=op.y_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, UnaryOp):
            return UnaryOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                params=op.params,
            )
        if isinstance(op, ClipOp):
            return ClipOp(
                input0=temp_map.get(op.input0, op.input0),
                input_min=temp_map.get(op.input_min, op.input_min)
                if op.input_min is not None
                else None,
                input_max=temp_map.get(op.input_max, op.input_max)
                if op.input_max is not None
                else None,
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                min_shape=op.min_shape,
                max_shape=op.max_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, MatMulOp):
            return MatMulOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                dtype=op.dtype,
            )
        if isinstance(op, CastOp):
            return CastOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                input_dtype=op.input_dtype,
                dtype=op.dtype,
            )
        if isinstance(op, GemmOp):
            return GemmOp(
                input_a=temp_map.get(op.input_a, op.input_a),
                input_b=temp_map.get(op.input_b, op.input_b),
                input_c=(
                    temp_map.get(op.input_c, op.input_c)
                    if op.input_c is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                m=op.m,
                n=op.n,
                k=op.k,
                trans_a=op.trans_a,
                trans_b=op.trans_b,
                alpha=op.alpha,
                beta=op.beta,
                c_shape=op.c_shape,
                dtype=op.dtype,
            )
        if isinstance(op, AttentionOp):
            return AttentionOp(
                input_q=temp_map.get(op.input_q, op.input_q),
                input_k=temp_map.get(op.input_k, op.input_k),
                input_v=temp_map.get(op.input_v, op.input_v),
                input_attn_mask=(
                    temp_map.get(op.input_attn_mask, op.input_attn_mask)
                    if op.input_attn_mask is not None
                    else None
                ),
                input_past_key=(
                    temp_map.get(op.input_past_key, op.input_past_key)
                    if op.input_past_key is not None
                    else None
                ),
                input_past_value=(
                    temp_map.get(op.input_past_value, op.input_past_value)
                    if op.input_past_value is not None
                    else None
                ),
                input_nonpad_kv_seqlen=(
                    temp_map.get(
                        op.input_nonpad_kv_seqlen, op.input_nonpad_kv_seqlen
                    )
                    if op.input_nonpad_kv_seqlen is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                output_present_key=(
                    temp_map.get(op.output_present_key, op.output_present_key)
                    if op.output_present_key is not None
                    else None
                ),
                output_present_value=(
                    temp_map.get(op.output_present_value, op.output_present_value)
                    if op.output_present_value is not None
                    else None
                ),
                output_qk_matmul=(
                    temp_map.get(op.output_qk_matmul, op.output_qk_matmul)
                    if op.output_qk_matmul is not None
                    else None
                ),
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                scale=op.scale,
                is_causal=op.is_causal,
                softcap=op.softcap,
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                mask_shape=op.mask_shape,
                mask_is_bool=op.mask_is_bool,
                mask_rank=op.mask_rank,
                mask_broadcast_batch=op.mask_broadcast_batch,
                mask_broadcast_heads=op.mask_broadcast_heads,
                mask_broadcast_q_seq=op.mask_broadcast_q_seq,
                mask_q_seq=op.mask_q_seq,
                mask_kv_seq=op.mask_kv_seq,
                head_group_size=op.head_group_size,
                dtype=op.dtype,
            )
        if isinstance(op, LstmOp):
            return LstmOp(
                input_x=temp_map.get(op.input_x, op.input_x),
                input_w=temp_map.get(op.input_w, op.input_w),
                input_r=temp_map.get(op.input_r, op.input_r),
                input_b=(
                    temp_map.get(op.input_b, op.input_b)
                    if op.input_b is not None
                    else None
                ),
                input_sequence_lens=(
                    temp_map.get(op.input_sequence_lens, op.input_sequence_lens)
                    if op.input_sequence_lens is not None
                    else None
                ),
                input_initial_h=(
                    temp_map.get(op.input_initial_h, op.input_initial_h)
                    if op.input_initial_h is not None
                    else None
                ),
                input_initial_c=(
                    temp_map.get(op.input_initial_c, op.input_initial_c)
                    if op.input_initial_c is not None
                    else None
                ),
                input_p=(
                    temp_map.get(op.input_p, op.input_p)
                    if op.input_p is not None
                    else None
                ),
                output_y=(
                    temp_map.get(op.output_y, op.output_y)
                    if op.output_y is not None
                    else None
                ),
                output_y_h=(
                    temp_map.get(op.output_y_h, op.output_y_h)
                    if op.output_y_h is not None
                    else None
                ),
                output_y_c=(
                    temp_map.get(op.output_y_c, op.output_y_c)
                    if op.output_y_c is not None
                    else None
                ),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                direction=op.direction,
                layout=op.layout,
                input_forget=op.input_forget,
                clip=op.clip,
                activation_kinds=op.activation_kinds,
                activation_alphas=op.activation_alphas,
                activation_betas=op.activation_betas,
                dtype=op.dtype,
                sequence_lens_dtype=op.sequence_lens_dtype,
            )
        if isinstance(op, ConvOp):
            return ConvOp(
                input0=temp_map.get(op.input0, op.input0),
                weights=temp_map.get(op.weights, op.weights),
                bias=temp_map.get(op.bias, op.bias) if op.bias else None,
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                in_channels=op.in_channels,
                out_channels=op.out_channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                group=op.group,
                dtype=op.dtype,
            )
        if isinstance(op, AveragePoolOp):
            return AveragePoolOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                channels=op.channels,
                in_h=op.in_h,
                in_w=op.in_w,
                out_h=op.out_h,
                out_w=op.out_w,
                kernel_h=op.kernel_h,
                kernel_w=op.kernel_w,
                stride_h=op.stride_h,
                stride_w=op.stride_w,
                pad_top=op.pad_top,
                pad_left=op.pad_left,
                pad_bottom=op.pad_bottom,
                pad_right=op.pad_right,
                count_include_pad=op.count_include_pad,
                dtype=op.dtype,
            )
        if isinstance(op, BatchNormOp):
            return BatchNormOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                mean=temp_map.get(op.mean, op.mean),
                variance=temp_map.get(op.variance, op.variance),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LpNormalizationOp):
            return LpNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                axis=op.axis,
                p=op.p,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                dtype=op.dtype,
            )
        if isinstance(op, InstanceNormalizationOp):
            return InstanceNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, GroupNormalizationOp):
            return GroupNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LayerNormalizationOp):
            return LayerNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=(temp_map.get(op.bias, op.bias) if op.bias else None),
                output=temp_map.get(op.output, op.output),
                mean_output=(
                    temp_map.get(op.mean_output, op.mean_output)
                    if op.mean_output
                    else None
                ),
                invstd_output=(
                    temp_map.get(op.invstd_output, op.invstd_output)
                    if op.invstd_output
                    else None
                ),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                bias_shape=op.bias_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, MeanVarianceNormalizationOp):
            return MeanVarianceNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, RMSNormalizationOp):
            return RMSNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LrnOp):
            return LrnOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                size=op.size,
                half=op.half,
                alpha=op.alpha,
                beta=op.beta,
                bias=op.bias,
                dtype=op.dtype,
            )
        if isinstance(op, SoftmaxOp):
            return SoftmaxOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, LogSoftmaxOp):
            return LogSoftmaxOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return NegativeLogLikelihoodLossOp(
                input0=temp_map.get(op.input0, op.input0),
                target=temp_map.get(op.target, op.target),
                weight=(
                    temp_map.get(op.weight, op.weight)
                    if op.weight is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return SoftmaxCrossEntropyLossOp(
                input0=temp_map.get(op.input0, op.input0),
                target=temp_map.get(op.target, op.target),
                weight=(
                    temp_map.get(op.weight, op.weight)
                    if op.weight is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                log_prob=(
                    temp_map.get(op.log_prob, op.log_prob)
                    if op.log_prob is not None
                    else None
                ),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                log_prob_shape=op.log_prob_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, MaxPoolOp):
            return MaxPoolOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                indices=(
                    temp_map.get(op.indices, op.indices)
                    if op.indices is not None
                    else None
                ),
                batch=op.batch,
                channels=op.channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                ceil_mode=op.ceil_mode,
                storage_order=op.storage_order,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherElementsOp):
            return GatherElementsOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherOp):
            return GatherOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, ConcatOp):
            return ConcatOp(
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, ConstantOfShapeOp):
            return ConstantOfShapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                shape=op.shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ShapeOp):
            return ShapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                values=op.values,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SizeOp):
            return SizeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ExpandOp):
            return ExpandOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                input_shape_padded=op.input_shape_padded,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, RangeOp):
            return RangeOp(
                start=temp_map.get(op.start, op.start),
                limit=temp_map.get(op.limit, op.limit),
                delta=temp_map.get(op.delta, op.delta),
                output=temp_map.get(op.output, op.output),
                output_shape=op.output_shape,
                length=op.length,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SplitOp):
            return SplitOp(
                input0=temp_map.get(op.input0, op.input0),
                outputs=tuple(
                    temp_map.get(name, name) for name in op.outputs
                ),
                input_shape=op.input_shape,
                output_shapes=op.output_shapes,
                axis=op.axis,
                split_sizes=op.split_sizes,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TransposeOp):
            return TransposeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                perm=op.perm,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ReshapeOp):
            return ReshapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, IdentityOp):
            return IdentityOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, EyeLikeOp):
            return EyeLikeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                output_shape=op.output_shape,
                k=op.k,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TileOp):
            return TileOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                repeats=op.repeats,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, DepthToSpaceOp):
            return DepthToSpaceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                mode=op.mode,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SpaceToDepthOp):
            return SpaceToDepthOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SliceOp):
            return SliceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                starts=op.starts,
                steps=op.steps,
                axes=op.axes,
                starts_input=temp_map.get(op.starts_input, op.starts_input)
                if op.starts_input
                else None,
                ends_input=temp_map.get(op.ends_input, op.ends_input)
                if op.ends_input
                else None,
                axes_input=temp_map.get(op.axes_input, op.axes_input)
                if op.axes_input
                else None,
                steps_input=temp_map.get(op.steps_input, op.steps_input)
                if op.steps_input
                else None,
                starts_shape=op.starts_shape,
                ends_shape=op.ends_shape,
                axes_shape=op.axes_shape,
                steps_shape=op.steps_shape,
                starts_dtype=op.starts_dtype,
                ends_dtype=op.ends_dtype,
                axes_dtype=op.axes_dtype,
                steps_dtype=op.steps_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ResizeOp):
            return ResizeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                scales=op.scales,
                scales_input=temp_map.get(op.scales_input, op.scales_input)
                if op.scales_input
                else None,
                sizes_input=temp_map.get(op.sizes_input, op.sizes_input)
                if op.sizes_input
                else None,
                roi_input=temp_map.get(op.roi_input, op.roi_input)
                if op.roi_input
                else None,
                axes=op.axes,
                scales_shape=op.scales_shape,
                sizes_shape=op.sizes_shape,
                roi_shape=op.roi_shape,
                scales_dtype=op.scales_dtype,
                sizes_dtype=op.sizes_dtype,
                roi_dtype=op.roi_dtype,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                mode=op.mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                nearest_mode=op.nearest_mode,
                cubic_coeff_a=op.cubic_coeff_a,
                exclude_outside=op.exclude_outside,
                extrapolation_value=op.extrapolation_value,
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
                dtype=op.dtype,
            )
        if isinstance(op, ReduceOp):
            return ReduceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axes=op.axes,
                axes_input=temp_map.get(op.axes_input, op.axes_input)
                if op.axes_input
                else None,
                axes_input_shape=op.axes_input_shape,
                axes_input_dtype=op.axes_input_dtype,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                reduce_kind=op.reduce_kind,
                reduce_count=op.reduce_count,
                dtype=op.dtype,
            )
        if isinstance(op, ArgReduceOp):
            return ArgReduceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                keepdims=op.keepdims,
                select_last_index=op.select_last_index,
                reduce_kind=op.reduce_kind,
                input_dtype=op.input_dtype,
                output_dtype=op.output_dtype,
            )
        return UnaryOp(
            input0=temp_map.get(op.input0, op.input0),
            output=temp_map.get(op.output, op.output),
            function=op.function,
            shape=op.shape,
            dtype=op.dtype,
        )

    def _render_op(
        self,
        model: LoweredModel,
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
        index: int,
        *,
        array_suffix: str,
        loop_vars: tuple[str, ...],
        c_type: str,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
        binary_template,
        multi_input_template,
        where_template,
        unary_template,
        clip_template,
        cast_template,
        matmul_template,
        gemm_template,
        attention_template,
        conv_template,
        avg_pool_template,
        batch_norm_template,
        lp_norm_template,
        instance_norm_template,
        group_norm_template,
        layer_norm_template,
        mean_variance_norm_template,
        rms_norm_template,
        lrn_template,
        lstm_template,
        softmax_template,
        logsoftmax_template,
        nllloss_template,
        softmax_cross_entropy_loss_template,
        maxpool_template,
        concat_template,
        gather_elements_template,
        gather_template,
        transpose_template,
        reshape_template,
        identity_template,
        eye_like_template,
        tile_template,
        depth_to_space_template,
        space_to_depth_template,
        slice_template,
        slice_dynamic_template,
        resize_template,
        reduce_template,
        reduce_dynamic_template,
        arg_reduce_template,
        constant_of_shape_template,
        shape_template,
        size_template,
        expand_template,
        range_template,
        split_template,
        scalar_registry: ScalarFunctionRegistry | None = None,
        dim_args: str = "",
        tensor_dim_names: Mapping[str, Mapping[int, str]] | None = None,
    ) -> str:
        node_info = model.node_infos[index]
        node_comment = CEmitter._emit_node_comment(node_info, index)
        tensor_dim_names = tensor_dim_names or {}

        def _dim_names_for(name: str) -> Mapping[int, str]:
            return tensor_dim_names.get(name, {})

        def with_node_comment(rendered: str) -> str:
            return f"{node_comment}\n{_format_c_indentation(rendered)}"

        if isinstance(op, BinaryOp):
            scalar_operator = None
            if (
                scalar_registry is not None
                and op.function not in COMPARE_FUNCTIONS
            ):
                scalar_operator = self._scalar_function_name(
                    op.function, op.input_dtype, scalar_registry
                )
            op_spec = binary_op_symbol(
                op.function,
                dtype=op.input_dtype,
                validate_attrs=False,
            )
            if op_spec is None:
                raise CodegenError(
                    f"Unsupported binary operator for rendering: {op.function.value}"
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            input_c_type = op.input_dtype.c_type
            output_c_type = op.dtype.c_type
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": input_c_type,
                "output_c_type": output_c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
            }
            left_expr = f"{op.input0}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            right_expr = f"{op.input1}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            operator_expr = None
            operator = op_spec.operator
            operator_kind = op.operator_kind
            if scalar_operator is not None:
                operator = scalar_operator
                operator_kind = OperatorKind.FUNC
            if operator_kind == OperatorKind.EXPR:
                operator_expr = op_spec.operator.format(
                    left=left_expr, right=right_expr
                )
            rendered = binary_template.render(
                **common,
                input0=op.input0,
                input1=op.input1,
                output=op.output,
                operator=operator,
                operator_kind=operator_kind.value,
                left_expr=left_expr,
                right_expr=right_expr,
                operator_expr=operator_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MultiInputBinaryOp):
            scalar_operator = None
            if (
                scalar_registry is not None
                and op.function not in COMPARE_FUNCTIONS
                and op.function != ScalarFunction.MEAN
            ):
                scalar_operator = self._scalar_function_name(
                    op.function, op.input_dtype, scalar_registry
                )
            op_spec = binary_op_symbol(
                op.function,
                dtype=op.input_dtype,
                validate_attrs=False,
            )
            if op_spec is None:
                raise CodegenError(
                    "Unsupported multi-input operator for rendering: "
                    f"{op.function.value}"
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            input_c_type = op.input_dtype.c_type
            output_c_type = op.dtype.c_type
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": input_c_type,
                "output_c_type": output_c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
            }
            input_exprs = [
                f"{name}" + "".join(f"[{var}]" for var in loop_vars)
                for name in op.inputs
            ]
            output_expr = f"{op.output}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            operator = op_spec.operator
            operator_kind = op.operator_kind
            operator_expr = None
            mean_scale = None
            if op.function == ScalarFunction.MEAN:
                operator = "+"
                operator_kind = OperatorKind.INFIX
                mean_scale = len(op.inputs)
            if scalar_operator is not None:
                operator = scalar_operator
                operator_kind = OperatorKind.FUNC
            if operator_kind == OperatorKind.EXPR:
                raise CodegenError(
                    "Multi-input operators do not support expression operators."
                )
            rendered = multi_input_template.render(
                **common,
                inputs=op.inputs,
                input_exprs=input_exprs,
                output=op.output,
                output_expr=output_expr,
                operator=operator,
                operator_kind=operator_kind.value,
                operator_expr=operator_expr,
                mean_scale=mean_scale,
                is_mean=op.function == ScalarFunction.MEAN,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, WhereOp):
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            output_array_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            condition_array_suffix = self._param_array_suffix(
                op.condition_shape, _dim_names_for(op.condition)
            )
            x_array_suffix = self._param_array_suffix(
                op.x_shape, _dim_names_for(op.input_x)
            )
            y_array_suffix = self._param_array_suffix(
                op.y_shape, _dim_names_for(op.input_y)
            )
            condition_expr = CEmitter._broadcast_index_expr(
                op.condition, op.condition_shape, op.output_shape, loop_vars
            )
            x_expr = CEmitter._broadcast_index_expr(
                op.input_x, op.x_shape, op.output_shape, loop_vars
            )
            y_expr = CEmitter._broadcast_index_expr(
                op.input_y, op.y_shape, op.output_shape, loop_vars
            )
            output_expr = f"{op.output}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            rendered = where_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                output_shape=output_shape,
                loop_vars=loop_vars,
                condition=op.condition,
                input_x=op.input_x,
                input_y=op.input_y,
                output=op.output,
                condition_array_suffix=condition_array_suffix,
                x_array_suffix=x_array_suffix,
                y_array_suffix=y_array_suffix,
                output_array_suffix=output_array_suffix,
                condition_expr=condition_expr,
                x_expr=x_expr,
                y_expr=y_expr,
                output_expr=output_expr,
                input_c_type=op.dtype.c_type,
                output_c_type=op.dtype.c_type,
                condition_c_type=ScalarType.BOOL.c_type,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MatMulOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            output_index_expr = f"{op.output}" + "".join(
                f"[{var}]" for var in output_loop_vars
            )
            batch_rank = len(op.batch_shape)
            batch_vars = output_loop_vars[:batch_rank]
            if op.left_vector and op.right_vector:
                row_var = None
                col_var = None
            elif op.left_vector:
                row_var = None
                col_var = output_loop_vars[-1]
            elif op.right_vector:
                row_var = output_loop_vars[-1]
                col_var = None
            else:
                row_var = output_loop_vars[-2]
                col_var = output_loop_vars[-1]
            input0_index_expr, input1_index_expr = CEmitter._matmul_index_exprs(
                op,
                batch_vars,
                row_var,
                col_var,
                batch_rank,
            )
            rendered = matmul_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                input1=op.input1,
                output=op.output,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                input0_suffix=self._param_array_suffix(op.input0_shape),
                input1_suffix=self._param_array_suffix(op.input1_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                m=op.m,
                n=op.n,
                k=op.k,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GemmOp):
            input_a_shape = (op.k, op.m) if op.trans_a else (op.m, op.k)
            input_b_shape = (op.n, op.k) if op.trans_b else (op.k, op.n)
            alpha_literal = CEmitter._format_literal(op.dtype, op.alpha)
            beta_literal = CEmitter._format_literal(op.dtype, op.beta)
            if op.c_shape is None:
                c_rank = 0
                c_dim0 = 0
                c_dim1 = 0
            elif len(op.c_shape) == 0:
                c_rank = 0
                c_dim0 = 0
                c_dim1 = 0
            elif len(op.c_shape) == 1:
                c_rank = 1
                c_dim0 = 1
                c_dim1 = op.c_shape[0]
            else:
                c_rank = 2
                c_dim0 = op.c_shape[0]
                c_dim1 = op.c_shape[1]
            rendered = gemm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_a=op.input_a,
                input_b=op.input_b,
                input_c=op.input_c,
                output=op.output,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                alpha_literal=alpha_literal,
                beta_literal=beta_literal,
                trans_a=int(op.trans_a),
                trans_b=int(op.trans_b),
                m=op.m,
                n=op.n,
                k=op.k,
                input_a_suffix=self._param_array_suffix(input_a_shape),
                input_b_suffix=self._param_array_suffix(input_b_shape),
                output_suffix=self._param_array_suffix((op.m, op.n)),
                c_suffix=(
                    self._param_array_suffix(op.c_shape)
                    if op.c_shape is not None
                    else None
                ),
                c_rank=c_rank,
                c_dim0=c_dim0,
                c_dim1=c_dim1,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AttentionOp):
            if op.q_rank == 4:
                input_q_shape = (op.batch, op.q_heads, op.q_seq, op.qk_head_size)
            else:
                input_q_shape = (op.batch, op.q_seq, op.q_hidden_size)
            if op.k_rank == 4:
                input_k_shape = (op.batch, op.kv_heads, op.kv_seq, op.qk_head_size)
            else:
                input_k_shape = (op.batch, op.kv_seq, op.k_hidden_size)
            if op.v_rank == 4:
                input_v_shape = (op.batch, op.kv_heads, op.kv_seq, op.v_head_size)
            else:
                input_v_shape = (op.batch, op.kv_seq, op.v_hidden_size)
            if op.output_rank == 4:
                output_shape = (op.batch, op.q_heads, op.q_seq, op.v_head_size)
            else:
                output_shape = (
                    op.batch,
                    op.q_seq,
                    op.q_heads * op.v_head_size,
                )
            present_key_shape = (
                (op.batch, op.kv_heads, op.total_seq, op.qk_head_size)
                if op.output_present_key is not None
                else None
            )
            present_value_shape = (
                (op.batch, op.kv_heads, op.total_seq, op.v_head_size)
                if op.output_present_value is not None
                else None
            )
            qk_matmul_shape = (
                (op.batch, op.q_heads, op.q_seq, op.total_seq)
                if op.output_qk_matmul is not None
                else None
            )
            rendered = attention_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_q=op.input_q,
                input_k=op.input_k,
                input_v=op.input_v,
                input_attn_mask=op.input_attn_mask,
                input_past_key=op.input_past_key,
                input_past_value=op.input_past_value,
                input_nonpad_kv_seqlen=op.input_nonpad_kv_seqlen,
                output=op.output,
                output_present_key=op.output_present_key,
                output_present_value=op.output_present_value,
                output_qk_matmul=op.output_qk_matmul,
                c_type=c_type,
                nonpad_c_type=ScalarType.I64.c_type,
                zero_literal=zero_literal,
                min_literal=min_literal,
                scale_literal=CEmitter._format_floating(op.scale, op.dtype),
                softcap_literal=CEmitter._format_floating(op.softcap, op.dtype),
                one_literal=CEmitter._format_literal(op.dtype, 1),
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                tanh_fn=CEmitter._math_fn(op.dtype, "tanhf", "tanh"),
                is_causal=int(op.is_causal),
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                head_group_size=op.head_group_size,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                has_attn_mask=int(op.input_attn_mask is not None),
                mask_rank=op.mask_rank or 0,
                mask_is_bool=int(op.mask_is_bool),
                mask_broadcast_batch=int(op.mask_broadcast_batch),
                mask_broadcast_heads=int(op.mask_broadcast_heads),
                mask_broadcast_q_seq=int(op.mask_broadcast_q_seq),
                mask_q_seq=op.mask_q_seq or 0,
                mask_kv_seq=op.mask_kv_seq or 0,
                input_q_suffix=self._param_array_suffix(input_q_shape),
                input_k_suffix=self._param_array_suffix(input_k_shape),
                input_v_suffix=self._param_array_suffix(input_v_shape),
                input_mask_suffix=(
                    self._param_array_suffix(op.mask_shape)
                    if op.input_attn_mask is not None
                    else ""
                ),
                input_past_key_suffix=(
                    self._param_array_suffix(
                        (op.batch, op.kv_heads, op.past_seq, op.qk_head_size)
                    )
                    if op.input_past_key is not None
                    else ""
                ),
                input_past_value_suffix=(
                    self._param_array_suffix(
                        (op.batch, op.kv_heads, op.past_seq, op.v_head_size)
                    )
                    if op.input_past_value is not None
                    else ""
                ),
                input_nonpad_suffix=(
                    self._param_array_suffix((op.batch,))
                    if op.input_nonpad_kv_seqlen is not None
                    else ""
                ),
                output_suffix=self._param_array_suffix(output_shape),
                output_present_key_suffix=(
                    self._param_array_suffix(present_key_shape)
                    if present_key_shape is not None
                    else ""
                ),
                output_present_value_suffix=(
                    self._param_array_suffix(present_value_shape)
                    if present_value_shape is not None
                    else ""
                ),
                output_qk_matmul_suffix=(
                    self._param_array_suffix(qk_matmul_shape)
                    if qk_matmul_shape is not None
                    else ""
                ),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConvOp):
            input_shape = (op.batch, op.in_channels, *op.in_spatial)
            weight_shape = (
                op.out_channels,
                op.in_channels // op.group,
                *op.kernel_shape,
            )
            output_shape = (op.batch, op.out_channels, *op.out_spatial)
            out_indices = tuple(f"od{dim}" for dim in range(op.spatial_rank))
            kernel_indices = tuple(
                f"kd{dim}" for dim in range(op.spatial_rank)
            )
            in_indices = tuple(f"id{dim}" for dim in range(op.spatial_rank))
            pad_begin = op.pads[: op.spatial_rank]
            group_in_channels = op.in_channels // op.group
            group_out_channels = op.out_channels // op.group
            rendered = conv_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                weights=op.weights,
                bias=op.bias,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(input_shape),
                weight_suffix=self._param_array_suffix(weight_shape),
                bias_suffix=self._param_array_suffix((op.out_channels,)),
                output_suffix=self._param_array_suffix(output_shape),
                batch=op.batch,
                in_channels=op.in_channels,
                out_channels=op.out_channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads_begin=pad_begin,
                dilations=op.dilations,
                group=op.group,
                group_in_channels=group_in_channels,
                group_out_channels=group_out_channels,
                out_indices=out_indices,
                kernel_indices=kernel_indices,
                in_indices=in_indices,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AveragePoolOp):
            input_shape = (op.batch, op.channels, op.in_h, op.in_w)
            output_shape = (op.batch, op.channels, op.out_h, op.out_w)
            rendered = avg_pool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(input_shape),
                output_suffix=self._param_array_suffix(output_shape),
                batch=op.batch,
                channels=op.channels,
                in_h=op.in_h,
                in_w=op.in_w,
                out_h=op.out_h,
                out_w=op.out_w,
                kernel_h=op.kernel_h,
                kernel_w=op.kernel_w,
                stride_h=op.stride_h,
                stride_w=op.stride_w,
                pad_top=op.pad_top,
                pad_left=op.pad_left,
                pad_bottom=op.pad_bottom,
                pad_right=op.pad_right,
                count_include_pad=int(op.count_include_pad),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, BatchNormOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = batch_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                mean=op.mean,
                variance=op.variance,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix((op.channels,)),
                bias_suffix=self._param_array_suffix((op.channels,)),
                mean_suffix=self._param_array_suffix((op.channels,)),
                variance_suffix=self._param_array_suffix((op.channels,)),
                shape=shape,
                loop_vars=loop_vars,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LpNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            rendered = lp_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=self._param_array_suffix(shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                p=op.p,
                zero_literal=zero_literal,
                abs_fn=CEmitter._math_fn(op.dtype, "fabsf", "fabs"),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, InstanceNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = instance_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix((op.channels,)),
                bias_suffix=self._param_array_suffix((op.channels,)),
                shape=shape,
                loop_vars=loop_vars,
                spatial_size=op.spatial_size,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GroupNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = group_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix((op.channels,)),
                bias_suffix=self._param_array_suffix((op.channels,)),
                shape=shape,
                loop_vars=loop_vars,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LayerNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            prefix_loop_vars = loop_vars[: op.axis]
            norm_loop_vars = loop_vars[op.axis :]
            scale_index_vars = [
                "0" if dim == 1 else var
                for dim, var in zip(op.scale_shape, norm_loop_vars)
            ]
            bias_index_vars = None
            if op.bias_shape is not None and op.bias is not None:
                bias_index_vars = [
                    "0" if dim == 1 else var
                    for dim, var in zip(op.bias_shape, norm_loop_vars)
                ]
            mean_index_vars = [
                *prefix_loop_vars,
                *("0" for _ in norm_loop_vars),
            ]
            rendered = layer_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                output=op.output,
                mean_output=op.mean_output,
                invstd_output=op.invstd_output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix(op.scale_shape),
                bias_suffix=(
                    self._param_array_suffix(op.bias_shape)
                    if op.bias_shape is not None
                    else ""
                ),
                mean_suffix=(
                    self._param_array_suffix(
                        op.shape[: op.axis]
                        + (1,) * len(op.normalized_shape)
                    )
                    if op.mean_output is not None
                    else ""
                ),
                invstd_suffix=(
                    self._param_array_suffix(
                        op.shape[: op.axis]
                        + (1,) * len(op.normalized_shape)
                    )
                    if op.invstd_output is not None
                    else ""
                ),
                prefix_shape=shape[: op.axis],
                norm_shape=shape[op.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                bias_index_vars=bias_index_vars,
                mean_index_vars=mean_index_vars,
                inner=op.inner,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MeanVarianceNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = mean_variance_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                shape=shape,
                loop_vars=loop_vars,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, RMSNormalizationOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            prefix_loop_vars = loop_vars[: op.axis]
            norm_loop_vars = loop_vars[op.axis :]
            scale_index_vars = [
                "0" if dim == 1 else var
                for dim, var in zip(op.scale_shape, norm_loop_vars)
            ]
            rendered = rms_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix(op.scale_shape),
                prefix_shape=shape[: op.axis],
                norm_shape=shape[op.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                inner=op.inner,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LrnOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = lrn_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                shape=shape,
                channels=op.channels,
                half=op.half,
                loop_vars=loop_vars,
                zero_literal=zero_literal,
                alpha_div_size_literal=CEmitter._format_floating(
                    op.alpha / op.size, op.dtype
                ),
                beta_literal=CEmitter._format_floating(op.beta, op.dtype),
                bias_literal=CEmitter._format_floating(op.bias, op.dtype),
                pow_fn=CEmitter._math_fn(op.dtype, "powf", "pow"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LstmOp):
            input_x_shape = (
                (op.seq_length, op.batch_size, op.input_size)
                if op.layout == 0
                else (op.batch_size, op.seq_length, op.input_size)
            )
            w_shape = (op.num_directions, 4 * op.hidden_size, op.input_size)
            r_shape = (op.num_directions, 4 * op.hidden_size, op.hidden_size)
            b_shape = (
                (op.num_directions, 8 * op.hidden_size)
                if op.input_b is not None
                else None
            )
            seq_shape = (op.batch_size,) if op.input_sequence_lens is not None else None
            h_shape = (
                (op.num_directions, op.batch_size, op.hidden_size)
                if op.input_initial_h is not None
                else None
            )
            c_shape = (
                (op.num_directions, op.batch_size, op.hidden_size)
                if op.input_initial_c is not None
                else None
            )
            p_shape = (
                (op.num_directions, 3 * op.hidden_size)
                if op.input_p is not None
                else None
            )
            y_shape = (
                (op.seq_length, op.num_directions, op.batch_size, op.hidden_size)
                if op.layout == 0
                else (op.batch_size, op.seq_length, op.num_directions, op.hidden_size)
            )
            rendered = lstm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_x=op.input_x,
                input_w=op.input_w,
                input_r=op.input_r,
                input_b=op.input_b,
                input_sequence_lens=op.input_sequence_lens,
                input_initial_h=op.input_initial_h,
                input_initial_c=op.input_initial_c,
                input_p=op.input_p,
                output_y=op.output_y,
                output_y_h=op.output_y_h,
                output_y_c=op.output_y_c,
                c_type=c_type,
                seq_c_type=(op.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                clip_literal=(
                    CEmitter._format_floating(op.clip, op.dtype)
                    if op.clip is not None
                    else CEmitter._format_literal(op.dtype, 0)
                ),
                use_clip=int(op.clip is not None and op.clip > 0),
                input_suffix=self._param_array_suffix(input_x_shape),
                w_suffix=self._param_array_suffix(w_shape),
                r_suffix=self._param_array_suffix(r_shape),
                b_suffix=self._param_array_suffix(b_shape) if b_shape else None,
                seq_suffix=self._param_array_suffix(seq_shape) if seq_shape else None,
                h_suffix=self._param_array_suffix(h_shape) if h_shape else None,
                c_suffix=self._param_array_suffix(c_shape) if c_shape else None,
                p_suffix=self._param_array_suffix(p_shape) if p_shape else None,
                y_suffix=self._param_array_suffix(y_shape) if op.output_y else None,
                y_h_suffix=(
                    self._param_array_suffix((op.num_directions, op.batch_size, op.hidden_size))
                    if op.output_y_h
                    else None
                ),
                y_c_suffix=(
                    self._param_array_suffix((op.num_directions, op.batch_size, op.hidden_size))
                    if op.output_y_c
                    else None
                ),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                layout=op.layout,
                direction=op.direction,
                input_forget=op.input_forget,
                activation_kinds=op.activation_kinds,
                activation_alphas=tuple(
                    CEmitter._format_floating(value, op.dtype)
                    for value in op.activation_alphas
                ),
                activation_betas=tuple(
                    CEmitter._format_floating(value, op.dtype)
                    for value in op.activation_betas
                ),
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                tanh_fn=CEmitter._math_fn(op.dtype, "tanhf", "tanh"),
                log1p_fn=CEmitter._math_fn(op.dtype, "log1pf", "log1p"),
                fabs_fn=CEmitter._math_fn(op.dtype, "fabsf", "fabs"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxOp):
            rendered = softmax_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=self._param_array_suffix(op.shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LogSoftmaxOp):
            rendered = logsoftmax_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=self._param_array_suffix(op.shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                log_fn=CEmitter._math_fn(op.dtype, "logf", "log"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            rendered = nllloss_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                target=op.target,
                weight=op.weight,
                output=op.output,
                c_type=c_type,
                target_c_type=op.target_dtype.c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                target_suffix=self._param_array_suffix(op.target_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            use_ignore_index = int(op.ignore_index is not None)
            ignore_index = op.ignore_index if op.ignore_index is not None else -1
            rendered = softmax_cross_entropy_loss_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                target=op.target,
                weight=op.weight,
                output=op.output,
                log_prob=op.log_prob,
                c_type=c_type,
                target_c_type=op.target_dtype.c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                target_suffix=self._param_array_suffix(op.target_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                log_prob_suffix=(
                    self._param_array_suffix(op.log_prob_shape)
                    if op.log_prob_shape is not None
                    else None
                ),
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                use_ignore_index=use_ignore_index,
                ignore_index=ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                log_fn=CEmitter._math_fn(op.dtype, "logf", "log"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MaxPoolOp):
            input_shape = (op.batch, op.channels, *op.in_spatial)
            output_shape = (op.batch, op.channels, *op.out_spatial)
            indices_c_type = (
                op.indices_dtype.c_type
                if op.indices is not None and op.indices_dtype is not None
                else None
            )
            rendered = maxpool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                indices=op.indices,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=self._param_array_suffix(input_shape),
                output_suffix=self._param_array_suffix(output_shape),
                indices_suffix=self._param_array_suffix(output_shape),
                indices_c_type=indices_c_type,
                batch=op.batch,
                channels=op.channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                ceil_mode=int(op.ceil_mode),
                storage_order=op.storage_order,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConcatOp):
            axis = op.axis
            if axis < 0:
                axis += len(op.output_shape)
            outer = CEmitter._element_count(op.output_shape[:axis] or (1,))
            inner = CEmitter._element_count(op.output_shape[axis + 1 :] or (1,))
            axis_sizes = tuple(shape[axis] for shape in op.input_shapes)
            rendered = concat_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                inputs=op.inputs,
                output=op.output,
                c_type=c_type,
                input_suffixes=tuple(
                    self._param_array_suffix(shape) for shape in op.input_shapes
                ),
                output_suffix=self._param_array_suffix(op.output_shape),
                axis_sizes=axis_sizes,
                input_count=len(op.inputs),
                outer=outer,
                inner=inner,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherElementsOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            data_indices = list(loop_vars)
            data_indices[op.axis] = "gather_index"
            rendered = gather_elements_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                data=op.data,
                indices=op.indices,
                output=op.output,
                c_type=c_type,
                indices_c_type=op.indices_dtype.c_type,
                data_suffix=self._param_array_suffix(op.data_shape),
                indices_suffix=self._param_array_suffix(op.indices_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                loop_vars=loop_vars,
                data_indices=data_indices,
                axis_dim=op.data_shape[op.axis],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            output_loop_vars = loop_vars if op.output_shape else ()
            indices_rank = len(op.indices_shape)
            if indices_rank == 0:
                indices_indices = ("0",)
            else:
                indices_indices = output_loop_vars[
                    op.axis : op.axis + indices_rank
                ]
            data_indices = [
                *output_loop_vars[: op.axis],
                "gather_index",
                *output_loop_vars[op.axis + indices_rank :],
            ]
            rendered = gather_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                data=op.data,
                indices=op.indices,
                output=op.output,
                c_type=c_type,
                indices_c_type=op.indices_dtype.c_type,
                data_suffix=self._param_array_suffix(op.data_shape),
                indices_suffix=self._param_array_suffix(op.indices_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                loop_vars=loop_vars,
                indices_indices=indices_indices,
                data_indices=data_indices,
                axis_dim=op.data_shape[op.axis],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TransposeOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            output_suffix = self._param_array_suffix(output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            if not op.input_shape:
                input_indices = [loop_vars[0]]
            else:
                input_indices = [None] * len(op.perm)
                for output_axis, input_axis in enumerate(op.perm):
                    input_indices[input_axis] = loop_vars[output_axis]
            rendered = transpose_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_indices=input_indices,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReshapeOp):
            rendered = reshape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                element_count=CEmitter._element_count(op.output_shape),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, IdentityOp):
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            output_suffix = self._param_array_suffix(shape, output_dim_names)
            input_suffix = self._param_array_suffix(shape, _dim_names_for(op.input0))
            rendered = identity_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                loop_vars=loop_vars,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, EyeLikeOp):
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.output_shape, output_dim_names)
            output_suffix = self._param_array_suffix(op.output_shape, output_dim_names)
            input_suffix = self._param_array_suffix(
                op.output_shape, _dim_names_for(op.input0)
            )
            batch_dims = op.output_shape[:-2]
            batch_size = CEmitter._element_count(batch_dims or (1,))
            rendered = eye_like_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                batch_size=batch_size,
                rows=op.output_shape[-2],
                cols=op.output_shape[-1],
                k=op.k,
                zero_literal=zero_literal,
                one_literal=f"(({c_type})1)",
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TileOp):
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            input_index_terms = [
                f"({var} % {dim}) * {stride}"
                for var, dim, stride in zip(
                    loop_vars, op.input_shape, op.input_strides
                )
            ]
            input_index_expr = " + ".join(input_index_terms) or "0"
            rendered = tile_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_index_expr=input_index_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, DepthToSpaceOp):
            output_suffix = self._param_array_suffix(op.output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            rendered = depth_to_space_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=op.input_shape[0],
                in_channels=op.input_shape[1],
                out_channels=op.output_shape[1],
                in_h=op.input_shape[2],
                in_w=op.input_shape[3],
                out_h=op.output_shape[2],
                out_w=op.output_shape[3],
                blocksize=op.blocksize,
                mode=op.mode,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SpaceToDepthOp):
            output_suffix = self._param_array_suffix(op.output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            rendered = space_to_depth_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=op.input_shape[0],
                in_channels=op.input_shape[1],
                out_channels=op.output_shape[1],
                in_h=op.input_shape[2],
                in_w=op.input_shape[3],
                out_h=op.output_shape[2],
                out_w=op.output_shape[3],
                blocksize=op.blocksize,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SliceOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            if op.starts is not None and op.steps is not None:
                input_indices: list[str] = []
                for start, step, loop_var in zip(op.starts, op.steps, loop_vars):
                    if step == 1:
                        if start == 0:
                            input_indices.append(loop_var)
                        else:
                            input_indices.append(f"{start} + {loop_var}")
                    else:
                        input_indices.append(f"{start} + {step} * {loop_var}")
                rendered = slice_template.render(
                    model_name=model.name,
                    op_name=f"{model.name}_op{index}",
                    input0=op.input0,
                    output=op.output,
                    c_type=c_type,
                    input_suffix=self._param_array_suffix(op.input_shape),
                    output_suffix=self._param_array_suffix(op.output_shape),
                    output_shape=output_shape,
                    loop_vars=loop_vars,
                    input_indices=input_indices,
                ).rstrip()
                return with_node_comment(rendered)
            params = [
                f"const {c_type} {op.input0}"
                f"{self._param_array_suffix(op.input_shape)}"
            ]
            if op.starts_input and op.starts_shape and op.starts_dtype:
                starts_suffix = self._param_array_suffix(op.starts_shape)
                params.append(
                    f"const {op.starts_dtype.c_type} "
                    f"{op.starts_input}{starts_suffix}"
                )
            if op.ends_input and op.ends_shape and op.ends_dtype:
                ends_suffix = self._param_array_suffix(op.ends_shape)
                params.append(
                    f"const {op.ends_dtype.c_type} {op.ends_input}{ends_suffix}"
                )
            if op.axes_input and op.axes_shape and op.axes_dtype:
                axes_suffix = self._param_array_suffix(op.axes_shape)
                params.append(
                    f"const {op.axes_dtype.c_type} {op.axes_input}{axes_suffix}"
                )
            if op.steps_input and op.steps_shape and op.steps_dtype:
                steps_suffix = self._param_array_suffix(op.steps_shape)
                params.append(
                    f"const {op.steps_dtype.c_type} {op.steps_input}{steps_suffix}"
                )
            params.append(
                f"{c_type} {op.output}"
                f"{self._param_array_suffix(op.output_shape)}"
            )
            input_dims = CEmitter._codegen_shape(op.input_shape)
            rendered = slice_dynamic_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                params=params,
                input0=op.input0,
                starts_input=op.starts_input,
                ends_input=op.ends_input,
                axes_input=op.axes_input,
                steps_input=op.steps_input,
                output=op.output,
                c_type=c_type,
                input_shape=input_dims,
                output_shape=output_shape,
                output_loop_vars=loop_vars,
                input_rank=len(op.input_shape),
                starts_len=op.starts_shape[0] if op.starts_shape else 0,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ResizeOp):
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            params = [f"const {c_type} {op.input0}{input_suffix}"]
            roi_suffix = None
            scales_suffix = None
            sizes_suffix = None
            roi_c_type = None
            scales_c_type = None
            sizes_c_type = None
            if op.roi_input and op.roi_shape and op.roi_dtype:
                roi_suffix = self._param_array_suffix(op.roi_shape)
                roi_c_type = op.roi_dtype.c_type
                params.append(
                    f"const {roi_c_type} {op.roi_input}{roi_suffix}"
                )
            if op.scales_input and op.scales_shape and op.scales_dtype:
                scales_suffix = self._param_array_suffix(op.scales_shape)
                scales_c_type = op.scales_dtype.c_type
                params.append(
                    f"const {scales_c_type} {op.scales_input}{scales_suffix}"
                )
            if op.sizes_input and op.sizes_shape and op.sizes_dtype:
                sizes_suffix = self._param_array_suffix(op.sizes_shape)
                sizes_c_type = op.sizes_dtype.c_type
                params.append(
                    f"const {sizes_c_type} {op.sizes_input}{sizes_suffix}"
                )
            params.append(f"{c_type} {op.output}{output_suffix}")
            scales_axis_map = None
            if op.scales_input:
                scales_axis_map = (
                    tuple(range(len(op.scales_axes)))
                    if op.scales_axes
                    else op.axes
                )
            sizes_axis_map = None
            if op.sizes_input:
                sizes_axis_map = (
                    tuple(range(len(op.sizes_axes)))
                    if op.sizes_axes
                    else op.axes
                )
            roi_axis_map = None
            if op.roi_input:
                roi_axis_map = (
                    tuple(range(len(op.roi_axes)))
                    if op.roi_axes
                    else op.axes
                )
            rendered = resize_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                params=params,
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                rank=len(op.input_shape),
                loop_vars=CEmitter._loop_vars(op.output_shape),
                scales=op.scales,
                scales_input=op.scales_input,
                sizes_input=op.sizes_input,
                roi_input=op.roi_input,
                roi_suffix=roi_suffix,
                scales_suffix=scales_suffix,
                sizes_suffix=sizes_suffix,
                roi_c_type=roi_c_type,
                scales_c_type=scales_c_type,
                sizes_c_type=sizes_c_type,
                axes=op.axes,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                scales_axis_map=scales_axis_map,
                sizes_axis_map=sizes_axis_map,
                roi_axis_map=roi_axis_map,
                mode=op.mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                nearest_mode=op.nearest_mode,
                cubic_coeff_a=CEmitter._format_double(op.cubic_coeff_a),
                exclude_outside=op.exclude_outside,
                extrapolation_value=CEmitter._format_double(
                    op.extrapolation_value
                ),
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReduceOp) and op.axes_input is None:
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            if not op.input_shape:
                reduce_loop_vars = ("r0",)
                reduce_dims = (1,)
            else:
                reduce_loop_vars = tuple(
                    f"r{idx}" for idx in range(len(op.axes))
                )
                reduce_dims = tuple(op.input_shape[axis] for axis in op.axes)
            if not op.input_shape:
                input_indices = [reduce_loop_vars[0]]
            elif op.keepdims:
                input_indices = [
                    reduce_loop_vars[op.axes.index(axis)]
                    if axis in op.axes
                    else output_loop_vars[axis]
                    for axis in range(len(op.input_shape))
                ]
            else:
                kept_axes = [
                    axis
                    for axis in range(len(op.input_shape))
                    if axis not in op.axes
                ]
                input_indices = [
                    reduce_loop_vars[op.axes.index(axis)]
                    if axis in op.axes
                    else output_loop_vars[kept_axes.index(axis)]
                    for axis in range(len(op.input_shape))
                ]
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            output_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            value_expr = f"{op.input0}{input_index_expr}"
            update_expr = None
            init_literal = None
            final_expr = "acc"
            fabs_fn = CEmitter._math_fn(op.dtype, "fabsf", "fabs")
            exp_fn = CEmitter._math_fn(op.dtype, "expf", "exp")
            log_fn = CEmitter._math_fn(op.dtype, "logf", "log")
            sqrt_fn = CEmitter._math_fn(op.dtype, "sqrtf", "sqrt")
            count_literal = CEmitter._format_literal(
                op.dtype, op.reduce_count
            )
            if op.reduce_kind == "sum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
            elif op.reduce_kind == "mean":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
                final_expr = f"acc / {count_literal}"
            elif op.reduce_kind == "max":
                init_literal = min_literal
                update_expr = f"if ({value_expr} > acc) acc = {value_expr};"
            elif op.reduce_kind == "min":
                init_literal = max_literal
                update_expr = f"if ({value_expr} < acc) acc = {value_expr};"
            elif op.reduce_kind == "prod":
                init_literal = CEmitter._format_literal(op.dtype, 1)
                update_expr = f"acc *= {value_expr};"
            elif op.reduce_kind == "l1":
                init_literal = zero_literal
                update_expr = f"acc += {fabs_fn}({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
                final_expr = f"{sqrt_fn}(acc)"
            elif op.reduce_kind == "logsum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "logsumexp":
                init_literal = zero_literal
                update_expr = f"acc += {exp_fn}({value_expr});"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "sumsquare":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
            else:
                raise CodegenError(
                    f"Unsupported reduce kind {op.reduce_kind}"
                )
            rendered = reduce_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_loop_vars=reduce_loop_vars,
                reduce_dims=reduce_dims,
                output_index_expr=output_index_expr,
                init_literal=init_literal,
                update_expr=update_expr,
                final_expr=final_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ArgReduceOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            reduce_var = "r0"
            reduce_dim = op.input_shape[op.axis]
            if op.keepdims:
                input_indices = [
                    reduce_var if axis == op.axis else output_loop_vars[axis]
                    for axis in range(len(op.input_shape))
                ]
            else:
                kept_axes = [
                    axis
                    for axis in range(len(op.input_shape))
                    if axis != op.axis
                ]
                input_indices = [
                    reduce_var
                    if axis == op.axis
                    else output_loop_vars[kept_axes.index(axis)]
                    for axis in range(len(op.input_shape))
                ]
            init_indices = [
                "0" if axis == op.axis else input_indices[axis]
                for axis in range(len(op.input_shape))
            ]
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            init_index_expr = "".join(f"[{var}]" for var in init_indices)
            output_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            if op.reduce_kind == "max":
                compare_op = ">=" if op.select_last_index else ">"
            elif op.reduce_kind == "min":
                compare_op = "<=" if op.select_last_index else "<"
            else:
                raise CodegenError(
                    f"Unsupported arg reduce kind {op.reduce_kind}"
                )
            rendered = arg_reduce_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=op.input_dtype.c_type,
                output_c_type=op.output_dtype.c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_var=reduce_var,
                reduce_dim=reduce_dim,
                input_index_expr=input_index_expr,
                init_index_expr=init_index_expr,
                output_index_expr=output_index_expr,
                compare_op=compare_op,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReduceOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            input_shape = CEmitter._codegen_shape(op.input_shape)
            input_loop_vars = CEmitter._loop_vars(input_shape)
            axes_shape = op.axes_input_shape or ()
            axes_count = 1
            for dim in axes_shape:
                if dim == 0:
                    axes_count = 0
                    break
                axes_count *= dim
            axes_c_type = (
                op.axes_input_dtype.c_type
                if op.axes_input_dtype
                else ScalarType.I64.c_type
            )
            input_indices = "".join(f"[{var}]" for var in input_loop_vars)
            output_indices = "".join(
                f"[out_indices[{idx}]]" for idx in range(len(output_shape))
            )
            output_loop_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            value_expr = f"{op.input0}{input_indices}"
            update_expr = None
            init_literal = None
            post_expr = None
            fabs_fn = CEmitter._math_fn(op.dtype, "fabsf", "fabs")
            exp_fn = CEmitter._math_fn(op.dtype, "expf", "exp")
            log_fn = CEmitter._math_fn(op.dtype, "logf", "log")
            sqrt_fn = CEmitter._math_fn(op.dtype, "sqrtf", "sqrt")
            if op.reduce_kind == "sum":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
            elif op.reduce_kind == "mean":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
                post_expr = "*out_ptr = *out_ptr / reduce_count;"
            elif op.reduce_kind == "max":
                init_literal = min_literal
                update_expr = f"if ({value_expr} > *out_ptr) *out_ptr = {value_expr};"
            elif op.reduce_kind == "min":
                init_literal = max_literal
                update_expr = f"if ({value_expr} < *out_ptr) *out_ptr = {value_expr};"
            elif op.reduce_kind == "prod":
                init_literal = CEmitter._format_literal(op.dtype, 1)
                update_expr = f"*out_ptr *= {value_expr};"
            elif op.reduce_kind == "l1":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {fabs_fn}({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr} * {value_expr};"
                post_expr = f"*out_ptr = {sqrt_fn}(*out_ptr);"
            elif op.reduce_kind == "logsum":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
                post_expr = f"*out_ptr = {log_fn}(*out_ptr);"
            elif op.reduce_kind == "logsumexp":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {exp_fn}({value_expr});"
                post_expr = f"*out_ptr = {log_fn}(*out_ptr);"
            elif op.reduce_kind == "sumsquare":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr} * {value_expr};"
            else:
                raise CodegenError(
                    f"Unsupported reduce kind {op.reduce_kind}"
                )
            params = [
                f"const {c_type} {op.input0}"
                f"{self._param_array_suffix(op.input_shape)}"
            ]
            if op.axes_input and op.axes_input_shape and op.axes_input_dtype:
                axes_suffix = self._param_array_suffix(op.axes_input_shape)
                params.append(
                    f"const {axes_c_type} {op.axes_input}{axes_suffix}"
                )
            params.append(
                f"{c_type} {op.output}"
                f"{self._param_array_suffix(op.output_shape)}"
            )
            rendered = reduce_dynamic_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                params=params,
                input0=op.input0,
                axes_input=op.axes_input,
                output=op.output,
                c_type=c_type,
                axes_c_type=axes_c_type,
                input_shape=input_shape,
                output_shape=output_shape,
                input_loop_vars=input_loop_vars,
                output_loop_vars=output_loop_vars,
                output_index_expr=output_indices,
                output_loop_index_expr=output_loop_index_expr,
                input_index_expr=input_indices,
                init_literal=init_literal,
                update_expr=update_expr,
                post_expr=post_expr,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                axes_count=axes_count,
                reduce_mask_vars=tuple(
                    f"reduce_mask_{idx}"
                    for idx in range(len(input_shape))
                ),
                output_rank=len(output_shape),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConstantOfShapeOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            rendered = constant_of_shape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                value_literal=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ShapeOp):
            rendered = shape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                values=[
                    CEmitter._format_literal(op.dtype, value)
                    for value in op.values
                ],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SizeOp):
            rendered = size_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                value=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ExpandOp):
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_index_terms = [
                f"{loop_var} * {stride}"
                for loop_var, input_dim, stride in zip(
                    loop_vars, op.input_shape_padded, op.input_strides
                )
                if input_dim != 1
            ]
            input_index_expr = (
                " + ".join(input_index_terms) if input_index_terms else "0"
            )
            rendered = expand_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(
                    op.input_shape, _dim_names_for(op.input0)
                ),
                output_suffix=self._param_array_suffix(
                    op.output_shape, output_dim_names
                ),
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_index_expr=input_index_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, RangeOp):
            scalar_suffix = self._param_array_suffix(())
            rendered = range_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                start=op.start,
                limit=op.limit,
                delta=op.delta,
                output=op.output,
                c_type=c_type,
                input_suffix=scalar_suffix,
                output_suffix=self._param_array_suffix(op.output_shape),
                length=op.length,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SplitOp):
            output_names = op.outputs
            output_suffixes = tuple(
                self._param_array_suffix(
                    shape, _dim_names_for(name)
                )
                for name, shape in zip(output_names, op.output_shapes)
            )
            outer = 1
            for dim in op.input_shape[: op.axis]:
                outer *= dim
            inner = 1
            for dim in op.input_shape[op.axis + 1 :]:
                inner *= dim
            rendered = split_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                outputs=output_names,
                output_suffixes=output_suffixes,
                c_type=c_type,
                input_suffix=self._param_array_suffix(
                    op.input_shape, _dim_names_for(op.input0)
                ),
                axis_sizes=op.split_sizes,
                axis_total=op.input_shape[op.axis],
                outer=outer,
                inner=inner,
                output_count=len(output_names),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, CastOp):
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            rendered = cast_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=op.input_dtype.c_type,
                output_c_type=op.dtype.c_type,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ClipOp):
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_expr = CEmitter._broadcast_index_expr(
                op.input0, op.input_shape, op.output_shape, loop_vars
            )
            min_expr = (
                CEmitter._broadcast_index_expr(
                    op.input_min,
                    op.min_shape,
                    op.output_shape,
                    loop_vars,
                )
                if op.input_min is not None
                else op.dtype.min_literal
            )
            max_expr = (
                CEmitter._broadcast_index_expr(
                    op.input_max,
                    op.max_shape,
                    op.output_shape,
                    loop_vars,
                )
                if op.input_max is not None
                else op.dtype.max_literal
            )
            rendered = clip_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                input_min=op.input_min,
                input_max=op.input_max,
                output=op.output,
                input_c_type=op.dtype.c_type,
                output_c_type=op.dtype.c_type,
                input_suffix=self._param_array_suffix(
                    op.input_shape, _dim_names_for(op.input0)
                ),
                min_suffix=(
                    self._param_array_suffix(
                        op.min_shape, _dim_names_for(op.input_min)
                    )
                    if op.min_shape is not None
                    else None
                ),
                max_suffix=(
                    self._param_array_suffix(
                        op.max_shape, _dim_names_for(op.input_max)
                    )
                    if op.max_shape is not None
                    else None
                ),
                output_suffix=self._param_array_suffix(
                    op.output_shape, output_dim_names
                ),
                shape=output_shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                min_expr=min_expr,
                max_expr=max_expr,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, UnaryOp):
            scalar_operator = None
            if scalar_registry is not None:
                scalar_operator = self._scalar_function_name(
                    op.function, op.dtype, scalar_registry, params=op.params
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            operator_symbol = unary_op_symbol(op.function, dtype=op.dtype)
            if op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}:
                operator_symbol = (
                    "isinf" if op.function == ScalarFunction.ISINF else "isnan"
                )
            if operator_symbol is None and scalar_operator is None:
                raise CodegenError(
                    f"Unsupported unary operator for rendering: {op.function.value}"
                )
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": op.input_dtype.c_type,
                "output_c_type": op.dtype.c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
            }
            rendered = unary_template.render(
                **common,
                input0=op.input0,
                output=op.output,
                operator=scalar_operator or operator_symbol,
            ).rstrip()
            return with_node_comment(rendered)
        raise CodegenError(f"Unsupported op for rendering: {type(op).__name__}")

    @staticmethod
    def _op_output(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> str:
        if isinstance(op, SplitOp):
            return op.outputs[0]
        return op.output

    @staticmethod
    def _op_inputs(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if isinstance(op, BinaryOp):
            return ((op.input0, op.shape), (op.input1, op.shape))
        if isinstance(op, MultiInputBinaryOp):
            return tuple((name, op.shape) for name in op.inputs)
        if isinstance(op, UnaryOp):
            return ((op.input0, op.shape),)
        if isinstance(op, LpNormalizationOp):
            return ((op.input0, op.shape),)
        if isinstance(op, InstanceNormalizationOp):
            return (
                (op.input0, op.shape),
                (op.scale, (op.channels,)),
                (op.bias, (op.channels,)),
            )
        if isinstance(op, GroupNormalizationOp):
            return (
                (op.input0, op.shape),
                (op.scale, (op.channels,)),
                (op.bias, (op.channels,)),
            )
        if isinstance(op, LayerNormalizationOp):
            inputs = [(op.input0, op.shape), (op.scale, op.scale_shape)]
            if op.bias is not None and op.bias_shape is not None:
                inputs.append((op.bias, op.bias_shape))
            return tuple(inputs)
        if isinstance(op, MeanVarianceNormalizationOp):
            return ((op.input0, op.shape),)
        if isinstance(op, RMSNormalizationOp):
            return ((op.input0, op.shape), (op.scale, op.scale_shape))
        if isinstance(op, ClipOp):
            return ((op.input0, op.shape),)
        if isinstance(op, CastOp):
            return ((op.input0, op.shape),)
        if isinstance(op, IdentityOp):
            return ((op.input0, op.shape),)
        if isinstance(op, EyeLikeOp):
            return ((op.input0, op.output_shape),)
        return ()

    def _propagate_tensor_dim_names(
        self,
        resolved_ops: Sequence[
            BinaryOp
            | MultiInputBinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
        tensor_dim_names: dict[str, dict[int, str]],
    ) -> None:
        for op in resolved_ops:
            for output_name, output_shape, _ in self._op_outputs(op):
                if output_name in tensor_dim_names:
                    continue
                for input_name, input_shape in self._op_inputs(op):
                    dim_names = tensor_dim_names.get(input_name)
                    if dim_names and input_shape == output_shape:
                        tensor_dim_names[output_name] = dict(dim_names)
                        break

    @staticmethod
    def _op_outputs(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        if isinstance(op, AttentionOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = [
                (op.output, CEmitter._op_output_shape(op), op.dtype)
            ]
            if op.output_present_key is not None:
                outputs.append(
                    (
                        op.output_present_key,
                        (op.batch, op.kv_heads, op.total_seq, op.qk_head_size),
                        op.dtype,
                    )
                )
            if op.output_present_value is not None:
                outputs.append(
                    (
                        op.output_present_value,
                        (op.batch, op.kv_heads, op.total_seq, op.v_head_size),
                        op.dtype,
                    )
                )
            if op.output_qk_matmul is not None:
                outputs.append(
                    (
                        op.output_qk_matmul,
                        (op.batch, op.q_heads, op.q_seq, op.total_seq),
                        op.dtype,
                    )
            )
            return tuple(outputs)
        if isinstance(op, LstmOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = []
            if op.output_y is not None:
                if op.layout == 0:
                    y_shape = (
                        op.seq_length,
                        op.num_directions,
                        op.batch_size,
                        op.hidden_size,
                    )
                else:
                    y_shape = (
                        op.batch_size,
                        op.seq_length,
                        op.num_directions,
                        op.hidden_size,
                    )
                outputs.append((op.output_y, y_shape, op.dtype))
            if op.output_y_h is not None:
                outputs.append(
                    (
                        op.output_y_h,
                        (op.num_directions, op.batch_size, op.hidden_size),
                        op.dtype,
                    )
                )
            if op.output_y_c is not None:
                outputs.append(
                    (
                        op.output_y_c,
                        (op.num_directions, op.batch_size, op.hidden_size),
                        op.dtype,
                    )
                )
            return tuple(outputs)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            outputs = [(op.output, op.output_shape, op.dtype)]
            if op.log_prob is not None and op.log_prob_shape is not None:
                outputs.append((op.log_prob, op.log_prob_shape, op.dtype))
            return tuple(outputs)
        if isinstance(op, LayerNormalizationOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = [
                (op.output, op.shape, op.dtype)
            ]
            if op.mean_output is not None:
                mean_shape = op.shape[: op.axis] + (1,) * len(op.normalized_shape)
                outputs.append((op.mean_output, mean_shape, op.dtype))
            if op.invstd_output is not None:
                invstd_shape = op.shape[: op.axis] + (1,) * len(op.normalized_shape)
                outputs.append((op.invstd_output, invstd_shape, op.dtype))
            return tuple(outputs)
        if isinstance(op, MaxPoolOp):
            outputs = [(op.output, CEmitter._op_output_shape(op), op.dtype)]
            if op.indices is not None and op.indices_dtype is not None:
                outputs.append(
                    (op.indices, CEmitter._op_output_shape(op), op.indices_dtype)
                )
            return tuple(outputs)
        if isinstance(op, SplitOp):
            return tuple(
                (name, shape, op.dtype)
                for name, shape in zip(op.outputs, op.output_shapes)
            )
        if isinstance(op, ArgReduceOp):
            return ((op.output, CEmitter._op_output_shape(op), op.output_dtype),)
        return ((op.output, CEmitter._op_output_shape(op), op.dtype),)

    @staticmethod
    def _op_output_shape(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | SliceOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> tuple[int, ...]:
        if isinstance(op, BinaryOp):
            return op.shape
        if isinstance(op, MultiInputBinaryOp):
            return op.shape
        if isinstance(op, WhereOp):
            return op.output_shape
        if isinstance(op, UnaryOp):
            return op.shape
        if isinstance(op, ClipOp):
            return op.output_shape
        if isinstance(op, CastOp):
            return op.shape
        if isinstance(op, MatMulOp):
            return (op.m, op.n)
        if isinstance(op, GemmOp):
            return (op.m, op.n)
        if isinstance(op, ConvOp):
            return (op.batch, op.out_channels, *op.out_spatial)
        if isinstance(op, AveragePoolOp):
            return (op.batch, op.channels, op.out_h, op.out_w)
        if isinstance(op, BatchNormOp):
            return op.shape
        if isinstance(
            op,
            (
                LpNormalizationOp,
                InstanceNormalizationOp,
                GroupNormalizationOp,
                LayerNormalizationOp,
                MeanVarianceNormalizationOp,
                RMSNormalizationOp,
            ),
        ):
            return op.shape
        if isinstance(op, LrnOp):
            return op.shape
        if isinstance(op, SoftmaxOp):
            return op.shape
        if isinstance(op, LogSoftmaxOp):
            return op.shape
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return op.output_shape
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return op.output_shape
        if isinstance(op, MaxPoolOp):
            return (op.batch, op.channels, *op.out_spatial)
        if isinstance(op, ConcatOp):
            return op.output_shape
        if isinstance(op, GatherElementsOp):
            return op.output_shape
        if isinstance(op, GatherOp):
            return op.output_shape
        if isinstance(op, TransposeOp):
            return op.output_shape
        if isinstance(op, ReshapeOp):
            return op.output_shape
        if isinstance(op, IdentityOp):
            return op.shape
        if isinstance(op, EyeLikeOp):
            return op.output_shape
        if isinstance(op, TileOp):
            return op.output_shape
        if isinstance(op, DepthToSpaceOp):
            return op.output_shape
        if isinstance(op, SpaceToDepthOp):
            return op.output_shape
        if isinstance(op, SliceOp):
            return op.output_shape
        if isinstance(op, ResizeOp):
            return op.output_shape
        if isinstance(op, ReduceOp):
            return op.output_shape
        if isinstance(op, ArgReduceOp):
            return op.output_shape
        if isinstance(op, ConstantOfShapeOp):
            return op.shape
        if isinstance(op, ShapeOp):
            return op.output_shape
        if isinstance(op, SizeOp):
            return op.output_shape
        if isinstance(op, ExpandOp):
            return op.output_shape
        if isinstance(op, RangeOp):
            return op.output_shape
        if op.output_rank == 3:
            return (op.batch, op.q_seq, op.q_heads * op.v_head_size)
        return (op.batch, op.q_heads, op.q_seq, op.v_head_size)

    @staticmethod
    def _op_output_dtype(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | RangeOp
        | SplitOp,
    ) -> ScalarType:
        if isinstance(op, ArgReduceOp):
            return op.output_dtype
        return op.dtype

    @staticmethod
    def _codegen_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return (1,)
        return shape

    @staticmethod
    def _array_suffix(shape: tuple[int, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        return "".join(f"[{dim}]" for dim in shape)

    def _param_array_suffix(
        self,
        shape: tuple[int, ...],
        dim_names: Mapping[int, str] | None = None,
    ) -> str:
        shape = CEmitter._codegen_shape(shape)
        dim_names = dim_names or {}
        if not self._restrict_arrays:
            return "".join(
                f"[{dim_names.get(index, dim)}]"
                for index, dim in enumerate(shape)
            )
        first, *rest = shape
        first_dim = dim_names.get(0, first)
        rest_dims = "".join(
            f"[{dim_names.get(index + 1, dim)}]"
            for index, dim in enumerate(rest)
        )
        return f"[restrict {first_dim}]{rest_dims}"

    @staticmethod
    def _format_dim_args(dim_order: Sequence[str]) -> list[str]:
        return [f"int {dim_name}" for dim_name in dim_order]

    @staticmethod
    def _format_dim_args_prefix(dim_order: Sequence[str]) -> str:
        if not dim_order:
            return ""
        return ", ".join(f"int {dim_name}" for dim_name in dim_order) + ", "

    def _build_variable_dim_names(
        self,
        model: LoweredModel,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None,
    ) -> tuple[
        list[str],
        dict[int, dict[int, str]],
        dict[int, dict[int, str]],
        dict[str, int],
    ]:
        variable_dim_inputs = variable_dim_inputs or {}
        variable_dim_outputs = variable_dim_outputs or {}
        dim_order: list[str] = []
        dim_vars: dict[tuple[str, int, int], str] = {}
        dim_values: dict[str, int] = {}
        reserved_names = set(model.input_names) | set(model.output_names)
        used_names = set(reserved_names)
        dim_aliases: dict[str, str] = {}

        def _unique_dim_name(dim_name: str) -> str:
            if dim_name in dim_aliases:
                return dim_aliases[dim_name]
            base_name = dim_name
            if base_name in used_names:
                base_name = f"{dim_name}_dim"
            candidate = base_name
            counter = 1
            while candidate in used_names:
                counter += 1
                candidate = f"{base_name}{counter}"
            dim_aliases[dim_name] = candidate
            used_names.add(candidate)
            return candidate

        def _register_dim(
            kind: str,
            tensor_index: int,
            dim_index: int,
            dim_name: str,
            dim_value: int,
        ) -> str:
            key = (kind, tensor_index, dim_index)
            dim_name = _unique_dim_name(dim_name)
            if key not in dim_vars:
                dim_vars[key] = dim_name
                if dim_name not in dim_order:
                    dim_order.append(dim_name)
                dim_values.setdefault(dim_name, dim_value)
            else:
                if dim_values[dim_name] != dim_value:
                    raise CodegenError(
                        "Variable dimension values must be consistent, "
                        f"got {dim_values[dim_name]} and {dim_value} for {dim_name}"
                    )
            return dim_name

        def _build_dim_names(
            kind: str,
            tensor_index: int,
            shape: tuple[int, ...],
            variable_dims: Mapping[int, Mapping[int, str]],
        ) -> dict[int, str]:
            dim_names: dict[int, str] = {}
            for dim_index, dim_name in variable_dims.get(tensor_index, {}).items():
                if dim_index < 0 or dim_index >= len(shape):
                    raise CodegenError(
                        f"Variable {kind} dim {dim_index} is out of range for shape {shape}"
                    )
                dim_names[dim_index] = _register_dim(
                    kind,
                    tensor_index,
                    dim_index,
                    dim_name,
                    shape[dim_index],
                )
            return dim_names

        input_dim_names: dict[int, dict[int, str]] = {}
        for index, shape in enumerate(model.input_shapes):
            dim_names = _build_dim_names(
                "input", index, shape, variable_dim_inputs
            )
            if dim_names:
                input_dim_names[index] = dim_names

        output_dim_names: dict[int, dict[int, str]] = {}
        for index, shape in enumerate(model.output_shapes):
            dim_names = _build_dim_names(
                "output", index, shape, variable_dim_outputs
            )
            if dim_names:
                output_dim_names[index] = dim_names

        return dim_order, input_dim_names, output_dim_names, dim_values

    @staticmethod
    def _shape_dim_exprs(
        shape: tuple[int, ...],
        dim_names: Mapping[int, str] | None,
    ) -> tuple[str | int, ...]:
        dim_names = dim_names or {}
        return tuple(
            dim_names.get(index, dim) for index, dim in enumerate(shape)
        )

    @staticmethod
    def _element_count_expr(shape_exprs: Sequence[str | int]) -> str:
        if not shape_exprs:
            return "1"
        return " * ".join(str(dim) for dim in shape_exprs)

    def _build_tensor_dim_names(
        self,
        model: LoweredModel,
        input_dim_names: Mapping[int, Mapping[int, str]],
        output_dim_names: Mapping[int, Mapping[int, str]],
    ) -> dict[str, dict[int, str]]:
        dim_names: dict[str, dict[int, str]] = {}
        for index, name in enumerate(model.input_names):
            if index in input_dim_names:
                dim_names[name] = dict(input_dim_names[index])
        for index, name in enumerate(model.output_names):
            if index in output_dim_names:
                dim_names[name] = dict(output_dim_names[index])
        return dim_names

    @staticmethod
    def _loop_vars(shape: tuple[int, ...]) -> tuple[str, ...]:
        shape = CEmitter._codegen_shape(shape)
        return tuple(f"i{index}" for index in range(len(shape)))

    @staticmethod
    def _broadcast_index_expr(
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        loop_vars: tuple[str, ...],
    ) -> str:
        if not input_shape:
            return f"{name}[0]"
        output_shape = CEmitter._codegen_shape(output_shape)
        if len(output_shape) != len(loop_vars):
            raise CodegenError("Loop variables must match output shape rank")
        offset = len(output_shape) - len(input_shape)
        indices: list[str] = []
        for input_dim, dim_size in enumerate(input_shape):
            output_dim = input_dim + offset
            if dim_size == 1:
                indices.append("[0]")
            else:
                indices.append(f"[{loop_vars[output_dim]}]")
        return f"{name}{''.join(indices)}"

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        shape = CEmitter._codegen_shape(shape)
        count = 1
        for dim in shape:
            if dim < 0:
                raise CodegenError("Dynamic dims are not supported")
            if dim == 0:
                return 0
            count *= dim
        return count

    @staticmethod
    def _matmul_index_exprs(
        op: MatMulOp,
        batch_vars: tuple[str, ...],
        row_var: str | None,
        col_var: str | None,
        batch_rank: int,
    ) -> tuple[str, str]:
        def batch_indices(
            batch_shape: tuple[int, ...], actual_rank: int
        ) -> list[str]:
            if actual_rank == 0:
                return []
            offset = batch_rank - actual_rank
            indices: list[str] = []
            for idx in range(actual_rank):
                dim = batch_shape[offset + idx]
                var = batch_vars[offset + idx]
                indices.append("0" if dim == 1 else var)
            return indices

        if op.left_vector:
            input0_indices = ["k"]
        else:
            input0_batch_rank = len(op.input0_shape) - 2
            input0_indices = batch_indices(
                op.input0_batch_shape, input0_batch_rank
            )
            input0_indices.append(row_var if row_var is not None else "0")
            input0_indices.append("k")
        if op.right_vector:
            input1_indices = ["k"]
        else:
            input1_batch_rank = len(op.input1_shape) - 2
            input1_indices = batch_indices(
                op.input1_batch_shape, input1_batch_rank
            )
            input1_indices.append("k")
            input1_indices.append(col_var if col_var is not None else "0")
        input0_index_expr = f"{op.input0}" + "".join(
            f"[{index}]" for index in input0_indices
        )
        input1_index_expr = f"{op.input1}" + "".join(
            f"[{index}]" for index in input1_indices
        )
        return input0_index_expr, input1_index_expr

    def _emit_testbench(
        self,
        model: LoweredModel,
        testbench_template,
        *,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        dim_order: Sequence[str],
        dim_values: Mapping[str, int],
    ) -> str:
        input_counts = tuple(
            self._element_count(shape) for shape in model.input_shapes
        )
        testbench_inputs = testbench_inputs or {}
        inputs = []
        for name, shape, count, dtype in zip(
            model.input_names, model.input_shapes, input_counts, model.input_dtypes
        ):
            codegen_shape = self._codegen_shape(shape)
            loop_vars = self._loop_vars(codegen_shape)
            if dtype in {ScalarType.F16, ScalarType.F32}:
                random_expr = "rng_next_float()"
            elif dtype == ScalarType.F64:
                random_expr = "rng_next_double()"
            elif dtype == ScalarType.BOOL:
                random_expr = "((rng_next_u64() & 1ull) != 0)"
            else:
                random_expr = f"({dtype.c_type})rng_next_i64()"
            constant_values = testbench_inputs.get(name)
            constant_name = None
            constant_lines = None
            if constant_values is not None:
                constant_name = f"{name}_testbench_data"
                constant_lines = [
                    self._format_value(value, dtype)
                    for value in constant_values
                ]
            inputs.append(
                {
                    "name": name,
                    "shape": codegen_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": count,
                    "array_suffix": self._array_suffix(codegen_shape),
                    "loop_vars": loop_vars,
                    "rank": len(codegen_shape),
                    "index_expr": self._index_expr(codegen_shape, loop_vars),
                    "dtype": dtype,
                    "c_type": dtype.c_type,
                    "random_expr": random_expr,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                    "constant_name": constant_name,
                    "constant_lines": constant_lines,
                }
            )
        outputs = []
        for name, shape, dtype in zip(
            model.output_names, model.output_shapes, model.output_dtypes
        ):
            codegen_shape = self._codegen_shape(shape)
            output_loop_vars = self._loop_vars(codegen_shape)
            outputs.append(
                {
                    "name": name,
                    "shape": codegen_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": self._element_count(codegen_shape),
                    "array_suffix": self._array_suffix(codegen_shape),
                    "loop_vars": output_loop_vars,
                    "rank": len(codegen_shape),
                    "index_expr": self._index_expr(codegen_shape, output_loop_vars),
                    "dtype": dtype,
                    "c_type": dtype.c_type,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                }
            )
        rendered = testbench_template.render(
            model_name=model.name,
            dim_args=[
                {"name": dim_name, "value": dim_values[dim_name]}
                for dim_name in dim_order
            ],
            inputs=inputs,
            outputs=outputs,
        ).rstrip()
        return _format_c_indentation(rendered)

    def _emit_constant_definitions(
        self,
        constants: tuple[ConstTensor, ...],
        *,
        storage_prefix: str = "static const",
    ) -> str:
        if not constants:
            return ""
        lines: list[str] = []
        for const in constants:
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape)
            values = [
                self._format_value(value, const.dtype) for value in const.data
            ]
            lines.append(
                f"{storage_prefix} {c_type} {const.name}{array_suffix} = {{"
            )
            if values:
                chunk_size = 8
                chunks = [
                    values[index : index + chunk_size]
                    for index in range(0, len(values), chunk_size)
                ]
                for chunk_index, chunk in enumerate(chunks):
                    line = "    " + ", ".join(chunk)
                    if chunk_index != len(chunks) - 1:
                        line += ","
                    lines.append(line)
            lines.append("};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _emit_constant_declarations(
        self, constants: tuple[ConstTensor, ...]
    ) -> str:
        if not constants:
            return ""
        lines = []
        for const in constants:
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape)
            lines.append(f"extern const {c_type} {const.name}{array_suffix};")
        return "\n".join(lines)

    @staticmethod
    def _index_expr(shape: tuple[int, ...], loop_vars: tuple[str, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        if len(shape) != len(loop_vars):
            raise CodegenError("Loop variables must match shape rank")
        if not shape:
            return "0"
        expr = loop_vars[0]
        for dim, var in zip(shape[1:], loop_vars[1:]):
            expr = f"({expr} * {dim} + {var})"
        return expr

    @staticmethod
    def _format_float(value: float) -> str:
        formatted = f"{value:.9g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return f"{formatted}f"

    @staticmethod
    def _format_float16(value: float) -> str:
        return f"(_Float16){CEmitter._format_float(value)}"

    @staticmethod
    def _format_double(value: float) -> str:
        formatted = f"{value:.17g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return formatted

    @staticmethod
    def _format_floating(value: float, dtype: ScalarType) -> str:
        if dtype == ScalarType.F64:
            return CEmitter._format_double(value)
        if dtype == ScalarType.F16:
            return CEmitter._format_float16(value)
        return CEmitter._format_float(value)

    @staticmethod
    def _math_fn(dtype: ScalarType, float_name: str, double_name: str) -> str:
        if dtype == ScalarType.F64:
            return double_name
        return float_name

    @staticmethod
    def _format_int64(value: int) -> str:
        min_value = -(2**63)
        if value == min_value:
            return "INT64_MIN"
        return f"{int(value)}LL"

    @staticmethod
    def _format_int(value: int, bits: int, min_macro: str) -> str:
        min_value = -(2 ** (bits - 1))
        if value == min_value:
            return min_macro
        return str(int(value))

    @staticmethod
    def _format_uint(value: int, bits: int, max_macro: str) -> str:
        max_value = 2**bits - 1
        if value == max_value:
            return max_macro
        return str(int(value))

    @staticmethod
    def _format_literal(dtype: ScalarType, value: float | int | bool) -> str:
        if dtype == ScalarType.F16:
            return CEmitter._format_float16(float(value))
        if dtype == ScalarType.F32:
            return CEmitter._format_float(float(value))
        if dtype == ScalarType.F64:
            return CEmitter._format_double(float(value))
        if dtype == ScalarType.BOOL:
            return "true" if bool(value) else "false"
        if dtype == ScalarType.U64:
            return CEmitter._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == ScalarType.U32:
            return CEmitter._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == ScalarType.U16:
            return CEmitter._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == ScalarType.U8:
            return CEmitter._format_uint(int(value), 8, "UINT8_MAX")
        if dtype == ScalarType.I64:
            return CEmitter._format_int64(int(value))
        if dtype == ScalarType.I32:
            return CEmitter._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return CEmitter._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return CEmitter._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _format_value(self, value: float | int | bool, dtype: ScalarType) -> str:
        if dtype == ScalarType.F16:
            return self._format_float16(float(value))
        if dtype == ScalarType.F32:
            return self._format_float(float(value))
        if dtype == ScalarType.F64:
            return self._format_double(float(value))
        if dtype == ScalarType.BOOL:
            return "true" if bool(value) else "false"
        if dtype == ScalarType.U64:
            return self._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == ScalarType.U32:
            return self._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == ScalarType.U16:
            return self._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == ScalarType.U8:
            return self._format_uint(int(value), 8, "UINT8_MAX")
        if dtype == ScalarType.I64:
            return self._format_int64(int(value))
        if dtype == ScalarType.I32:
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return self._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    @staticmethod
    def _print_format(dtype: ScalarType) -> str:
        if dtype == ScalarType.F16:
            return "%.8g"
        if dtype == ScalarType.F32:
            return "%.8g"
        if dtype == ScalarType.F64:
            return "%.17g"
        if dtype == ScalarType.BOOL:
            return "%d"
        if dtype == ScalarType.U64:
            return "%llu"
        if dtype == ScalarType.U32:
            return "%u"
        if dtype == ScalarType.U16:
            return "%hu"
        if dtype == ScalarType.U8:
            return "%hhu"
        if dtype == ScalarType.I64:
            return "%lld"
        if dtype == ScalarType.I32:
            return "%d"
        if dtype == ScalarType.I16:
            return "%hd"
        if dtype == ScalarType.I8:
            return "%hhd"
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    @staticmethod
    def _print_cast(dtype: ScalarType) -> str:
        if dtype.is_float:
            return "(double)"
        if dtype == ScalarType.BOOL:
            return "(int)"
        if dtype == ScalarType.U64:
            return "(unsigned long long)"
        if dtype in {ScalarType.U32, ScalarType.U16, ScalarType.U8}:
            return "(unsigned int)"
        if dtype == ScalarType.I64:
            return "(long long)"
        if dtype in {ScalarType.I32, ScalarType.I16, ScalarType.I8}:
            return "(int)"
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")


def _format_multiline_value(value: str | None) -> list[str]:
    if not value:
        return ["  n/a"]
    lines = value.splitlines() or [""]
    return [f"  {line}" if line else "  " for line in lines]
