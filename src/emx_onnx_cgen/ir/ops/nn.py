from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shared.scalar_types import ScalarType

from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import ConvLikeOpBase, GemmLikeOpBase, MatMulLikeOpBase, RenderableOpBase
from ..op_context import OpContext


class EinsumKind(str, Enum):
    REDUCE_ALL = "reduce_all"
    SUM_J = "sum_j"
    TRANSPOSE = "transpose"
    DOT = "dot"
    BATCH_MATMUL = "batch_matmul"
    BATCH_DIAGONAL = "batch_diagonal"


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        if dim < 0:
            raise ShapeInferenceError("Dynamic dims are not supported")
        product *= dim
    return product


def _broadcast_batch_shapes(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    max_rank = max(len(left), len(right))
    left_padded = (1,) * (max_rank - len(left)) + left
    right_padded = (1,) * (max_rank - len(right)) + right
    broadcast_shape: list[int] = []
    for left_dim, right_dim in zip(left_padded, right_padded):
        if not (left_dim == right_dim or left_dim == 1 or right_dim == 1):
            raise ShapeInferenceError(
                "MatMul batch dimensions must be broadcastable, "
                f"got {left} x {right}"
            )
        broadcast_shape.append(max(left_dim, right_dim))
    return tuple(broadcast_shape), left_padded, right_padded


def _resolve_matmul_spec(ctx: OpContext, input0: str, input1: str) -> dict[str, object]:
    input0_shape = ctx.shape(input0)
    input1_shape = ctx.shape(input1)
    if len(input0_shape) < 1 or len(input1_shape) < 1:
        raise UnsupportedOpError(
            "MatMul inputs must be at least 1D, " f"got {input0_shape} x {input1_shape}"
        )
    left_vector = len(input0_shape) == 1
    right_vector = len(input1_shape) == 1
    input0_effective = (1, input0_shape[0]) if left_vector else input0_shape
    input1_effective = (input1_shape[0], 1) if right_vector else input1_shape
    m, k_left = input0_effective[-2], input0_effective[-1]
    k_right, n = input1_effective[-2], input1_effective[-1]
    if k_left != k_right:
        raise ShapeInferenceError(
            f"MatMul inner dimensions must match, got {k_left} and {k_right}"
        )
    batch_shape, input0_batch_shape, input1_batch_shape = _broadcast_batch_shapes(
        input0_effective[:-2],
        input1_effective[:-2],
    )
    if left_vector and right_vector:
        output_shape = batch_shape
    elif left_vector:
        output_shape = batch_shape + (n,)
    elif right_vector:
        output_shape = batch_shape + (m,)
    else:
        output_shape = batch_shape + (m, n)
    return {
        "input0_shape": input0_shape,
        "input1_shape": input1_shape,
        "output_shape": output_shape,
        "batch_shape": batch_shape,
        "input0_batch_shape": input0_batch_shape,
        "input1_batch_shape": input1_batch_shape,
        "m": m,
        "n": n,
        "k": k_left,
        "left_vector": left_vector,
        "right_vector": right_vector,
    }


@dataclass(frozen=True)
class MatMulOp(MatMulLikeOpBase):
    input0: str
    input1: str
    output: str

    def infer_types(self, ctx: OpContext) -> None:
        input0_dtype = ctx.dtype(self.input0)
        input1_dtype = ctx.dtype(self.input1)
        if input0_dtype != input1_dtype:
            raise UnsupportedOpError(
                "MatMul expects matching input dtypes, "
                f"got {input0_dtype.onnx_name} and {input1_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input0_dtype)
            output_dtype = input0_dtype
        if output_dtype != input0_dtype:
            raise UnsupportedOpError(
                "MatMul expects output dtype to match inputs, "
                f"got {output_dtype.onnx_name} and {input0_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        spec = _resolve_matmul_spec(ctx, self.input0, self.input1)
        output_shape = spec["output_shape"]
        try:
            expected = ctx.shape(self.output)
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"MatMul output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self.output, output_shape)
        ctx.set_derived(self, "batch_shape", spec["batch_shape"])
        ctx.set_derived(self, "input0_batch_shape", spec["input0_batch_shape"])
        ctx.set_derived(self, "input1_batch_shape", spec["input1_batch_shape"])
        ctx.set_derived(self, "m", spec["m"])
        ctx.set_derived(self, "n", spec["n"])
        ctx.set_derived(self, "k", spec["k"])
        ctx.set_derived(self, "left_vector", spec["left_vector"])
        ctx.set_derived(self, "right_vector", spec["right_vector"])


@dataclass(frozen=True)
class QLinearMatMulOp(MatMulLikeOpBase):
    __io_inputs__ = (
        "input0",
        "input0_scale",
        "input0_zero_point",
        "input1",
        "input1_scale",
        "input1_zero_point",
        "output_scale",
        "output_zero_point",
    )
    input0: str
    input0_scale: str
    input0_zero_point: str
    input1: str
    input1_scale: str
    input1_zero_point: str
    output_scale: str
    output_zero_point: str
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
    input0_dtype: ScalarType
    input1_dtype: ScalarType
    dtype: ScalarType
    input0_scale_dtype: ScalarType
    input1_scale_dtype: ScalarType
    output_scale_dtype: ScalarType
    input0_scale_shape: tuple[int, ...]
    input1_scale_shape: tuple[int, ...]
    output_scale_shape: tuple[int, ...]
    input0_zero_shape: tuple[int, ...]
    input1_zero_shape: tuple[int, ...]
    output_zero_shape: tuple[int, ...]


@dataclass(frozen=True)
class MatMulIntegerOp(MatMulLikeOpBase):
    __io_inputs__ = (
        "input0",
        "input1",
        "input0_zero_point",
        "input1_zero_point",
    )
    input0: str
    input1: str
    input0_zero_point: str | None
    input1_zero_point: str | None
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
    input0_dtype: ScalarType
    input1_dtype: ScalarType
    dtype: ScalarType
    input0_zero_shape: tuple[int, ...] | None
    input1_zero_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class EinsumOp(MatMulLikeOpBase):
    __io_inputs__ = ("inputs",)
    inputs: tuple[str, ...]
    output: str
    kind: EinsumKind
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class GemmOp(GemmLikeOpBase):
    input_a: str
    input_b: str
    input_c: str | None
    output: str
    trans_a: int
    trans_b: int
    alpha: float | int
    beta: float | int

    @staticmethod
    def _normalize_attrs(
        dtype: ScalarType,
        *,
        alpha: float | int,
        beta: float | int,
        trans_a: int,
        trans_b: int,
    ) -> tuple[float | int, float | int, bool, bool]:
        if trans_a not in {0, 1} or trans_b not in {0, 1}:
            raise UnsupportedOpError(
                "Gemm only supports transA/transB values of 0 or 1"
            )
        if dtype == ScalarType.BOOL:
            raise UnsupportedOpError("Gemm supports numeric inputs only")
        if not dtype.is_float:
            alpha_int = int(alpha)
            beta_int = int(beta)
            if alpha != alpha_int or beta != beta_int:
                raise UnsupportedOpError(
                    "Gemm alpha and beta must be integers for non-float inputs"
                )
            alpha = alpha_int
            beta = beta_int
        return alpha, beta, bool(trans_a), bool(trans_b)

    @staticmethod
    def _validate_bias_shape(
        output_shape: tuple[int, int], bias_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], str]:
        if len(bias_shape) == 0:
            return bias_shape, "scalar"
        if len(bias_shape) == 1:
            m, n = output_shape
            if bias_shape[0] == 1:
                return bias_shape, "scalar"
            if bias_shape[0] == n:
                return bias_shape, "col"
            if bias_shape[0] == m:
                return bias_shape, "row"
            raise ShapeInferenceError(
                "Gemm bias input must be broadcastable to output shape, "
                f"got {bias_shape} vs {output_shape}"
            )
        if len(bias_shape) == 2:
            m, n = output_shape
            if bias_shape[0] not in {1, m} or bias_shape[1] not in {1, n}:
                raise ShapeInferenceError(
                    "Gemm bias input must be broadcastable to output shape, "
                    f"got {bias_shape} vs {output_shape}"
                )
            return bias_shape, "matrix"
        raise ShapeInferenceError(
            f"Gemm bias input must be rank 1 or 2, got {bias_shape}"
        )

    def infer_types(self, ctx: OpContext) -> None:
        input_a_dtype = ctx.dtype(self.input_a)
        input_b_dtype = ctx.dtype(self.input_b)
        if input_a_dtype != input_b_dtype:
            raise UnsupportedOpError(
                "Gemm expects matching input dtypes, "
                f"got {input_a_dtype.onnx_name} and {input_b_dtype.onnx_name}"
            )
        if self.input_c is not None:
            input_c_dtype = ctx.dtype(self.input_c)
            if input_c_dtype != input_a_dtype:
                raise UnsupportedOpError(
                    "Gemm expects bias dtype to match inputs, "
                    f"got {input_c_dtype.onnx_name} and {input_a_dtype.onnx_name}"
                )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_a_dtype)
            output_dtype = input_a_dtype
        if output_dtype != input_a_dtype:
            raise UnsupportedOpError(
                "Gemm expects output dtype to match inputs, "
                f"got {output_dtype.onnx_name} and {input_a_dtype.onnx_name}"
            )
        alpha, beta, trans_a, trans_b = self._normalize_attrs(
            output_dtype,
            alpha=self.alpha,
            beta=self.beta,
            trans_a=self.trans_a,
            trans_b=self.trans_b,
        )
        ctx.set_derived(self, "alpha", alpha)
        ctx.set_derived(self, "beta", beta)
        ctx.set_derived(self, "trans_a", trans_a)
        ctx.set_derived(self, "trans_b", trans_b)

    def infer_shapes(self, ctx: OpContext) -> None:
        trans_a = ctx.require_derived(self, "trans_a")
        trans_b = ctx.require_derived(self, "trans_b")
        input_a_shape = ctx.shape(self.input_a)
        input_b_shape = ctx.shape(self.input_b)
        if len(input_a_shape) != 2 or len(input_b_shape) != 2:
            raise UnsupportedOpError(
                "Gemm supports 2D inputs only, "
                f"got {input_a_shape} x {input_b_shape}"
            )
        if trans_a:
            m, k_left = input_a_shape[1], input_a_shape[0]
        else:
            m, k_left = input_a_shape
        if trans_b:
            n, k_right = input_b_shape[0], input_b_shape[1]
        else:
            k_right, n = input_b_shape
        if k_left != k_right:
            raise ShapeInferenceError(
                f"Gemm inner dimensions must match, got {k_left} and {k_right}"
            )
        output_shape = (m, n)
        try:
            expected = ctx.shape(self.output)
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"Gemm output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self.output, output_shape)
        c_shape = None
        c_axis = "none"
        if self.input_c is not None:
            bias_shape = ctx.shape(self.input_c)
            c_shape, c_axis = self._validate_bias_shape(output_shape, bias_shape)
        ctx.set_derived(self, "m", m)
        ctx.set_derived(self, "n", n)
        ctx.set_derived(self, "k", k_left)
        ctx.set_derived(self, "c_shape", c_shape)
        ctx.set_derived(self, "c_axis", c_axis)


@dataclass(frozen=True)
class AttentionOp(RenderableOpBase):
    __io_inputs__ = (
        "input_q",
        "input_k",
        "input_v",
        "input_attn_mask",
        "input_past_key",
        "input_past_value",
        "input_nonpad_kv_seqlen",
    )
    __io_outputs__ = (
        "output",
        "output_present_key",
        "output_present_value",
        "output_qk_matmul",
    )
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
class RotaryEmbeddingOp(RenderableOpBase):
    __io_inputs__ = ("input0", "cos_cache", "sin_cache", "position_ids")
    __io_outputs__ = ("output",)
    input0: str
    cos_cache: str
    sin_cache: str
    position_ids: str | None
    output: str
    input_shape: tuple[int, ...]
    cos_shape: tuple[int, ...]
    sin_shape: tuple[int, ...]
    position_ids_shape: tuple[int, ...] | None
    dtype: ScalarType
    position_ids_dtype: ScalarType | None
    rotary_dim: int
    rotary_dim_half: int
    head_size: int
    num_heads: int
    seq_len: int
    batch: int
    input_rank: int
    interleaved: bool


@dataclass(frozen=True)
class ConvOp(ConvLikeOpBase):
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
class ConvIntegerOp(ConvLikeOpBase):
    __io_inputs__ = ("input0", "weights", "x_zero_point", "w_zero_point")
    input0: str
    weights: str
    x_zero_point: str | None
    w_zero_point: str | None
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
    input_dtype: ScalarType
    weight_dtype: ScalarType
    dtype: ScalarType
    x_zero_point_shape: tuple[int, ...] | None
    w_zero_point_shape: tuple[int, ...] | None
    w_zero_point_per_channel: bool


@dataclass(frozen=True)
class QLinearConvOp(ConvLikeOpBase):
    __io_inputs__ = (
        "input0",
        "input_scale",
        "input_zero_point",
        "weights",
        "weight_scale",
        "weight_zero_point",
        "output_scale",
        "output_zero_point",
        "bias",
    )
    input0: str
    input_scale: str
    input_zero_point: str
    weights: str
    weight_scale: str
    weight_zero_point: str
    output_scale: str
    output_zero_point: str
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
    input_dtype: ScalarType
    weight_dtype: ScalarType
    dtype: ScalarType
    input_scale_dtype: ScalarType
    weight_scale_dtype: ScalarType
    output_scale_dtype: ScalarType
    input_scale_shape: tuple[int, ...]
    weight_scale_shape: tuple[int, ...]
    output_scale_shape: tuple[int, ...]
    input_zero_shape: tuple[int, ...]
    weight_zero_shape: tuple[int, ...]
    output_zero_shape: tuple[int, ...]
    weight_scale_per_channel: bool
    weight_zero_per_channel: bool


@dataclass(frozen=True)
class ConvTransposeOp(ConvLikeOpBase):
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
    output_padding: tuple[int, ...]
    group: int
    dtype: ScalarType


@dataclass(frozen=True)
class AveragePoolOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    dilation_h: int
    dilation_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool
    dtype: ScalarType
    spatial_rank: int = 2
    in_d: int = 1
    out_d: int = 1
    kernel_d: int = 1
    dilation_d: int = 1
    stride_d: int = 1
    pad_front: int = 0
    pad_back: int = 0


@dataclass(frozen=True)
class QLinearAveragePoolOp(RenderableOpBase):
    __io_inputs__ = (
        "input0",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
    )
    __io_outputs__ = ("output",)
    input0: str
    input_scale: str
    input_zero_point: str
    output_scale: str
    output_zero_point: str
    output: str
    batch: int
    channels: int
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    dilation_h: int
    dilation_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool
    input_dtype: ScalarType
    dtype: ScalarType
    input_scale_dtype: ScalarType
    output_scale_dtype: ScalarType
    input_scale_shape: tuple[int, ...]
    output_scale_shape: tuple[int, ...]
    input_zero_shape: tuple[int, ...]
    output_zero_shape: tuple[int, ...]
    spatial_rank: int = 2
    in_d: int = 1
    out_d: int = 1
    kernel_d: int = 1
    dilation_d: int = 1
    stride_d: int = 1
    pad_front: int = 0
    pad_back: int = 0


@dataclass(frozen=True)
class LpPoolOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    dilation_h: int
    dilation_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    p: int
    dtype: ScalarType


@dataclass(frozen=True)
class SoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        if not input_dtype.is_float:
            raise UnsupportedOpError(
                "Softmax supports bfloat16, float16, float, and double inputs only"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            return None
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                "Softmax expects output dtype to match input dtype"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axis = self.axis
        if axis is None:
            axis = 1 if self.use_legacy_axis_semantics else -1
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis >= len(input_shape):
            raise ShapeInferenceError(
                f"Softmax axis {self.axis} is out of bounds for shape {input_shape}"
            )
        if self.use_legacy_axis_semantics:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = _shape_product(input_shape[axis:])
            inner = 1
        else:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = input_shape[axis]
            inner = (
                _shape_product(input_shape[axis + 1 :])
                if axis + 1 < len(input_shape)
                else 1
            )
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)
        ctx.set_derived(self, "outer", outer)
        ctx.set_derived(self, "axis_size", axis_size)
        ctx.set_derived(self, "inner", inner)


@dataclass(frozen=True)
class LogSoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        if not input_dtype.is_float:
            raise UnsupportedOpError(
                "LogSoftmax supports bfloat16, float16, float, and double inputs only"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            return None
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                "LogSoftmax expects output dtype to match input dtype"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axis = self.axis
        if axis is None:
            axis = 1 if self.use_legacy_axis_semantics else -1
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis >= len(input_shape):
            raise ShapeInferenceError(
                f"LogSoftmax axis {self.axis} is out of bounds for shape {input_shape}"
            )
        if self.use_legacy_axis_semantics:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = _shape_product(input_shape[axis:])
            inner = 1
        else:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = input_shape[axis]
            inner = (
                _shape_product(input_shape[axis + 1 :])
                if axis + 1 < len(input_shape)
                else 1
            )
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)
        ctx.set_derived(self, "outer", outer)
        ctx.set_derived(self, "axis_size", axis_size)
        ctx.set_derived(self, "inner", inner)


@dataclass(frozen=True)
class HardmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        if input_dtype not in {
            ScalarType.F16,
            ScalarType.BF16,
            ScalarType.F32,
            ScalarType.F64,
        }:
            raise UnsupportedOpError(
                "Hardmax supports bfloat16, float16, float, and double inputs only"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            return None
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                "Hardmax expects output dtype to match input dtype"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axis = self.axis
        legacy_axis_semantics = False
        if axis is None:
            opset_version = ctx.opset_version()
            legacy_axis_semantics = opset_version is not None and opset_version < 13
            axis = 1 if legacy_axis_semantics else -1
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis >= len(input_shape):
            raise ShapeInferenceError(
                f"Hardmax axis {self.axis} is out of bounds for shape {input_shape}"
            )
        if legacy_axis_semantics:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = _shape_product(input_shape[axis:])
            inner = 1
        else:
            outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
            axis_size = input_shape[axis]
            inner = (
                _shape_product(input_shape[axis + 1 :])
                if axis + 1 < len(input_shape)
                else 1
            )
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)
        ctx.set_derived(self, "outer", outer)
        ctx.set_derived(self, "axis_size", axis_size)
        ctx.set_derived(self, "inner", inner)


@dataclass(frozen=True)
class NegativeLogLikelihoodLossOp(RenderableOpBase):
    __io_inputs__ = ("input0", "target", "weight")
    __io_outputs__ = ("output",)
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
class SoftmaxCrossEntropyLossOp(RenderableOpBase):
    __io_inputs__ = ("input0", "target", "weight")
    __io_outputs__ = ("output", "log_prob")
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
class BatchNormOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "bias", "mean", "variance")
    __io_outputs__ = ("output", "running_mean", "running_variance")
    input0: str
    scale: str
    bias: str
    mean: str
    variance: str
    output: str
    running_mean: str | None
    running_variance: str | None
    shape: tuple[int, ...]
    channels: int
    epsilon: float
    momentum: float
    training_mode: bool
    dtype: ScalarType


@dataclass(frozen=True)
class LpNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
class InstanceNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "bias")
    __io_outputs__ = ("output",)
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
class GroupNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "bias")
    __io_outputs__ = ("output",)
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
class LayerNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "bias")
    __io_outputs__ = ("output", "mean_output", "invstd_output")
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
class MeanVarianceNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    non_axes: tuple[int, ...]
    reduce_count: int
    epsilon: float
    dtype: ScalarType


@dataclass(frozen=True)
class RMSNormalizationOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale")
    __io_outputs__ = ("output",)
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
class LrnOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
class GruOp(RenderableOpBase):
    __io_inputs__ = (
        "input_x",
        "input_w",
        "input_r",
        "input_b",
        "input_sequence_lens",
        "input_initial_h",
    )
    __io_outputs__ = ("output_y", "output_y_h")
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
    linear_before_reset: int
    clip: float | None
    activation_kinds: tuple[int, ...]
    activation_alphas: tuple[float, ...]
    activation_betas: tuple[float, ...]
    dtype: ScalarType
    sequence_lens_dtype: ScalarType | None


@dataclass(frozen=True)
class LstmOp(RenderableOpBase):
    __io_inputs__ = (
        "input_x",
        "input_w",
        "input_r",
        "input_b",
        "input_sequence_lens",
        "input_initial_h",
        "input_initial_c",
        "input_p",
    )
    __io_outputs__ = ("output_y", "output_y_h", "output_y_c")
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
class AdagradOp(RenderableOpBase):
    __io_inputs__ = ("rate", "timestep", "inputs", "gradients", "accumulators")
    __io_outputs__ = ("outputs", "accumulator_outputs")
    rate: str
    timestep: str
    inputs: tuple[str, ...]
    gradients: tuple[str, ...]
    accumulators: tuple[str, ...]
    outputs: tuple[str, ...]
    accumulator_outputs: tuple[str, ...]
    rate_shape: tuple[int, ...]
    timestep_shape: tuple[int, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtype: ScalarType
    rate_dtype: ScalarType
    timestep_dtype: ScalarType
    norm_coefficient: float
    epsilon: float
    decay_factor: float

    def call_args(self) -> tuple[str, ...]:
        args = [self.rate, self.timestep]
        for index in range(len(self.inputs)):
            args.extend(
                [
                    self.inputs[index],
                    self.gradients[index],
                    self.accumulators[index],
                    self.outputs[index],
                    self.accumulator_outputs[index],
                ]
            )
        return tuple(args)


@dataclass(frozen=True)
class MaxPoolOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output", "indices")
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
