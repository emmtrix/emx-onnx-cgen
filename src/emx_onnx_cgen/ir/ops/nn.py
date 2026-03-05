from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shared.scalar_types import ScalarType

from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import (
    CEmitterCompat,
    ConvLikeOpBase,
    EmitContext,
    Emitter,
    GemmLikeOpBase,
    MatMulLikeOpBase,
    RenderableOpBase,
)
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
                f"MatMul batch dimensions must be broadcastable, got {left} x {right}"
            )
        broadcast_shape.append(max(left_dim, right_dim))
    return tuple(broadcast_shape), left_padded, right_padded


def _resolve_matmul_spec(ctx: OpContext, input0: str, input1: str) -> dict[str, object]:
    input0_shape = ctx.shape(input0)
    input1_shape = ctx.shape(input1)
    if len(input0_shape) < 1 or len(input1_shape) < 1:
        raise UnsupportedOpError(
            f"MatMul inputs must be at least 1D, got {input0_shape} x {input1_shape}"
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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        params = emitter.shared_param_map(
            [
                ("input_a", self.input_a),
                ("input_b", self.input_b),
                ("input_c", self.input_c),
                ("output", self.output),
            ]
        )
        m = int(emitter.derived(self, "m"))
        n = int(emitter.derived(self, "n"))
        k = int(emitter.derived(self, "k"))
        trans_a = bool(emitter.derived(self, "trans_a"))
        trans_b = bool(emitter.derived(self, "trans_b"))
        c_shape = emitter.derived(self, "c_shape")
        c_axis = str(emitter.derived(self, "c_axis"))
        input_a_shape = (k, m) if trans_a else (m, k)
        input_b_shape = (n, k) if trans_b else (k, n)
        input_a_suffix = emitter.param_array_suffix(input_a_shape)
        input_b_suffix = emitter.param_array_suffix(input_b_shape)
        output_suffix = emitter.param_array_suffix((m, n))
        c_suffix = emitter.param_array_suffix(c_shape) if c_shape is not None else ""
        param_decls = emitter.build_param_decls(
            [
                (params["input_a"], c_type, input_a_suffix, True),
                (params["input_b"], c_type, input_b_suffix, True),
                (
                    (
                        params["input_c"],
                        c_type,
                        c_suffix,
                        True,
                    )
                    if params["input_c"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        dtype = emitter.ctx_dtype(self.output)
        alpha_literal = emitter.format_literal(dtype, emitter.derived(self, "alpha"))
        beta_literal = emitter.format_literal(dtype, emitter.derived(self, "beta"))
        acc_dtype = emitter.accumulation_dtype(dtype)
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        if c_shape is None:
            c_rank = 0
            c_dim0 = 0
            c_dim1 = 0
        elif len(c_shape) == 0:
            c_rank = 0
            c_dim0 = 0
            c_dim1 = 0
        elif len(c_shape) == 1:
            c_rank = 1
            c_dim0 = 1
            c_dim1 = c_shape[0]
        else:
            c_rank = 2
            c_dim0 = c_shape[0]
            c_dim1 = c_shape[1]
        rendered = state.templates["gemm"].render(
            model_name=model.name,
            op_name=op_name,
            input_a=params["input_a"],
            input_b=params["input_b"],
            input_c=params["input_c"],
            output=params["output"],
            params=param_decls,
            c_type=dtype.c_type,
            acc_type=acc_dtype.c_type,
            zero_literal=acc_zero_literal,
            alpha_literal=alpha_literal,
            beta_literal=beta_literal,
            trans_a=int(trans_a),
            trans_b=int(trans_b),
            m=m,
            n=n,
            k=k,
            input_a_suffix=input_a_suffix,
            input_b_suffix=input_b_suffix,
            output_suffix=output_suffix,
            c_suffix=(c_suffix if c_shape is not None else None),
            c_rank=c_rank,
            c_dim0=c_dim0,
            c_dim1=c_dim1,
            c_axis=c_axis,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)

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
                f"Gemm supports 2D inputs only, got {input_a_shape} x {input_b_shape}"
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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("weights", self.weights),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )
        acc_dtype = emitter.accumulation_dtype(self.dtype)
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        input_shape = (self.batch, self.in_channels, *self.in_spatial)
        weight_shape = (
            self.out_channels,
            self.in_channels // self.group,
            *self.kernel_shape,
        )
        output_shape = (self.batch, self.out_channels, *self.out_spatial)
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        in_indices = tuple(f"id{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        bias_suffix = emitter.param_array_suffix((self.out_channels,))
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["weights"], c_type, weight_suffix, True),
                (
                    (params["bias"], c_type, bias_suffix, True)
                    if params["bias"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["conv"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                weights=params["weights"],
                bias=params["bias"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                weight_suffix=weight_suffix,
                bias_suffix=bias_suffix,
                output_suffix=output_suffix,
                batch=self.batch,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                spatial_rank=self.spatial_rank,
                in_spatial=self.in_spatial,
                out_spatial=self.out_spatial,
                kernel_shape=self.kernel_shape,
                strides=self.strides,
                pads_begin=pad_begin,
                dilations=self.dilations,
                group=self.group,
                group_in_channels=group_in_channels,
                group_out_channels=group_out_channels,
                out_indices=out_indices,
                kernel_indices=kernel_indices,
                in_indices=in_indices,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("weights", self.weights),
                ("x_zero_point", self.x_zero_point),
                ("w_zero_point", self.w_zero_point),
                ("output", self.output),
            ]
        )
        acc_dtype = self.dtype
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        input_shape = (self.batch, self.in_channels, *self.in_spatial)
        weight_shape = (
            self.out_channels,
            self.in_channels // self.group,
            *self.kernel_shape,
        )
        output_shape = (self.batch, self.out_channels, *self.out_spatial)
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        in_indices = tuple(f"id{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        x_zero_suffix = (
            emitter.param_array_suffix(self.x_zero_point_shape)
            if self.x_zero_point_shape is not None
            else ""
        )
        w_zero_suffix = (
            emitter.param_array_suffix(self.w_zero_point_shape)
            if self.w_zero_point_shape is not None
            else ""
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    self.input_dtype.c_type,
                    input_suffix,
                    True,
                ),
                (
                    params["weights"],
                    self.weight_dtype.c_type,
                    weight_suffix,
                    True,
                ),
                (
                    (
                        params["x_zero_point"],
                        self.input_dtype.c_type,
                        x_zero_suffix,
                        True,
                    )
                    if params["x_zero_point"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["w_zero_point"],
                        self.weight_dtype.c_type,
                        w_zero_suffix,
                        True,
                    )
                    if params["w_zero_point"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        x_zero_expr = (
            f"{params['x_zero_point']}[0]" if params["x_zero_point"] else "0"
        )
        if params["w_zero_point"]:
            if self.w_zero_point_per_channel:
                w_zero_expr = f"{params['w_zero_point']}[oc_global]"
            else:
                w_zero_expr = f"{params['w_zero_point']}[0]"
        else:
            w_zero_expr = "0"
        rendered = state.templates["conv_integer"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            weights=params["weights"],
            x_zero_point=params["x_zero_point"],
            w_zero_point=params["w_zero_point"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            acc_type=acc_type,
            acc_zero_literal=acc_zero_literal,
            input_suffix=input_suffix,
            weight_suffix=weight_suffix,
            x_zero_suffix=x_zero_suffix,
            w_zero_suffix=w_zero_suffix,
            output_suffix=output_suffix,
            batch=self.batch,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            spatial_rank=self.spatial_rank,
            in_spatial=self.in_spatial,
            out_spatial=self.out_spatial,
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads_begin=pad_begin,
            dilations=self.dilations,
            group=self.group,
            group_in_channels=group_in_channels,
            group_out_channels=group_out_channels,
            out_indices=out_indices,
            kernel_indices=kernel_indices,
            in_indices=in_indices,
            x_zero_expr=x_zero_expr,
            w_zero_expr=w_zero_expr,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input_scale", self.input_scale),
                ("input_zero_point", self.input_zero_point),
                ("weights", self.weights),
                ("weight_scale", self.weight_scale),
                ("weight_zero_point", self.weight_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )
        input_shape = (self.batch, self.in_channels, *self.in_spatial)
        weight_shape = (
            self.out_channels,
            self.in_channels // self.group,
            *self.kernel_shape,
        )
        output_shape = (self.batch, self.out_channels, *self.out_spatial)
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        in_indices = tuple(f"id{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        input_scale_suffix = emitter.param_array_suffix(self.input_scale_shape)
        weight_scale_suffix = emitter.param_array_suffix(self.weight_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input_zero_suffix = emitter.param_array_suffix(self.input_zero_shape)
        weight_zero_suffix = emitter.param_array_suffix(self.weight_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
        bias_suffix = emitter.param_array_suffix((self.out_channels,))
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], self.input_dtype.c_type, input_suffix, True),
                (
                    params["input_scale"],
                    self.input_scale_dtype.c_type,
                    input_scale_suffix,
                    True,
                ),
                (
                    params["input_zero_point"],
                    self.input_dtype.c_type,
                    input_zero_suffix,
                    True,
                ),
                (params["weights"], self.weight_dtype.c_type, weight_suffix, True),
                (
                    params["weight_scale"],
                    self.weight_scale_dtype.c_type,
                    weight_scale_suffix,
                    True,
                ),
                (
                    params["weight_zero_point"],
                    self.weight_dtype.c_type,
                    weight_zero_suffix,
                    True,
                ),
                (
                    params["output_scale"],
                    self.output_scale_dtype.c_type,
                    output_scale_suffix,
                    True,
                ),
                (
                    params["output_zero_point"],
                    self.dtype.c_type,
                    output_zero_suffix,
                    True,
                ),
                (
                    (params["bias"], ScalarType.I32.c_type, bias_suffix, True)
                    if params["bias"]
                    else (None, "", "", True)
                ),
                (params["output"], self.dtype.c_type, output_suffix, False),
            ]
        )

        compute_dtype = (
            ScalarType.F64
            if ScalarType.F64
            in {
                self.input_scale_dtype,
                self.weight_scale_dtype,
                self.output_scale_dtype,
            }
            else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        weight_scale_expr = (
            f"{params['weight_scale']}[oc_global]"
            if self.weight_scale_per_channel
            else f"{params['weight_scale']}[0]"
        )
        weight_zero_expr = (
            f"{params['weight_zero_point']}[oc_global]"
            if self.weight_zero_per_channel
            else f"{params['weight_zero_point']}[0]"
        )
        rendered = state.templates["qlinear_conv"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            input_scale=params["input_scale"],
            input_zero_point=params["input_zero_point"],
            weights=params["weights"],
            weight_scale=params["weight_scale"],
            weight_zero_point=params["weight_zero_point"],
            output_scale=params["output_scale"],
            output_zero_point=params["output_zero_point"],
            bias=params["bias"],
            output=params["output"],
            params=param_decls,
            output_c_type=self.dtype.c_type,
            compute_type=compute_type,
            input_suffix=input_suffix,
            weight_suffix=weight_suffix,
            input_scale_suffix=input_scale_suffix,
            weight_scale_suffix=weight_scale_suffix,
            output_scale_suffix=output_scale_suffix,
            input_zero_suffix=input_zero_suffix,
            weight_zero_suffix=weight_zero_suffix,
            output_zero_suffix=output_zero_suffix,
            bias_suffix=bias_suffix,
            output_suffix=output_suffix,
            batch=self.batch,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            spatial_rank=self.spatial_rank,
            in_spatial=self.in_spatial,
            out_spatial=self.out_spatial,
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads_begin=pad_begin,
            dilations=self.dilations,
            group=self.group,
            group_in_channels=group_in_channels,
            group_out_channels=group_out_channels,
            out_indices=out_indices,
            kernel_indices=kernel_indices,
            in_indices=in_indices,
            input_scale_expr=f"{params['input_scale']}[0]",
            weight_scale_expr=weight_scale_expr,
            output_scale_expr=f"{params['output_scale']}[0]",
            input_zero_expr=f"{params['input_zero_point']}[0]",
            weight_zero_expr=weight_zero_expr,
            output_zero_expr=f"{params['output_zero_point']}[0]",
            min_literal=self.dtype.min_literal,
            max_literal=self.dtype.max_literal,
            round_dtype=ScalarType.F64,
            compute_dtype=compute_dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("weights", self.weights),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )
        input_shape = (self.batch, self.in_channels, *self.in_spatial)
        weight_shape = (
            self.in_channels,
            self.out_channels // self.group,
            *self.kernel_shape,
        )
        output_shape = (self.batch, self.out_channels, *self.out_spatial)
        in_indices = tuple(f"id{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        bias_suffix = emitter.param_array_suffix((self.out_channels,))
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["weights"], c_type, weight_suffix, True),
                (
                    (
                        params["bias"],
                        c_type,
                        bias_suffix,
                        True,
                    )
                    if params["bias"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["conv_transpose"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            weights=params["weights"],
            bias=params["bias"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            zero_literal=zero_literal,
            input_suffix=input_suffix,
            weight_suffix=weight_suffix,
            bias_suffix=bias_suffix,
            output_suffix=output_suffix,
            batch=self.batch,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            spatial_rank=self.spatial_rank,
            in_spatial=self.in_spatial,
            out_spatial=self.out_spatial,
            kernel_shape=self.kernel_shape,
            strides=self.strides,
            pads_begin=pad_begin,
            dilations=self.dilations,
            group=self.group,
            group_in_channels=group_in_channels,
            group_out_channels=group_out_channels,
            in_indices=in_indices,
            kernel_indices=kernel_indices,
            out_indices=out_indices,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class Col2ImOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    batch: int
    channels: int
    spatial_rank: int
    image_shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    col_dims: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    dtype: ScalarType

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output", self.output),
            ]
        )
        kernel_total = 1
        for b in self.block_shape:
            kernel_total *= b
        col_total = 1
        for c in self.col_dims:
            col_total *= c
        input_shape = (self.batch, self.channels * kernel_total, col_total)
        output_shape = (self.batch, self.channels, *self.image_shape)
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        col_indices = tuple(f"cd{dim}" for dim in range(self.spatial_rank))
        im_indices = tuple(f"im{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        kernel_multipliers: list[int] = []
        for dim in range(self.spatial_rank - 1):
            mul = 1
            for d in range(dim + 1, self.spatial_rank):
                mul *= self.block_shape[d]
            kernel_multipliers.append(mul)
        col_multipliers: list[int] = []
        for dim in range(self.spatial_rank - 1):
            mul = 1
            for d in range(dim + 1, self.spatial_rank):
                mul *= self.col_dims[d]
            col_multipliers.append(mul)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["col2im"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            zero_literal=zero_literal,
            batch=self.batch,
            channels=self.channels,
            spatial_rank=self.spatial_rank,
            image_shape=self.image_shape,
            block_shape=self.block_shape,
            col_dims=self.col_dims,
            strides=self.strides,
            pads_begin=pad_begin,
            dilations=self.dilations,
            kernel_total=kernel_total,
            kernel_multipliers=kernel_multipliers,
            col_multipliers=col_multipliers,
            out_indices=out_indices,
            kernel_indices=kernel_indices,
            col_indices=col_indices,
            im_indices=im_indices,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class DeformConvOp(ConvLikeOpBase):
    __io_inputs__ = ("input0", "weights", "offset", "bias", "mask")
    input0: str
    weights: str
    offset: str
    bias: str | None
    mask: str | None
    output: str
    batch: int
    in_channels: int
    out_channels: int
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
    dilation_h: int
    dilation_w: int
    group: int
    offset_group: int
    dtype: ScalarType

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("weights", self.weights),
                ("offset", self.offset),
                ("bias", self.bias),
                ("mask", self.mask),
                ("output", self.output),
            ]
        )
        acc_dtype = emitter.accumulation_dtype(self.dtype)
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        acc_one_literal = emitter.format_literal(acc_dtype, 1)
        input_shape = (self.batch, self.in_channels, self.in_h, self.in_w)
        weight_shape = (
            self.out_channels,
            self.in_channels // self.group,
            self.kernel_h,
            self.kernel_w,
        )
        offset_shape = (
            self.batch,
            self.offset_group * self.kernel_h * self.kernel_w * 2,
            self.out_h,
            self.out_w,
        )
        bias_shape = (self.out_channels,)
        mask_shape = (
            self.batch,
            self.offset_group * self.kernel_h * self.kernel_w,
            self.out_h,
            self.out_w,
        )
        output_shape = (self.batch, self.out_channels, self.out_h, self.out_w)
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        ics_per_offset_group = self.in_channels // self.offset_group
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        offset_suffix = emitter.param_array_suffix(offset_shape)
        bias_suffix = emitter.param_array_suffix(bias_shape)
        mask_suffix = emitter.param_array_suffix(mask_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["weights"], c_type, weight_suffix, True),
                (params["offset"], c_type, offset_suffix, True),
                (
                    (params["bias"], c_type, bias_suffix, True)
                    if params["bias"]
                    else (None, "", "", True)
                ),
                (
                    (params["mask"], c_type, mask_suffix, True)
                    if params["mask"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["deform_conv"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            weights=params["weights"],
            offset=params["offset"],
            bias=params["bias"],
            mask=params["mask"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            acc_type=acc_type,
            acc_zero_literal=acc_zero_literal,
            zero_literal=acc_zero_literal,
            one_literal=acc_one_literal,
            acc_scalar_dtype=acc_dtype,
            batch=self.batch,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            in_h=self.in_h,
            in_w=self.in_w,
            out_h=self.out_h,
            out_w=self.out_w,
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            pad_top=self.pad_top,
            pad_left=self.pad_left,
            dilation_h=self.dilation_h,
            dilation_w=self.dilation_w,
            group=self.group,
            offset_group=self.offset_group,
            group_in_channels=group_in_channels,
            group_out_channels=group_out_channels,
            ics_per_offset_group=ics_per_offset_group,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        if self.spatial_rank == 3:
            input_shape = (
                self.batch,
                self.channels,
                self.in_d,
                self.in_h,
                self.in_w,
            )
            output_shape = (
                self.batch,
                self.channels,
                self.out_d,
                self.out_h,
                self.out_w,
            )
        elif self.spatial_rank == 1:
            input_shape = (self.batch, self.channels, self.in_w)
            output_shape = (self.batch, self.channels, self.out_w)
        else:
            input_shape = (self.batch, self.channels, self.in_h, self.in_w)
            output_shape = (self.batch, self.channels, self.out_h, self.out_w)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["avg_pool"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            zero_literal=zero_literal,
            input_suffix=input_suffix,
            output_suffix=output_suffix,
            batch=self.batch,
            channels=self.channels,
            spatial_rank=self.spatial_rank,
            in_d=self.in_d,
            in_h=self.in_h,
            in_w=self.in_w,
            out_d=self.out_d,
            out_h=self.out_h,
            out_w=self.out_w,
            kernel_d=self.kernel_d,
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            dilation_d=self.dilation_d,
            dilation_h=self.dilation_h,
            dilation_w=self.dilation_w,
            stride_d=self.stride_d,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            pad_front=self.pad_front,
            pad_top=self.pad_top,
            pad_left=self.pad_left,
            pad_back=self.pad_back,
            pad_bottom=self.pad_bottom,
            pad_right=self.pad_right,
            count_include_pad=int(self.count_include_pad),
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = (self.batch, self.channels, self.in_h, self.in_w)
        output_shape = (self.batch, self.channels, self.out_h, self.out_w)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["lp_pool"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            input_suffix=input_suffix,
            output_suffix=output_suffix,
            batch=self.batch,
            channels=self.channels,
            in_h=self.in_h,
            in_w=self.in_w,
            out_h=self.out_h,
            out_w=self.out_w,
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            dilation_h=self.dilation_h,
            dilation_w=self.dilation_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            pad_top=self.pad_top,
            pad_left=self.pad_left,
            pad_bottom=self.pad_bottom,
            pad_right=self.pad_right,
            p=self.p,
            zero_literal=zero_literal,
            dtype=self.dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class SoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_shape = emitter.ctx_shape(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        outer = emitter.derived(self, "outer")
        axis_size = emitter.derived(self, "axis_size")
        inner = emitter.derived(self, "inner")
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], output_dtype.c_type, array_suffix, True),
                (params["output"], output_dtype.c_type, array_suffix, False),
            ]
        )
        rendered = (
            state.templates["softmax"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=output_dtype.c_type,
                array_suffix=array_suffix,
                outer=outer,
                axis_size=axis_size,
                inner=inner,
                dtype=output_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

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
class QLinearSoftmaxOp(RenderableOpBase):
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
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: int
    outer: int
    axis_size: int
    inner: int
    dtype: ScalarType
    input_dtype: ScalarType
    input_scale_dtype: ScalarType
    output_scale_dtype: ScalarType
    input_scale_shape: tuple[int, ...]
    output_scale_shape: tuple[int, ...]
    input_zero_shape: tuple[int, ...]
    output_zero_shape: tuple[int, ...]

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            return None
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                "QLinearSoftmax expects output dtype to match input dtype"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        try:
            output_shape = ctx.shape(self.output)
        except ShapeInferenceError:
            ctx.set_shape(self.output, input_shape)
            return None
        if output_shape != input_shape:
            raise ShapeInferenceError(
                f"QLinearSoftmax output shape must be {input_shape}, got {output_shape}"
            )


@dataclass(frozen=True)
class LogSoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_shape = emitter.ctx_shape(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        outer = emitter.derived(self, "outer")
        axis_size = emitter.derived(self, "axis_size")
        inner = emitter.derived(self, "inner")
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], output_dtype.c_type, array_suffix, True),
                (params["output"], output_dtype.c_type, array_suffix, False),
            ]
        )
        rendered = (
            state.templates["logsoftmax"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=output_dtype.c_type,
                array_suffix=array_suffix,
                outer=outer,
                axis_size=axis_size,
                inner=inner,
                dtype=output_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_shape = emitter.ctx_shape(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        zero_literal = output_dtype.zero_literal
        outer = emitter.derived(self, "outer")
        axis_size = emitter.derived(self, "axis_size")
        inner = emitter.derived(self, "inner")
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], output_dtype.c_type, array_suffix, True),
                (params["output"], output_dtype.c_type, array_suffix, False),
            ]
        )
        rendered = (
            state.templates["hardmax"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=output_dtype.c_type,
                array_suffix=array_suffix,
                outer=outer,
                axis_size=axis_size,
                inner=inner,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(output_dtype, 1),
                dtype=output_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("bias", self.bias),
                ("mean", self.mean),
                ("variance", self.variance),
                ("output", self.output),
                ("running_mean", self.running_mean),
                ("running_variance", self.running_variance),
            ]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        scale_suffix = emitter.param_array_suffix((self.channels,))
        bias_suffix = emitter.param_array_suffix((self.channels,))
        mean_suffix = emitter.param_array_suffix((self.channels,))
        variance_suffix = emitter.param_array_suffix((self.channels,))
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["scale"], c_type, scale_suffix, True),
                (params["bias"], c_type, bias_suffix, True),
                (params["mean"], c_type, mean_suffix, True),
                (params["variance"], c_type, variance_suffix, True),
                (params["output"], c_type, output_suffix, False),
                (params.get("running_mean"), c_type, mean_suffix, False),
                (params.get("running_variance"), c_type, variance_suffix, False),
            ]
        )
        reduce_count_expr = " * ".join(
            str(dim) for idx, dim in enumerate(shape) if idx != 1
        )
        rendered = state.templates["batch_norm"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            scale=params["scale"],
            bias=params["bias"],
            mean=params["mean"],
            variance=params["variance"],
            output=params["output"],
            running_mean=params.get("running_mean"),
            running_variance=params.get("running_variance"),
            params=param_decls,
            c_type=c_type,
            input_suffix=input_suffix,
            output_suffix=output_suffix,
            scale_suffix=scale_suffix,
            bias_suffix=bias_suffix,
            mean_suffix=mean_suffix,
            variance_suffix=variance_suffix,
            shape=shape,
            loop_vars=loop_vars,
            channels=self.channels,
            reduce_count_expr=reduce_count_expr or "1",
            training_mode=self.training_mode,
            momentum_literal=emitter.format_floating(self.momentum, self.dtype),
            one_minus_momentum_literal=emitter.format_floating(
                1.0 - self.momentum, self.dtype
            ),
            epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
            dtype=self.dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        array_suffix = emitter.param_array_suffix(shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, array_suffix, True),
                (params["output"], c_type, array_suffix, False),
            ]
        )
        rendered = state.templates["lp_norm"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            array_suffix=array_suffix,
            outer=self.outer,
            axis_size=self.axis_size,
            inner=self.inner,
            p=self.p,
            zero_literal=zero_literal,
            dtype=self.dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        scale_suffix = emitter.param_array_suffix((self.channels,))
        bias_suffix = emitter.param_array_suffix((self.channels,))
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["scale"], c_type, scale_suffix, True),
                (params["bias"], c_type, bias_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["instance_norm"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            scale=params["scale"],
            bias=params["bias"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            zero_literal=zero_literal,
            input_suffix=input_suffix,
            output_suffix=output_suffix,
            scale_suffix=scale_suffix,
            bias_suffix=bias_suffix,
            shape=shape,
            loop_vars=loop_vars,
            spatial_size=self.spatial_size,
            epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
            dtype=self.dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        scale_suffix = emitter.param_array_suffix((self.channels,))
        bias_suffix = emitter.param_array_suffix((self.channels,))
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["scale"], c_type, scale_suffix, True),
                (params["bias"], c_type, bias_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["group_norm"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            scale=params["scale"],
            bias=params["bias"],
            output=params["output"],
            params=param_decls,
            c_type=c_type,
            zero_literal=zero_literal,
            input_suffix=input_suffix,
            output_suffix=output_suffix,
            scale_suffix=scale_suffix,
            bias_suffix=bias_suffix,
            shape=shape,
            loop_vars=loop_vars,
            num_groups=self.num_groups,
            group_size=self.group_size,
            spatial_size=self.spatial_size,
            epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
            dtype=self.dtype,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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
class MomentumOp(RenderableOpBase):
    __io_inputs__ = ("rate", "timestep", "inputs", "gradients", "velocities")
    __io_outputs__ = ("outputs", "velocity_outputs")
    rate: str
    timestep: str
    inputs: tuple[str, ...]
    gradients: tuple[str, ...]
    velocities: tuple[str, ...]
    outputs: tuple[str, ...]
    velocity_outputs: tuple[str, ...]
    rate_shape: tuple[int, ...]
    timestep_shape: tuple[int, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtype: ScalarType
    rate_dtype: ScalarType
    timestep_dtype: ScalarType
    norm_coefficient: float
    alpha: float
    beta: float
    mode: str

    def call_args(self) -> tuple[str, ...]:
        args = [self.rate, self.timestep]
        for index in range(len(self.inputs)):
            args.extend(
                [
                    self.inputs[index],
                    self.gradients[index],
                    self.velocities[index],
                    self.outputs[index],
                    self.velocity_outputs[index],
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
