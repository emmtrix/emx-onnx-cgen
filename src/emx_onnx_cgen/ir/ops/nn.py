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


def _shape_product_expr(
    shape: tuple[int, ...],
    dim_names: dict[int, str] | None = None,
) -> int | str:
    if not shape:
        return 1
    dim_exprs = CEmitterCompat.shape_dim_exprs(shape, dim_names)
    if len(dim_exprs) == 1:
        return dim_exprs[0]
    return " * ".join(str(dim) for dim in dim_exprs)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dtype = emitter.op_output_dtype(self)
        c_type = dtype.c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("output", self.output),
            ]
        )
        input0_shape = emitter.ctx_shape(self.input0)
        input1_shape = emitter.ctx_shape(self.input1)
        input0_dim_names = emitter.dim_names_for(self.input0)
        input1_dim_names = emitter.dim_names_for(self.input1)
        output_shape_raw = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        batch_shape = emitter.derived(self, "batch_shape")
        batch_rank = len(batch_shape)
        batch_vars = output_loop_vars[:batch_rank]
        left_vector = bool(emitter.derived(self, "left_vector"))
        right_vector = bool(emitter.derived(self, "right_vector"))
        if left_vector and right_vector:
            row_var = None
            col_var = None
        elif left_vector:
            row_var = None
            col_var = output_loop_vars[-1]
        elif right_vector:
            row_var = output_loop_vars[-1]
            col_var = None
        else:
            row_var = output_loop_vars[-2]
            col_var = output_loop_vars[-1]
        input0_batch_shape = emitter.derived(self, "input0_batch_shape")
        input1_batch_shape = emitter.derived(self, "input1_batch_shape")
        input0_index_expr, input1_index_expr = CEmitterCompat.matmul_index_exprs(
            batch_vars,
            row_var,
            col_var,
            batch_rank,
            input0=params["input0"],
            input1=params["input1"],
            left_vector=left_vector,
            right_vector=right_vector,
            input0_shape=input0_shape,
            input1_shape=input1_shape,
            input0_batch_shape=input0_batch_shape,
            input1_batch_shape=input1_batch_shape,
        )
        input0_suffix = emitter.param_array_suffix(input0_shape, input0_dim_names)
        input1_suffix = emitter.param_array_suffix(input1_shape, input1_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        acc_dtype = emitter.accumulation_dtype(emitter.ctx_dtype(self.output))
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input0_suffix, True),
                (params["input1"], c_type, input1_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        m = int(emitter.derived(self, "m"))
        n = int(emitter.derived(self, "n"))
        k = int(emitter.derived(self, "k"))
        rendered = (
            state.templates["matmul"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input1=params["input1"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                acc_type=acc_dtype.c_type,
                zero_literal=acc_zero_literal,
                input0_suffix=input0_suffix,
                input1_suffix=input1_suffix,
                output_suffix=output_suffix,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                m=m,
                n=n,
                k=k,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class FusedMatMulOp(MatMulLikeOpBase):
    input0: str
    input1: str
    output: str
    alpha: float
    trans_a: bool
    trans_b: bool
    trans_batch_a: bool
    trans_batch_b: bool

    @staticmethod
    def _effective_shape(
        shape: tuple[int, ...], trans: bool, trans_batch: bool
    ) -> tuple[int, ...]:
        """Return the effective shape after batch and matrix transpositions."""
        if len(shape) < 2:
            return shape
        batch = shape[:-2]
        mat = shape[-2:]
        if trans_batch and len(batch) > 1:
            batch = tuple(reversed(batch))
        if trans:
            mat = (mat[1], mat[0])
        return batch + mat

    def infer_types(self, ctx: OpContext) -> None:
        input0_dtype = ctx.dtype(self.input0)
        input1_dtype = ctx.dtype(self.input1)
        if input0_dtype != input1_dtype:
            raise UnsupportedOpError(
                "FusedMatMul expects matching input dtypes, "
                f"got {input0_dtype.onnx_name} and {input1_dtype.onnx_name}"
            )
        if not input0_dtype.is_float:
            raise UnsupportedOpError(
                f"FusedMatMul supports float types only, got {input0_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input0_dtype)
            output_dtype = input0_dtype
        if output_dtype != input0_dtype:
            raise UnsupportedOpError(
                "FusedMatMul expects output dtype to match inputs, "
                f"got {output_dtype.onnx_name} and {input0_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input0_shape = ctx.shape(self.input0)
        input1_shape = ctx.shape(self.input1)

        eff0 = self._effective_shape(input0_shape, self.trans_a, self.trans_batch_a)
        eff1 = self._effective_shape(input1_shape, self.trans_b, self.trans_batch_b)

        left_vector = len(eff0) == 1
        right_vector = len(eff1) == 1
        eff0_expanded = (1, eff0[0]) if left_vector else eff0
        eff1_expanded = (eff1[0], 1) if right_vector else eff1

        m, k_left = eff0_expanded[-2], eff0_expanded[-1]
        k_right, n = eff1_expanded[-2], eff1_expanded[-1]

        if k_left != k_right:
            raise ShapeInferenceError(
                f"FusedMatMul inner dimensions must match, got {k_left} and {k_right}"
            )

        batch_shape, input0_batch_shape, input1_batch_shape = _broadcast_batch_shapes(
            eff0_expanded[:-2], eff1_expanded[:-2]
        )

        if left_vector and right_vector:
            output_shape = batch_shape
        elif left_vector:
            output_shape = batch_shape + (n,)
        elif right_vector:
            output_shape = batch_shape + (m,)
        else:
            output_shape = batch_shape + (m, n)

        try:
            expected = ctx.shape(self.output)
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"FusedMatMul output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self.output, output_shape)
        ctx.set_derived(self, "batch_shape", batch_shape)
        ctx.set_derived(self, "input0_batch_shape", input0_batch_shape)
        ctx.set_derived(self, "input1_batch_shape", input1_batch_shape)
        ctx.set_derived(self, "m", m)
        ctx.set_derived(self, "n", n)
        ctx.set_derived(self, "k", k_left)
        ctx.set_derived(self, "left_vector", left_vector)
        ctx.set_derived(self, "right_vector", right_vector)

    def _fused_matmul_index_exprs(
        self,
        batch_vars: tuple[str, ...],
        row_var: str | None,
        col_var: str | None,
        batch_rank: int,
        *,
        input0: str,
        input1: str,
        left_vector: bool,
        right_vector: bool,
        input0_shape: tuple[int, ...],
        input1_shape: tuple[int, ...],
        input0_batch_shape: tuple[int, ...],
        input1_batch_shape: tuple[int, ...],
    ) -> tuple[str, str]:
        def batch_indices(
            batch_shape: tuple[int, ...],
            actual_rank: int,
            trans_batch: bool,
        ) -> list[str]:
            if actual_rank == 0:
                return []
            offset = batch_rank - actual_rank
            indices: list[str] = []
            for idx in range(actual_rank):
                dim = batch_shape[offset + idx]
                var = batch_vars[offset + idx]
                indices.append("0" if dim == 1 else var)
            if trans_batch and len(indices) > 1:
                indices = list(reversed(indices))
            return indices

        if left_vector:
            input0_indices: list[str] = ["k"]
        else:
            input0_batch_rank = len(input0_shape) - 2
            input0_bi = batch_indices(
                input0_batch_shape, input0_batch_rank, self.trans_batch_a
            )
            if self.trans_a:
                input0_indices = [
                    *input0_bi,
                    "k",
                    row_var if row_var is not None else "0",
                ]
            else:
                input0_indices = [
                    *input0_bi,
                    row_var if row_var is not None else "0",
                    "k",
                ]

        if right_vector:
            input1_indices: list[str] = ["k"]
        else:
            input1_batch_rank = len(input1_shape) - 2
            input1_bi = batch_indices(
                input1_batch_shape, input1_batch_rank, self.trans_batch_b
            )
            if self.trans_b:
                input1_indices = [
                    *input1_bi,
                    col_var if col_var is not None else "0",
                    "k",
                ]
            else:
                input1_indices = [
                    *input1_bi,
                    "k",
                    col_var if col_var is not None else "0",
                ]

        input0_index_expr = f"{input0}" + "".join(f"[{i}]" for i in input0_indices)
        input1_index_expr = f"{input1}" + "".join(f"[{i}]" for i in input1_indices)
        return input0_index_expr, input1_index_expr

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dtype = emitter.op_output_dtype(self)
        c_type = dtype.c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("output", self.output),
            ]
        )
        input0_shape = emitter.ctx_shape(self.input0)
        input1_shape = emitter.ctx_shape(self.input1)
        input0_dim_names = emitter.dim_names_for(self.input0)
        input1_dim_names = emitter.dim_names_for(self.input1)
        output_shape_raw = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        batch_shape = emitter.derived(self, "batch_shape")
        batch_rank = len(batch_shape)
        batch_vars = output_loop_vars[:batch_rank]
        left_vector = bool(emitter.derived(self, "left_vector"))
        right_vector = bool(emitter.derived(self, "right_vector"))
        if left_vector and right_vector:
            row_var = None
            col_var = None
        elif left_vector:
            row_var = None
            col_var = output_loop_vars[-1]
        elif right_vector:
            row_var = output_loop_vars[-1]
            col_var = None
        else:
            row_var = output_loop_vars[-2]
            col_var = output_loop_vars[-1]
        input0_batch_shape = emitter.derived(self, "input0_batch_shape")
        input1_batch_shape = emitter.derived(self, "input1_batch_shape")
        input0_index_expr, input1_index_expr = self._fused_matmul_index_exprs(
            batch_vars,
            row_var,
            col_var,
            batch_rank,
            input0=params["input0"],
            input1=params["input1"],
            left_vector=left_vector,
            right_vector=right_vector,
            input0_shape=input0_shape,
            input1_shape=input1_shape,
            input0_batch_shape=input0_batch_shape,
            input1_batch_shape=input1_batch_shape,
        )
        input0_suffix = emitter.param_array_suffix(input0_shape, input0_dim_names)
        input1_suffix = emitter.param_array_suffix(input1_shape, input1_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        acc_dtype = emitter.accumulation_dtype(emitter.ctx_dtype(self.output))
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        has_alpha = self.alpha != 1.0
        alpha_literal = emitter.format_literal(dtype, self.alpha) if has_alpha else None
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input0_suffix, True),
                (params["input1"], c_type, input1_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        m = int(emitter.derived(self, "m"))
        n = int(emitter.derived(self, "n"))
        k = int(emitter.derived(self, "k"))
        rendered = (
            state.templates["fused_matmul"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                c_type=c_type,
                acc_type=acc_dtype.c_type,
                zero_literal=acc_zero_literal,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                m=m,
                n=n,
                k=k,
                has_alpha=has_alpha,
                alpha_literal=alpha_literal,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = {"#include <math.h>"}
        if ctx.dtype(self.output).is_integer:
            includes.add("#include <limits.h>")
        return includes

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input0_scale", self.input0_scale),
                ("input0_zero_point", self.input0_zero_point),
                ("input1", self.input1),
                ("input1_scale", self.input1_scale),
                ("input1_zero_point", self.input1_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("output", self.output),
            ]
        )
        output_shape = CEmitterCompat.codegen_shape(self.output_shape)
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        batch_rank = len(self.batch_shape)
        batch_vars = output_loop_vars[:batch_rank]
        if self.left_vector and self.right_vector:
            row_var = None
            col_var = None
        elif self.left_vector:
            row_var = None
            col_var = output_loop_vars[-1]
        elif self.right_vector:
            row_var = output_loop_vars[-1]
            col_var = None
        else:
            row_var = output_loop_vars[-2]
            col_var = output_loop_vars[-1]
        input0_index_expr, input1_index_expr = CEmitterCompat.matmul_index_exprs(
            batch_vars,
            row_var,
            col_var,
            batch_rank,
            input0=params["input0"],
            input1=params["input1"],
            left_vector=self.left_vector,
            right_vector=self.right_vector,
            input0_shape=self.input0_shape,
            input1_shape=self.input1_shape,
            input0_batch_shape=self.input0_batch_shape,
            input1_batch_shape=self.input1_batch_shape,
        )
        input0_suffix = emitter.param_array_suffix(self.input0_shape)
        input1_suffix = emitter.param_array_suffix(self.input1_shape)
        input0_scale_suffix = emitter.param_array_suffix(self.input0_scale_shape)
        input1_scale_suffix = emitter.param_array_suffix(self.input1_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input0_zero_suffix = emitter.param_array_suffix(self.input0_zero_shape)
        input1_zero_suffix = emitter.param_array_suffix(self.input1_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    self.input0_dtype.c_type,
                    input0_suffix,
                    True,
                ),
                (
                    params["input0_scale"],
                    self.input0_scale_dtype.c_type,
                    input0_scale_suffix,
                    True,
                ),
                (
                    params["input0_zero_point"],
                    self.input0_dtype.c_type,
                    input0_zero_suffix,
                    True,
                ),
                (
                    params["input1"],
                    self.input1_dtype.c_type,
                    input1_suffix,
                    True,
                ),
                (
                    params["input1_scale"],
                    self.input1_scale_dtype.c_type,
                    input1_scale_suffix,
                    True,
                ),
                (
                    params["input1_zero_point"],
                    self.input1_dtype.c_type,
                    input1_zero_suffix,
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
                    params["output"],
                    self.dtype.c_type,
                    output_suffix,
                    False,
                ),
            ]
        )
        if ScalarType.F64 in {
            self.input0_scale_dtype,
            self.input1_scale_dtype,
            self.output_scale_dtype,
        }:
            scale_dtype = ScalarType.F64
        elif ScalarType.F32 in {
            self.input0_scale_dtype,
            self.input1_scale_dtype,
            self.output_scale_dtype,
        }:
            scale_dtype = ScalarType.F32
        else:
            scale_dtype = ScalarType.F16
        compute_dtype = ScalarType.F64
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"
        scale_index = "0"
        if self.dtype.is_signed:
            min_literal = "-128.0"
            max_literal = "127.0"
        else:
            min_literal = "0.0"
            max_literal = "255.0"
        rendered = (
            state.templates["qlinear_matmul"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input1=params["input1"],
                input0_scale=params["input0_scale"],
                input0_zero_point=params["input0_zero_point"],
                input1_scale=params["input1_scale"],
                input1_zero_point=params["input1_zero_point"],
                output_scale=params["output_scale"],
                output_zero_point=params["output_zero_point"],
                output=params["output"],
                params=param_decls,
                scale_type=scale_dtype.c_type,
                scale_is_float16=scale_dtype == ScalarType.F16,
                compute_type=compute_type,
                output_c_type=self.dtype.c_type,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                input0_scale_expr=f"{params['input0_scale']}[{scale_index}]",
                input1_scale_expr=f"{params['input1_scale']}[{scale_index}]",
                output_scale_expr=f"{params['output_scale']}[{scale_index}]",
                input0_zero_expr=f"{params['input0_zero_point']}[{scale_index}]",
                input1_zero_expr=f"{params['input1_zero_point']}[{scale_index}]",
                output_zero_expr=f"{params['output_zero_point']}[{scale_index}]",
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                k=self.k,
                compute_dtype=compute_dtype,
                dtype=self.dtype,
                min_literal=min_literal,
                max_literal=max_literal,
                enable_integer_requant=scale_dtype != ScalarType.F16,
                output_wrap=not emitter.replicate_ort_bugs,
                output_is_signed=self.dtype.is_signed,
                dim_args=emitter.dim_args_str(),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("input0_zero_point", self.input0_zero_point),
                ("input1_zero_point", self.input1_zero_point),
                ("output", self.output),
            ]
        )
        output_shape = CEmitterCompat.codegen_shape(self.output_shape)
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        batch_rank = len(self.batch_shape)
        batch_vars = output_loop_vars[:batch_rank]
        if self.left_vector and self.right_vector:
            row_var = None
            col_var = None
        elif self.left_vector:
            row_var = None
            col_var = output_loop_vars[-1]
        elif self.right_vector:
            row_var = output_loop_vars[-1]
            col_var = None
        else:
            row_var = output_loop_vars[-2]
            col_var = output_loop_vars[-1]
        input0_index_expr, input1_index_expr = CEmitterCompat.matmul_index_exprs(
            batch_vars,
            row_var,
            col_var,
            batch_rank,
            input0=params["input0"],
            input1=params["input1"],
            left_vector=self.left_vector,
            right_vector=self.right_vector,
            input0_shape=self.input0_shape,
            input1_shape=self.input1_shape,
            input0_batch_shape=self.input0_batch_shape,
            input1_batch_shape=self.input1_batch_shape,
        )
        input0_suffix = emitter.param_array_suffix(self.input0_shape)
        input1_suffix = emitter.param_array_suffix(self.input1_shape)
        input0_zero_suffix = emitter.param_array_suffix(self.input0_zero_shape or ())
        input1_zero_suffix = emitter.param_array_suffix(self.input1_zero_shape or ())
        output_suffix = emitter.param_array_suffix(self.output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], self.input0_dtype.c_type, input0_suffix, True),
                (params["input1"], self.input1_dtype.c_type, input1_suffix, True),
                (
                    (
                        params["input0_zero_point"],
                        self.input0_dtype.c_type,
                        input0_zero_suffix,
                        True,
                    )
                    if params["input0_zero_point"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input1_zero_point"],
                        self.input1_dtype.c_type,
                        input1_zero_suffix,
                        True,
                    )
                    if params["input1_zero_point"]
                    else (None, "", "", True)
                ),
                (params["output"], self.dtype.c_type, output_suffix, False),
            ]
        )
        input0_zero_expr = (
            f"{params['input0_zero_point']}[0]" if params["input0_zero_point"] else "0"
        )
        input1_zero_expr = (
            f"{params['input1_zero_point']}[0]" if params["input1_zero_point"] else "0"
        )
        rendered = (
            state.templates["matmul_integer"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input1=params["input1"],
                output=params["output"],
                params=param_decls,
                input0_c_type=self.input0_dtype.c_type,
                input1_c_type=self.input1_dtype.c_type,
                output_c_type=self.dtype.c_type,
                input0_zero_expr=input0_zero_expr,
                input1_zero_expr=input1_zero_expr,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                k=self.k,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape


@dataclass(frozen=True)
class MatMulNBitsOp(RenderableOpBase):
    """N-bit block-quantized matrix multiplication (com.microsoft contrib op).

    Computes ``Y = A @ dequantize(B) [+ bias]`` where B is packed N-bit
    unsigned integers with per-block scales and optional zero-points.
    """

    __io_inputs__ = ("input0", "input1", "scales", "zero_points", "bias")
    __io_outputs__ = ("output",)

    input0: str
    input1: str
    scales: str
    zero_points: str | None
    bias: str | None
    output: str

    bits: int
    block_size: int
    k: int
    n: int
    accuracy_level: int
    n_blocks_per_col: int
    blob_size: int

    input0_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    m: int

    input0_dtype: ScalarType
    output_dtype: ScalarType
    b_dtype: ScalarType
    scales_dtype: ScalarType
    zero_points_dtype: ScalarType | None
    zero_points_packed: bool
    bias_dtype: ScalarType | None

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()

        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("scales", self.scales),
                ("zero_points", self.zero_points),
                ("bias", self.bias),
                ("output", self.output),
            ]
        )

        output_shape_raw = CEmitterCompat.codegen_shape(self.output_shape)
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            self.output_shape, output_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )

        batch_rank = len(self.batch_shape)
        batch_vars = output_loop_vars[:batch_rank]
        m_var = output_loop_vars[-2] if len(output_loop_vars) >= 2 else "i0"
        n_var = output_loop_vars[-1] if output_loop_vars else "i0"

        input0_index_parts = list(batch_vars) + [m_var, "k"]
        input0_index_expr = f"{params['input0']}" + "".join(
            f"[{var}]" for var in input0_index_parts
        )

        input0_suffix = emitter.param_array_suffix(
            self.input0_shape, emitter.dim_names_for(self.input0)
        )

        b_shape = (self.n, self.n_blocks_per_col, self.blob_size)
        b_suffix = emitter.param_array_suffix(b_shape)

        scales_shape = (self.n, self.n_blocks_per_col)
        scales_suffix = emitter.param_array_suffix(scales_shape)

        acc_dtype = emitter.accumulation_dtype(self.output_dtype)
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)

        b_c_type = self.b_dtype.c_type

        param_entries: list[tuple[str | None, str, str, bool] | tuple] = [
            (params["input0"], self.input0_dtype.c_type, input0_suffix, True),
            (params["input1"], b_c_type, b_suffix, True),
            (params["scales"], self.scales_dtype.c_type, scales_suffix, True),
        ]

        zp_suffix = ""
        if params["zero_points"]:
            zp_shape_raw = emitter.ctx_shape(self.zero_points)  # type: ignore[arg-type]
            zp_suffix = emitter.param_array_suffix(zp_shape_raw)
            zp_c_type = (
                self.zero_points_dtype.c_type if self.zero_points_dtype else "uint8_t"
            )
            param_entries.append((params["zero_points"], zp_c_type, zp_suffix, True))
        else:
            param_entries.append((None, "", "", True))

        if params["bias"]:
            bias_shape = (self.n,)
            bias_suffix = emitter.param_array_suffix(bias_shape)
            bias_c_type = (
                self.bias_dtype.c_type if self.bias_dtype else self.output_dtype.c_type
            )
            param_entries.append((params["bias"], bias_c_type, bias_suffix, True))
        else:
            param_entries.append((None, "", "", True))

        output_suffix = emitter.param_array_suffix(self.output_shape, output_dim_names)
        param_entries.append(
            (params["output"], self.output_dtype.c_type, output_suffix, False)
        )

        param_decls = emitter.build_param_decls(param_entries)

        bit_mask = (1 << self.bits) - 1
        default_zero_point = 1 << (self.bits - 1)

        rendered = (
            state.templates["matmul_nbits"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                acc_type=acc_dtype.c_type,
                acc_zero=acc_zero_literal,
                output_c_type=self.output_dtype.c_type,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1=params["input1"],
                scales=params["scales"],
                zero_points=params["zero_points"],
                bias=params["bias"],
                n_var=n_var,
                k=self.k,
                bits=self.bits,
                block_size=self.block_size,
                bit_mask=bit_mask,
                default_zero_point=default_zero_point,
                has_zero_points=params["zero_points"] is not None,
                zero_points_packed=self.zero_points_packed,
                has_bias=params["bias"] is not None,
                b_c_type=b_c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, emitter.ctx_shape(self.input0)),
            (self.input1, (self.n, self.n_blocks_per_col, self.blob_size)),
            (self.scales, (self.n, self.n_blocks_per_col)),
        ]
        if self.zero_points is not None:
            inputs.append((self.zero_points, emitter.ctx_shape(self.zero_points)))
        if self.bias is not None:
            inputs.append((self.bias, (self.n,)))
        return tuple(inputs)


# ---------------------------------------------------------------------------
# FP4 / NF4 lookup tables used by bitsandbytes 4-bit quantisation.
# These must match ORT's ``blockwise_quant_block_bnb4.h``.
# ---------------------------------------------------------------------------
_FP4_DEQUANT_TABLE: tuple[float, ...] = (
    0.0,
    5.208333333e-03,
    0.66666667,
    1.0,
    0.33333333,
    0.50000000,
    0.16666667,
    0.25000000,
    -0.0,
    -5.208333333e-03,
    -0.66666667,
    -1.0,
    -0.33333333,
    -0.50000000,
    -0.16666667,
    -0.25000000,
)

_NF4_DEQUANT_TABLE: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


@dataclass(frozen=True)
class MatMulBnb4Op(RenderableOpBase):
    """BitsAndBytes 4-bit block-quantized matrix multiplication (com.microsoft).

    Computes ``Y = A @ dequantize(B)`` where B is a flat packed uint8 array
    of 4-bit FP4 or NF4 values, dequantised using a per-block ``absmax``
    scale factor.  Two 4-bit values are packed per byte (high-nibble first).
    The weight matrix is the transposed original ``[K, N]`` weight stored as
    ``[N, K]`` flattened and blockwise-quantised.
    """

    __io_inputs__ = ("input0", "input1", "absmax")
    __io_outputs__ = ("output",)

    input0: str
    input1: str
    absmax: str
    output: str

    k: int
    n: int
    block_size: int
    quant_type: int  # 0 = FP4, 1 = NF4

    input0_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    m: int

    input0_dtype: ScalarType
    output_dtype: ScalarType
    b_dtype: ScalarType
    absmax_dtype: ScalarType

    n_blocks: int
    packed_size: int

    def _dequant_table(self) -> tuple[float, ...]:
        return _NF4_DEQUANT_TABLE if self.quant_type == 1 else _FP4_DEQUANT_TABLE

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()

        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("absmax", self.absmax),
                ("output", self.output),
            ]
        )

        # Output loop structure  (..., M, N)
        output_shape_raw = CEmitterCompat.codegen_shape(self.output_shape)
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            self.output_shape, output_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )

        n_var = output_loop_vars[-1] if output_loop_vars else "i0"

        # Input A indexing  (..., m, k)
        input0_index_parts = list(output_loop_vars[:-1]) + ["k"]
        input0_index_expr = f"{params['input0']}" + "".join(
            f"[{var}]" for var in input0_index_parts
        )

        input0_suffix = emitter.param_array_suffix(
            self.input0_shape, emitter.dim_names_for(self.input0)
        )

        # B is flat packed: shape (packed_size,)
        b_suffix = emitter.param_array_suffix((self.packed_size,))

        # absmax is flat: shape (n_blocks,)
        absmax_suffix = emitter.param_array_suffix((self.n_blocks,))

        acc_dtype = emitter.accumulation_dtype(self.output_dtype)
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)

        param_entries: list[tuple[str | None, str, str, bool] | tuple] = [
            (params["input0"], self.input0_dtype.c_type, input0_suffix, True),
            (params["input1"], self.b_dtype.c_type, b_suffix, True),
            (params["absmax"], self.absmax_dtype.c_type, absmax_suffix, True),
            (
                params["output"],
                self.output_dtype.c_type,
                emitter.param_array_suffix(self.output_shape, output_dim_names),
                False,
            ),
        ]

        param_decls = emitter.build_param_decls(param_entries)

        # Format lookup table values as C float literals
        table = self._dequant_table()
        table_strs = [f"{v:.17e}f" for v in table]

        rendered = (
            state.templates["matmul_bnb4"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                acc_type=acc_dtype.c_type,
                acc_zero=acc_zero_literal,
                output_c_type=self.output_dtype.c_type,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1=params["input1"],
                absmax=params["absmax"],
                n_var=n_var,
                k=self.k,
                block_size=self.block_size,
                dequant_table=table_strs,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (self.input0, emitter.ctx_shape(self.input0)),
            (self.input1, (self.packed_size,)),
            (self.absmax, (self.n_blocks,)),
        )


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                *((f"input{idx}", name) for idx, name in enumerate(self.inputs)),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            self.output_shape, output_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(self.output_shape)
        if self.output_shape:
            output_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in output_loop_vars
            )
        else:
            output_expr = f"{params['output']}[0]"
        input_shapes = self.input_shapes
        input_dim_names = [emitter.dim_names_for(name) for name in self.inputs]
        input_suffixes = [
            emitter.param_array_suffix(shape, dim_names)
            for shape, dim_names in zip(input_shapes, input_dim_names)
        ]
        param_decls = emitter.build_param_decls(
            [
                *(
                    (
                        params[f"input{idx}"],
                        self.input_dtype.c_type,
                        input_suffixes[idx],
                        True,
                    )
                    for idx in range(len(self.inputs))
                ),
                (
                    params["output"],
                    self.dtype.c_type,
                    emitter.param_array_suffix(self.output_shape, output_dim_names),
                    False,
                ),
            ]
        )
        acc_dtype = emitter.accumulation_dtype(emitter.ctx_dtype(self.output))
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        input_loop_vars: tuple[str, ...] = ()
        input_loop_bounds: tuple[str | int, ...] = ()
        reduce_loop_var = "k"
        reduce_loop_bound: str | int | None = None
        input_expr = None
        input0_expr = None
        input1_expr = None
        if self.kind == EinsumKind.REDUCE_ALL:
            input_loop_vars = CEmitterCompat.loop_vars(input_shapes[0])
            input_loop_bounds = tuple(
                CEmitterCompat.shape_dim_exprs(input_shapes[0], input_dim_names[0])
            )
            if input_loop_vars:
                input_expr = f"{params['input0']}" + "".join(
                    f"[{var}]" for var in input_loop_vars
                )
            else:
                input_expr = f"{params['input0']}[0]"
        elif self.kind == EinsumKind.SUM_J:
            input_shape_exprs = CEmitterCompat.shape_dim_exprs(
                input_shapes[0], input_dim_names[0]
            )
            reduce_loop_bound = input_shape_exprs[1]
            input_expr = f"{params['input0']}[{output_loop_vars[0]}][{reduce_loop_var}]"
        elif self.kind == EinsumKind.TRANSPOSE:
            input_expr = (
                f"{params['input0']}[{output_loop_vars[1]}][{output_loop_vars[0]}]"
            )
        elif self.kind == EinsumKind.DOT:
            input_shape_exprs = CEmitterCompat.shape_dim_exprs(
                input_shapes[0], input_dim_names[0]
            )
            reduce_loop_bound = input_shape_exprs[0]
            input0_expr = f"{params['input0']}[{reduce_loop_var}]"
            input1_expr = f"{params['input1']}[{reduce_loop_var}]"
        elif self.kind == EinsumKind.BATCH_MATMUL:
            input_shape_exprs = CEmitterCompat.shape_dim_exprs(
                input_shapes[0], input_dim_names[0]
            )
            reduce_loop_bound = input_shape_exprs[2]
            input0_expr = (
                f"{params['input0']}"
                f"[{output_loop_vars[0]}]"
                f"[{output_loop_vars[1]}][{reduce_loop_var}]"
            )
            input1_expr = (
                f"{params['input1']}"
                f"[{output_loop_vars[0]}]"
                f"[{reduce_loop_var}][{output_loop_vars[2]}]"
            )
        elif self.kind == EinsumKind.BATCH_DIAGONAL:
            diag_var = output_loop_vars[-1]
            prefix_vars = output_loop_vars[:-1]
            input_expr = f"{params['input0']}" + "".join(
                f"[{var}]" for var in prefix_vars
            )
            input_expr += f"[{diag_var}][{diag_var}]"
        rendered = (
            state.templates["einsum"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                kind=self.kind.value,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_expr=output_expr,
                acc_type=acc_dtype.c_type,
                zero_literal=acc_zero_literal,
                input_loop_vars=input_loop_vars,
                input_loop_bounds=input_loop_bounds,
                reduce_loop_var=reduce_loop_var,
                reduce_loop_bound=reduce_loop_bound,
                input_expr=input_expr,
                input0_expr=input0_expr,
                input1_expr=input1_expr,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (name, shape) for name, shape in zip(self.inputs, self.input_shapes)
        )


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
        rendered = (
            state.templates["gemm"]
            .render(
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
            )
            .rstrip()
        )
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
class QGemmOp(GemmLikeOpBase):
    """Quantized GEMM (com.microsoft contrib op).

    Computes ``Y = alpha * (dequantize(A) @ dequantize(B)) [+ C]``, optionally
    re-quantized to the output type when ``y_scale`` / ``y_zero_point`` are
    present.  B may use per-column quantization.
    """

    __io_inputs__ = (
        "input_a",
        "a_scale",
        "a_zero_point",
        "input_b",
        "b_scale",
        "b_zero_point",
        "input_c",
        "y_scale",
        "y_zero_point",
    )

    input_a: str
    a_scale: str
    a_zero_point: str
    input_b: str
    b_scale: str
    b_zero_point: str
    input_c: str | None
    y_scale: str | None
    y_zero_point: str | None
    output: str
    trans_a: int
    trans_b: int
    alpha: float
    input_a_dtype: ScalarType
    input_b_dtype: ScalarType
    dtype: ScalarType
    a_scale_dtype: ScalarType
    b_scale_dtype: ScalarType
    y_scale_dtype: ScalarType | None
    c_dtype: ScalarType | None
    a_scale_shape: tuple[int, ...]
    a_zero_shape: tuple[int, ...]
    b_scale_shape: tuple[int, ...]
    b_zero_shape: tuple[int, ...]
    y_scale_shape: tuple[int, ...] | None
    y_zero_shape: tuple[int, ...] | None
    b_scale_per_column: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = {"#include <math.h>"}
        output_dtype = ctx.dtype(self.output)
        if output_dtype.is_integer:
            includes.add("#include <limits.h>")
        return includes

    def infer_types(self, ctx: OpContext) -> None:
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, self.dtype)
            output_dtype = self.dtype
        if output_dtype != self.dtype:
            raise UnsupportedOpError(
                "QGemm output dtype mismatch, "
                f"expected {self.dtype.onnx_name}, got {output_dtype.onnx_name}"
            )
        ctx.set_derived(self, "trans_a", bool(self.trans_a))
        ctx.set_derived(self, "trans_b", bool(self.trans_b))

    def infer_shapes(self, ctx: OpContext) -> None:
        trans_a = ctx.require_derived(self, "trans_a")
        trans_b = ctx.require_derived(self, "trans_b")
        input_a_shape = ctx.shape(self.input_a)
        input_b_shape = ctx.shape(self.input_b)
        if len(input_a_shape) != 2 or len(input_b_shape) != 2:
            raise UnsupportedOpError(
                f"QGemm supports 2D inputs only, got {input_a_shape} x {input_b_shape}"
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
                f"QGemm inner dimensions must match, got {k_left} and {k_right}"
            )
        output_shape = (m, n)
        try:
            expected = ctx.shape(self.output)
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"QGemm output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self.output, output_shape)
        c_shape = None
        c_axis = "none"
        if self.input_c is not None:
            bias_shape = ctx.shape(self.input_c)
            c_shape, c_axis = GemmOp._validate_bias_shape(output_shape, bias_shape)
        ctx.set_derived(self, "m", m)
        ctx.set_derived(self, "n", n)
        ctx.set_derived(self, "k", k_left)
        ctx.set_derived(self, "c_shape", c_shape)
        ctx.set_derived(self, "c_axis", c_axis)

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input_a", self.input_a),
                ("a_scale", self.a_scale),
                ("a_zero_point", self.a_zero_point),
                ("input_b", self.input_b),
                ("b_scale", self.b_scale),
                ("b_zero_point", self.b_zero_point),
                ("input_c", self.input_c),
                ("y_scale", self.y_scale),
                ("y_zero_point", self.y_zero_point),
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
        a_scale_suffix = emitter.param_array_suffix(self.a_scale_shape)
        a_zero_suffix = emitter.param_array_suffix(self.a_zero_shape)
        b_scale_suffix = emitter.param_array_suffix(self.b_scale_shape)
        b_zero_suffix = emitter.param_array_suffix(self.b_zero_shape)
        c_suffix = emitter.param_array_suffix(c_shape) if c_shape is not None else ""

        output_dtype = emitter.ctx_dtype(self.output)
        output_c_type = output_dtype.c_type
        a_c_type = self.input_a_dtype.c_type
        b_c_type = self.input_b_dtype.c_type

        param_entries: list[tuple[str | None, str, str, bool]] = [
            (params["input_a"], a_c_type, input_a_suffix, True),
            (params["a_scale"], self.a_scale_dtype.c_type, a_scale_suffix, True),
            (
                params["a_zero_point"],
                self.input_a_dtype.c_type,
                a_zero_suffix,
                True,
            ),
            (params["input_b"], b_c_type, input_b_suffix, True),
            (params["b_scale"], self.b_scale_dtype.c_type, b_scale_suffix, True),
            (
                params["b_zero_point"],
                self.input_b_dtype.c_type,
                b_zero_suffix,
                True,
            ),
        ]
        if params["input_c"]:
            assert self.c_dtype is not None
            param_entries.append(
                (params["input_c"], self.c_dtype.c_type, c_suffix, True)
            )
        else:
            param_entries.append((None, "", "", True))

        if self.y_scale is not None:
            assert self.y_scale_dtype is not None
            assert self.y_scale_shape is not None
            assert self.y_zero_shape is not None
            y_scale_suffix = emitter.param_array_suffix(self.y_scale_shape)
            y_zero_suffix = emitter.param_array_suffix(self.y_zero_shape)
            param_entries.append(
                (params["y_scale"], self.y_scale_dtype.c_type, y_scale_suffix, True)
            )
            param_entries.append(
                (params["y_zero_point"], output_c_type, y_zero_suffix, True)
            )
        param_entries.append((params["output"], output_c_type, output_suffix, False))
        param_decls = emitter.build_param_decls(param_entries)

        # Determine compute type (double for best precision).
        compute_dtype = ScalarType.F64
        compute_type = "double"

        if output_dtype.is_signed:
            min_literal = "-128"
            max_literal = "127"
        elif output_dtype.is_integer:
            min_literal = "0"
            max_literal = "255"
        else:
            min_literal = "0"
            max_literal = "255"

        has_y_scale = self.y_scale is not None
        alpha_literal = emitter.format_literal(ScalarType.F64, self.alpha)

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

        rendered = (
            state.templates["qgemm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_a=params["input_a"],
                a_scale=params["a_scale"],
                a_zero_point=params["a_zero_point"],
                input_b=params["input_b"],
                b_scale=params["b_scale"],
                b_zero_point=params["b_zero_point"],
                input_c=params["input_c"],
                y_scale=params.get("y_scale"),
                y_zero_point=params.get("y_zero_point"),
                output=params["output"],
                params=param_decls,
                a_c_type=a_c_type,
                b_c_type=b_c_type,
                output_c_type=output_c_type,
                c_c_type=self.c_dtype.c_type if self.c_dtype is not None else "",
                compute_type=compute_type,
                compute_dtype=compute_dtype,
                dtype=output_dtype,
                alpha_literal=alpha_literal,
                trans_a=int(trans_a),
                trans_b=int(trans_b),
                m=m,
                n=n,
                k=k,
                c_rank=c_rank,
                c_dim0=c_dim0,
                c_dim1=c_dim1,
                c_axis=c_axis,
                has_y_scale=has_y_scale,
                b_scale_per_column=self.b_scale_per_column,
                min_literal=min_literal,
                max_literal=max_literal,
                dim_args=emitter.dim_args_str(),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        m = int(emitter.derived(self, "m"))
        n = int(emitter.derived(self, "n"))
        return (m, n)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        min_literal = output_dtype.min_literal
        params = emitter.shared_param_map(
            [
                ("input_q", self.input_q),
                ("input_k", self.input_k),
                ("input_v", self.input_v),
                ("input_attn_mask", self.input_attn_mask),
                ("input_past_key", self.input_past_key),
                ("input_past_value", self.input_past_value),
                ("input_nonpad_kv_seqlen", self.input_nonpad_kv_seqlen),
                ("output", self.output),
                ("output_present_key", self.output_present_key),
                ("output_present_value", self.output_present_value),
                ("output_qk_matmul", self.output_qk_matmul),
            ]
        )
        if self.q_rank == 4:
            input_q_shape = (self.batch, self.q_heads, self.q_seq, self.qk_head_size)
        else:
            input_q_shape = (self.batch, self.q_seq, self.q_hidden_size)
        if self.k_rank == 4:
            input_k_shape = (self.batch, self.kv_heads, self.kv_seq, self.qk_head_size)
        else:
            input_k_shape = (self.batch, self.kv_seq, self.k_hidden_size)
        if self.v_rank == 4:
            input_v_shape = (self.batch, self.kv_heads, self.kv_seq, self.v_head_size)
        else:
            input_v_shape = (self.batch, self.kv_seq, self.v_hidden_size)
        if self.output_rank == 4:
            output_shape = (self.batch, self.q_heads, self.q_seq, self.v_head_size)
        else:
            output_shape = (
                self.batch,
                self.q_seq,
                self.q_heads * self.v_head_size,
            )
        present_key_shape = (
            (self.batch, self.kv_heads, self.total_seq, self.qk_head_size)
            if self.output_present_key is not None
            else None
        )
        present_value_shape = (
            (self.batch, self.kv_heads, self.total_seq, self.v_head_size)
            if self.output_present_value is not None
            else None
        )
        qk_matmul_shape = (
            (self.batch, self.q_heads, self.q_seq, self.total_seq)
            if self.output_qk_matmul is not None
            else None
        )
        input_q_suffix = emitter.param_array_suffix(input_q_shape)
        input_k_suffix = emitter.param_array_suffix(input_k_shape)
        input_v_suffix = emitter.param_array_suffix(input_v_shape)
        input_mask_suffix = (
            emitter.param_array_suffix(self.mask_shape)
            if self.input_attn_mask is not None
            else ""
        )
        input_past_key_suffix = (
            emitter.param_array_suffix(
                (self.batch, self.kv_heads, self.past_seq, self.qk_head_size)
            )
            if self.input_past_key is not None
            else ""
        )
        input_past_value_suffix = (
            emitter.param_array_suffix(
                (self.batch, self.kv_heads, self.past_seq, self.v_head_size)
            )
            if self.input_past_value is not None
            else ""
        )
        input_nonpad_suffix = (
            emitter.param_array_suffix((self.batch,))
            if self.input_nonpad_kv_seqlen is not None
            else ""
        )
        output_suffix = emitter.param_array_suffix(output_shape)
        output_present_key_suffix = (
            emitter.param_array_suffix(present_key_shape)
            if present_key_shape is not None
            else ""
        )
        output_present_value_suffix = (
            emitter.param_array_suffix(present_value_shape)
            if present_value_shape is not None
            else ""
        )
        output_qk_matmul_suffix = (
            emitter.param_array_suffix(qk_matmul_shape)
            if qk_matmul_shape is not None
            else ""
        )
        mask_c_type = "bool" if self.mask_is_bool else c_type
        param_decls = emitter.build_param_decls(
            [
                (params["input_q"], c_type, input_q_suffix, True),
                (params["input_k"], c_type, input_k_suffix, True),
                (params["input_v"], c_type, input_v_suffix, True),
                (
                    (
                        params["input_attn_mask"],
                        mask_c_type,
                        input_mask_suffix,
                        True,
                    )
                    if params["input_attn_mask"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_past_key"],
                        c_type,
                        input_past_key_suffix,
                        True,
                    )
                    if params["input_past_key"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_past_value"],
                        c_type,
                        input_past_value_suffix,
                        True,
                    )
                    if params["input_past_value"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_nonpad_kv_seqlen"],
                        ScalarType.I64.c_type,
                        input_nonpad_suffix,
                        True,
                    )
                    if params["input_nonpad_kv_seqlen"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
                (
                    (
                        params["output_present_key"],
                        c_type,
                        output_present_key_suffix,
                        False,
                    )
                    if params["output_present_key"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_present_value"],
                        c_type,
                        output_present_value_suffix,
                        False,
                    )
                    if params["output_present_value"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_qk_matmul"],
                        c_type,
                        output_qk_matmul_suffix,
                        False,
                    )
                    if params["output_qk_matmul"]
                    else (None, "", "", False)
                ),
            ]
        )
        rendered = (
            state.templates["attention"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_q=params["input_q"],
                input_k=params["input_k"],
                input_v=params["input_v"],
                input_attn_mask=params["input_attn_mask"],
                input_past_key=params["input_past_key"],
                input_past_value=params["input_past_value"],
                input_nonpad_kv_seqlen=params["input_nonpad_kv_seqlen"],
                output=params["output"],
                output_present_key=params["output_present_key"],
                output_present_value=params["output_present_value"],
                output_qk_matmul=params["output_qk_matmul"],
                params=param_decls,
                c_type=c_type,
                nonpad_c_type=ScalarType.I64.c_type,
                zero_literal=zero_literal,
                min_literal=min_literal,
                scale_literal=emitter.format_floating(self.scale, self.dtype),
                softcap_literal=emitter.format_floating(self.softcap, self.dtype),
                one_literal=emitter.format_literal(self.dtype, 1),
                dtype=self.dtype,
                is_causal=int(self.is_causal),
                qk_matmul_output_mode=self.qk_matmul_output_mode,
                batch=self.batch,
                q_heads=self.q_heads,
                kv_heads=self.kv_heads,
                q_seq=self.q_seq,
                kv_seq=self.kv_seq,
                total_seq=self.total_seq,
                past_seq=self.past_seq,
                qk_head_size=self.qk_head_size,
                v_head_size=self.v_head_size,
                head_group_size=self.head_group_size,
                q_rank=self.q_rank,
                k_rank=self.k_rank,
                v_rank=self.v_rank,
                output_rank=self.output_rank,
                q_hidden_size=self.q_hidden_size,
                k_hidden_size=self.k_hidden_size,
                v_hidden_size=self.v_hidden_size,
                has_attn_mask=int(self.input_attn_mask is not None),
                mask_rank=self.mask_rank or 0,
                mask_is_bool=int(self.mask_is_bool),
                mask_broadcast_batch=int(self.mask_broadcast_batch),
                mask_broadcast_heads=int(self.mask_broadcast_heads),
                mask_broadcast_q_seq=int(self.mask_broadcast_q_seq),
                mask_q_seq=self.mask_q_seq or 0,
                mask_kv_seq=self.mask_kv_seq or 0,
                input_q_suffix=input_q_suffix,
                input_k_suffix=input_k_suffix,
                input_v_suffix=input_v_suffix,
                input_mask_suffix=input_mask_suffix,
                input_past_key_suffix=input_past_key_suffix,
                input_past_value_suffix=input_past_value_suffix,
                input_nonpad_suffix=input_nonpad_suffix,
                output_suffix=output_suffix,
                output_present_key_suffix=output_present_key_suffix,
                output_present_value_suffix=output_present_value_suffix,
                output_qk_matmul_suffix=output_qk_matmul_suffix,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        if self.output_rank == 3:
            return (self.batch, self.q_seq, self.q_heads * self.v_head_size)
        return (self.batch, self.q_heads, self.q_seq, self.v_head_size)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (self.output, self.computed_output_shape(emitter), self.dtype)
        ]
        if self.output_present_key is not None:
            outputs.append(
                (
                    self.output_present_key,
                    (self.batch, self.kv_heads, self.total_seq, self.qk_head_size),
                    self.dtype,
                )
            )
        if self.output_present_value is not None:
            outputs.append(
                (
                    self.output_present_value,
                    (self.batch, self.kv_heads, self.total_seq, self.v_head_size),
                    self.dtype,
                )
            )
        if self.output_qk_matmul is not None:
            outputs.append(
                (
                    self.output_qk_matmul,
                    (self.batch, self.q_heads, self.q_seq, self.total_seq),
                    self.dtype,
                )
            )
        return tuple(outputs)


@dataclass(frozen=True)
class MsAttentionOp(RenderableOpBase):
    """com.microsoft::Attention contrib operator.

    Fused self-attention that projects input via a combined QKV weight matrix,
    splits into Q/K/V heads, optionally concatenates past key/value state,
    computes scaled dot-product attention with optional masking, and writes
    output plus optional present state.
    """

    __io_inputs__ = (
        "input0",
        "weight",
        "bias",
        "mask_index",
        "past",
        "extra_add_qk",
    )
    __io_outputs__ = ("output", "present")
    input0: str
    weight: str
    bias: str
    mask_index: str | None
    past: str | None
    extra_add_qk: str | None
    output: str
    present: str | None
    batch: int
    seq_len: int
    num_heads: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int
    k_hidden_size: int
    v_hidden_size: int
    input_hidden_size: int
    past_seq: int
    total_seq: int
    scale: float
    unidirectional: bool
    mask_filter_value: float
    mask_type: int  # 0=NONE, 1=1D_END, 2=1D_END_START, 3=2D, 4=3D
    mask_shape: tuple[int, ...] | None
    dtype: ScalarType

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        min_literal = output_dtype.min_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("weight", self.weight),
                ("bias", self.bias),
                ("mask_index", self.mask_index),
                ("past", self.past),
                ("extra_add_qk", self.extra_add_qk),
                ("output", self.output),
                ("present", self.present),
            ]
        )
        input_shape = (self.batch, self.seq_len, self.input_hidden_size)
        weight_shape = (
            self.input_hidden_size,
            self.q_hidden_size + self.k_hidden_size + self.v_hidden_size,
        )
        bias_shape = (self.q_hidden_size + self.k_hidden_size + self.v_hidden_size,)
        output_shape = (self.batch, self.seq_len, self.v_hidden_size)
        input_suffix = emitter.param_array_suffix(input_shape)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        bias_suffix = emitter.param_array_suffix(bias_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        mask_suffix = (
            emitter.param_array_suffix(self.mask_shape)
            if self.mask_index is not None
            else ""
        )
        past_shape = (
            (2, self.batch, self.num_heads, self.past_seq, self.qk_head_size)
            if self.past is not None
            else None
        )
        past_suffix = (
            emitter.param_array_suffix(past_shape) if past_shape is not None else ""
        )
        extra_shape = (
            (self.batch, self.num_heads, self.seq_len, self.total_seq)
            if self.extra_add_qk is not None
            else None
        )
        extra_suffix = (
            emitter.param_array_suffix(extra_shape)
            if extra_shape is not None
            else ""
        )
        present_shape = (
            (2, self.batch, self.num_heads, self.total_seq, self.qk_head_size)
            if self.present is not None
            else None
        )
        present_suffix = (
            emitter.param_array_suffix(present_shape)
            if present_shape is not None
            else ""
        )
        mask_c_type = ScalarType.I32.c_type
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["weight"], c_type, weight_suffix, True),
                (params["bias"], c_type, bias_suffix, True),
                (
                    (params["mask_index"], mask_c_type, mask_suffix, True)
                    if params["mask_index"]
                    else (None, "", "", True)
                ),
                (
                    (params["past"], c_type, past_suffix, True)
                    if params["past"]
                    else (None, "", "", True)
                ),
                (
                    (params["extra_add_qk"], c_type, extra_suffix, True)
                    if params["extra_add_qk"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
                (
                    (params["present"], c_type, present_suffix, False)
                    if params["present"]
                    else (None, "", "", False)
                ),
            ]
        )
        rendered = (
            state.templates["ms_attention"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                weight=params["weight"],
                bias=params["bias"],
                mask_index=params["mask_index"],
                past=params["past"],
                extra_add_qk=params["extra_add_qk"],
                output=params["output"],
                present=params["present"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                min_literal=min_literal,
                mask_filter_literal=emitter.format_floating(
                    self.mask_filter_value, self.dtype
                ),
                scale_literal=emitter.format_floating(self.scale, self.dtype),
                dtype=self.dtype,
                batch=self.batch,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                qk_head_size=self.qk_head_size,
                v_head_size=self.v_head_size,
                q_hidden_size=self.q_hidden_size,
                k_hidden_size=self.k_hidden_size,
                v_hidden_size=self.v_hidden_size,
                input_hidden_size=self.input_hidden_size,
                past_seq=self.past_seq,
                total_seq=self.total_seq,
                unidirectional=int(self.unidirectional),
                mask_type=self.mask_type,
                mask_shape=self.mask_shape,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, self.seq_len, self.v_hidden_size)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (self.output, self.computed_output_shape(emitter), self.dtype)
        ]
        if self.present is not None:
            outputs.append(
                (
                    self.present,
                    (
                        2,
                        self.batch,
                        self.num_heads,
                        self.total_seq,
                        self.qk_head_size,
                    ),
                    self.dtype,
                )
            )
        return tuple(outputs)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("cos_cache", self.cos_cache),
                ("sin_cache", self.sin_cache),
                ("position_ids", self.position_ids),
                ("output", self.output),
            ]
        )
        input_suffix = emitter.param_array_suffix(
            self.input_shape, emitter.dim_names_for(self.input0)
        )
        cos_suffix = emitter.param_array_suffix(self.cos_shape)
        sin_suffix = emitter.param_array_suffix(self.sin_shape)
        position_suffix = (
            emitter.param_array_suffix(self.position_ids_shape)
            if self.position_ids_shape is not None
            else ""
        )
        output_suffix = emitter.param_array_suffix(
            self.input_shape, emitter.dim_names_for(self.output)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["cos_cache"], c_type, cos_suffix, True),
                (params["sin_cache"], c_type, sin_suffix, True),
                (
                    (
                        params["position_ids"],
                        self.position_ids_dtype.c_type,
                        position_suffix,
                        True,
                    )
                    if params["position_ids"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["rotary_embedding"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                cos_cache=params["cos_cache"],
                sin_cache=params["sin_cache"],
                position_ids=params["position_ids"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                cos_suffix=cos_suffix,
                sin_suffix=sin_suffix,
                position_suffix=position_suffix,
                output_suffix=output_suffix,
                batch=self.batch,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                rotary_dim_half=self.rotary_dim_half,
                input_rank=self.input_rank,
                interleaved=int(self.interleaved),
                has_position_ids=int(self.position_ids is not None),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.input_shape


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
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape_expr = CEmitterCompat.shape_dim_exprs(input_shape, input_dim_names)
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )
        out_indices = tuple(f"od{dim}" for dim in range(self.spatial_rank))
        kernel_indices = tuple(f"kd{dim}" for dim in range(self.spatial_rank))
        in_indices = tuple(f"id{dim}" for dim in range(self.spatial_rank))
        pad_begin = self.pads[: self.spatial_rank]
        group_in_channels = self.in_channels // self.group
        group_out_channels = self.out_channels // self.group
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        weight_suffix = emitter.param_array_suffix(weight_shape)
        bias_suffix = emitter.param_array_suffix((self.out_channels,))
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
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
                batch=input_shape_expr[0],
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                spatial_rank=self.spatial_rank,
                in_spatial=input_shape_expr[2:],
                out_spatial=output_shape_expr[2:],
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
        x_zero_expr = f"{params['x_zero_point']}[0]" if params["x_zero_point"] else "0"
        if params["w_zero_point"]:
            if self.w_zero_point_per_channel:
                w_zero_expr = f"{params['w_zero_point']}[oc_global]"
            else:
                w_zero_expr = f"{params['w_zero_point']}[0]"
        else:
            w_zero_expr = "0"
        rendered = (
            state.templates["conv_integer"]
            .render(
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
            )
            .rstrip()
        )
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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["qlinear_conv"]
            .render(
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
            )
            .rstrip()
        )
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
        rendered = (
            state.templates["conv_transpose"]
            .render(
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
            )
            .rstrip()
        )
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
        rendered = (
            state.templates["col2im"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, self.channels, *self.image_shape)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["deform_conv"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, self.out_channels, self.out_h, self.out_w)


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
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
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
        input_shape_expr = CEmitterCompat.shape_dim_exprs(input_shape, input_dim_names)
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )
        if self.spatial_rank == 3:
            in_d = input_shape_expr[2]
            in_h = input_shape_expr[3]
            in_w = input_shape_expr[4]
            out_d = output_shape_expr[2]
            out_h = output_shape_expr[3]
            out_w = output_shape_expr[4]
        elif self.spatial_rank == 1:
            in_d = self.in_d
            in_h = self.in_h
            in_w = input_shape_expr[2]
            out_d = self.out_d
            out_h = self.out_h
            out_w = output_shape_expr[2]
        else:
            in_d = self.in_d
            in_h = input_shape_expr[2]
            in_w = input_shape_expr[3]
            out_d = self.out_d
            out_h = output_shape_expr[2]
            out_w = output_shape_expr[3]
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["avg_pool"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=input_shape_expr[0],
                channels=input_shape_expr[1],
                spatial_rank=self.spatial_rank,
                in_d=in_d,
                in_h=in_h,
                in_w=in_w,
                out_d=out_d,
                out_h=out_h,
                out_w=out_w,
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        if self.spatial_rank == 3:
            return (self.batch, self.channels, self.out_d, self.out_h, self.out_w)
        if self.spatial_rank == 1:
            return (self.batch, self.channels, self.out_w)
        return (self.batch, self.channels, self.out_h, self.out_w)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input_scale", self.input_scale),
                ("input_zero_point", self.input_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("output", self.output),
            ]
        )
        input_shape = (
            (self.batch, self.channels, self.in_d, self.in_h, self.in_w)
            if self.spatial_rank == 3
            else (
                (self.batch, self.channels, self.in_w)
                if self.spatial_rank == 1
                else (self.batch, self.channels, self.in_h, self.in_w)
            )
        )
        output_shape = (
            (self.batch, self.channels, self.out_d, self.out_h, self.out_w)
            if self.spatial_rank == 3
            else (
                (self.batch, self.channels, self.out_w)
                if self.spatial_rank == 1
                else (self.batch, self.channels, self.out_h, self.out_w)
            )
        )
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        input_scale_suffix = emitter.param_array_suffix(self.input_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input_zero_suffix = emitter.param_array_suffix(self.input_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
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
                (params["output"], self.dtype.c_type, output_suffix, False),
            ]
        )
        compute_dtype = (
            ScalarType.F64
            if ScalarType.F64 in {self.input_scale_dtype, self.output_scale_dtype}
            else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        rendered = (
            state.templates["qlinear_avg_pool"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input_scale=params["input_scale"],
                input_zero_point=params["input_zero_point"],
                output_scale=params["output_scale"],
                output_zero_point=params["output_zero_point"],
                output=params["output"],
                params=param_decls,
                compute_type=compute_type,
                compute_dtype=compute_dtype,
                dtype=self.dtype,
                min_literal=self.dtype.min_literal,
                max_literal=self.dtype.max_literal,
                input_scale_expr=f"{params['input_scale']}[0]",
                input_zero_expr=f"{params['input_zero_point']}[0]",
                output_scale_expr=f"{params['output_scale']}[0]",
                output_zero_expr=f"{params['output_zero_point']}[0]",
                spatial_rank=self.spatial_rank,
                batch=self.batch,
                channels=self.channels,
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
                count_include_pad=self.count_include_pad,
                dim_args=emitter.dim_args_str(),
                output_c_type=self.dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        if self.spatial_rank == 3:
            return (self.batch, self.channels, self.out_d, self.out_h, self.out_w)
        if self.spatial_rank == 1:
            return (self.batch, self.channels, self.out_w)
        return (self.batch, self.channels, self.out_h, self.out_w)


@dataclass(frozen=True)
class LpPoolOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    dilations: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    p: float
    dtype: ScalarType

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        zero_literal = emitter.ctx_dtype(self.output).zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = (self.batch, self.channels, *self.in_spatial)
        output_shape = (self.batch, self.channels, *self.out_spatial)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["lp_pool"]
            .render(
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
                spatial_rank=self.spatial_rank,
                in_spatial=self.in_spatial,
                out_spatial=self.out_spatial,
                kernel_shape=self.kernel_shape,
                dilations=self.dilations,
                strides=self.strides,
                pads=self.pads,
                p=self.p,
                zero_literal=zero_literal,
                dtype=self.dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, self.channels, *self.out_spatial)


@dataclass(frozen=True)
class SoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        axis = emitter.derived(self, "axis")
        if self.use_legacy_axis_semantics:
            outer = (
                _shape_product_expr(output_shape[:axis], output_dim_names)
                if axis > 0
                else 1
            )
            axis_size = _shape_product_expr(output_shape[axis:], output_dim_names)
            inner = 1
        else:
            outer = (
                _shape_product_expr(output_shape[:axis], output_dim_names)
                if axis > 0
                else 1
            )
            axis_size = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)[
                axis
            ]
            inner = (
                _shape_product_expr(output_shape[axis + 1 :], output_dim_names)
                if axis + 1 < len(output_shape)
                else 1
            )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
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
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)
        outer = 1
        for dim in input_shape[:axis]:
            outer *= dim
        if self.use_legacy_axis_semantics:
            axis_size = 1
            for dim in input_shape[axis:]:
                axis_size *= dim
            inner = 1
        else:
            axis_size = input_shape[axis]
            inner = 1
            for dim in input_shape[axis + 1 :]:
                inner *= dim
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

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = {"#include <math.h>"}
        if ctx.dtype(self.output).is_integer:
            includes.add("#include <limits.h>")
        return includes

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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input_scale", self.input_scale),
                ("input_zero_point", self.input_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("output", self.output),
            ]
        )
        input_suffix = emitter.param_array_suffix(self.input_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        input_scale_suffix = emitter.param_array_suffix(self.input_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input_zero_suffix = emitter.param_array_suffix(self.input_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
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
                (params["output"], self.dtype.c_type, output_suffix, False),
            ]
        )
        compute_dtype = ScalarType.F32
        if (
            self.input_scale_dtype == ScalarType.F64
            or self.output_scale_dtype == ScalarType.F64
        ):
            compute_dtype = ScalarType.F64
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        rendered = (
            state.templates["qlinear_softmax"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                compute_type=compute_type,
                input_c_type=self.input_dtype.c_type,
                output_c_type=self.dtype.c_type,
                input0=params["input0"],
                output=params["output"],
                input_scale_expr=f"{params['input_scale']}[0]",
                output_scale_expr=f"{params['output_scale']}[0]",
                input_zero_expr=f"{params['input_zero_point']}[0]",
                output_zero_expr=f"{params['output_zero_point']}[0]",
                outer=self.outer,
                axis_size=self.axis_size,
                inner=self.inner,
                min_literal=self.dtype.min_literal,
                max_literal=self.dtype.max_literal,
                compute_dtype=compute_dtype,
                output_wrap=emitter.replicate_ort_bugs,
                output_is_signed=self.dtype.is_signed,
                dim_args=emitter.dim_args_str(),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape


@dataclass(frozen=True)
class LogSoftmaxOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    axis: int | None
    use_legacy_axis_semantics: bool = False

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        axis = emitter.derived(self, "axis")
        if self.use_legacy_axis_semantics:
            outer = (
                _shape_product_expr(output_shape[:axis], output_dim_names)
                if axis > 0
                else 1
            )
            axis_size = _shape_product_expr(output_shape[axis:], output_dim_names)
            inner = 1
        else:
            outer = (
                _shape_product_expr(output_shape[:axis], output_dim_names)
                if axis > 0
                else 1
            )
            axis_size = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)[
                axis
            ]
            inner = (
                _shape_product_expr(output_shape[axis + 1 :], output_dim_names)
                if axis + 1 < len(output_shape)
                else 1
            )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
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
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)


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
        output_dim_names = emitter.dim_names_for(self.output)
        output_dtype = emitter.ctx_dtype(self.output)
        zero_literal = output_dtype.zero_literal
        axis = emitter.derived(self, "axis")
        outer = (
            _shape_product_expr(output_shape[:axis], output_dim_names)
            if axis > 0
            else 1
        )
        axis_size = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)[axis]
        inner = (
            _shape_product_expr(output_shape[axis + 1 :], output_dim_names)
            if axis + 1 < len(output_shape)
            else 1
        )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        array_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
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
        ctx.set_shape(self.output, input_shape)
        ctx.set_derived(self, "axis", axis)


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

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        return {self.target_dtype}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        acc_dtype = emitter.accumulation_dtype(self.dtype)
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        acc_one_literal = emitter.format_literal(acc_dtype, 1)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("target", self.target),
                ("weight", self.weight),
                ("output", self.output),
            ]
        )
        input_suffix = emitter.param_array_suffix(self.input_shape)
        target_suffix = emitter.param_array_suffix(self.target_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        weight_suffix = f"[{self.c}]"
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["target"], self.target_dtype.c_type, target_suffix, True),
                (
                    (
                        params["weight"],
                        c_type,
                        weight_suffix,
                        True,
                    )
                    if params["weight"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["nllloss"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                target=params["target"],
                weight=params["weight"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                target_c_type=self.target_dtype.c_type,
                input_suffix=input_suffix,
                target_suffix=target_suffix,
                output_suffix=output_suffix,
                n=self.n,
                c=self.c,
                d=self.d,
                reduction=self.reduction,
                ignore_index=self.ignore_index,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(self.dtype, 1),
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = {"#include <math.h>"}
        if self.ignore_index is not None:
            includes.add("#include <stdbool.h>")
        return includes

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        return {self.target_dtype}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        acc_dtype = emitter.accumulation_dtype(self.dtype)
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        acc_one_literal = emitter.format_literal(acc_dtype, 1)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("target", self.target),
                ("weight", self.weight),
                ("output", self.output),
                ("log_prob", self.log_prob),
            ]
        )
        use_ignore_index = int(self.ignore_index is not None)
        ignore_index = self.ignore_index if self.ignore_index is not None else -1
        input_suffix = emitter.param_array_suffix(self.input_shape)
        target_suffix = emitter.param_array_suffix(self.target_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        log_prob_suffix = (
            emitter.param_array_suffix(self.log_prob_shape)
            if self.log_prob_shape is not None
            else ""
        )
        weight_suffix = f"[{self.c}]"
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["target"], self.target_dtype.c_type, target_suffix, True),
                (
                    (
                        params["weight"],
                        c_type,
                        weight_suffix,
                        True,
                    )
                    if params["weight"]
                    else (None, "", "", True)
                ),
                (params["output"], c_type, output_suffix, False),
                (
                    (
                        params["log_prob"],
                        c_type,
                        log_prob_suffix,
                        False,
                    )
                    if params["log_prob"]
                    else (None, "", "", False)
                ),
            ]
        )
        rendered = (
            state.templates["softmax_cross_entropy_loss"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                target=params["target"],
                weight=params["weight"],
                output=params["output"],
                log_prob=params["log_prob"],
                params=param_decls,
                c_type=c_type,
                target_c_type=self.target_dtype.c_type,
                input_suffix=input_suffix,
                target_suffix=target_suffix,
                output_suffix=output_suffix,
                log_prob_suffix=(
                    log_prob_suffix if self.log_prob_shape is not None else None
                ),
                n=self.n,
                c=self.c,
                d=self.d,
                reduction=self.reduction,
                use_ignore_index=use_ignore_index,
                ignore_index=ignore_index,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(self.dtype, 1),
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
                acc_dtype=acc_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_shape

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (self.output, self.output_shape, self.dtype)
        ]
        if self.log_prob is not None and self.log_prob_shape is not None:
            outputs.append((self.log_prob, self.log_prob_shape, self.dtype))
        return tuple(outputs)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["batch_norm"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["lp_norm"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, self.shape),)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["instance_norm"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (self.input0, self.shape),
            (self.scale, (self.channels,)),
            (self.bias, (self.channels,)),
        )


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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
        rendered = (
            state.templates["group_norm"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (self.input0, self.shape),
            (self.scale, (self.channels,)),
            (self.bias, (self.channels,)),
        )


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        acc_dtype = emitter.accumulation_dtype(self.dtype)
        acc_type = acc_dtype.c_type
        acc_zero_literal = emitter.format_literal(acc_dtype, 0)
        acc_one_literal = emitter.format_literal(acc_dtype, 1)
        acc_epsilon_literal = emitter.format_floating(self.epsilon, acc_dtype)
        use_kahan = False
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("bias", self.bias),
                ("output", self.output),
                ("mean_output", self.mean_output),
                ("invstd_output", self.invstd_output),
            ]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        prefix_loop_vars = loop_vars[: self.axis]
        norm_loop_vars = loop_vars[self.axis :]
        scale_index_vars = [
            "0" if dim == 1 else var
            for dim, var in zip(self.scale_shape, norm_loop_vars)
        ]
        bias_index_vars = None
        if self.bias_shape is not None and self.bias is not None:
            bias_index_vars = [
                "0" if dim == 1 else var
                for dim, var in zip(self.bias_shape, norm_loop_vars)
            ]
        mean_index_vars = [
            *prefix_loop_vars,
            *("0" for _ in norm_loop_vars),
        ]
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        scale_suffix = emitter.param_array_suffix(self.scale_shape)
        bias_suffix = (
            emitter.param_array_suffix(self.bias_shape)
            if self.bias_shape is not None
            else ""
        )
        mean_suffix = (
            emitter.param_array_suffix(
                self.shape[: self.axis] + (1,) * len(self.normalized_shape)
            )
            if self.mean_output is not None
            else ""
        )
        invstd_suffix = (
            emitter.param_array_suffix(
                self.shape[: self.axis] + (1,) * len(self.normalized_shape)
            )
            if self.invstd_output is not None
            else ""
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["scale"], c_type, scale_suffix, True),
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
                (
                    (
                        params["mean_output"],
                        c_type,
                        mean_suffix,
                        False,
                    )
                    if params["mean_output"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["invstd_output"],
                        c_type,
                        invstd_suffix,
                        False,
                    )
                    if params["invstd_output"]
                    else (None, "", "", False)
                ),
            ]
        )
        rendered = (
            state.templates["layer_norm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                bias=params["bias"],
                output=params["output"],
                mean_output=params["mean_output"],
                invstd_output=params["invstd_output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                bias_suffix=bias_suffix,
                mean_suffix=mean_suffix,
                invstd_suffix=invstd_suffix,
                prefix_shape=shape[: self.axis],
                norm_shape=shape[self.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                bias_index_vars=bias_index_vars,
                mean_index_vars=mean_index_vars,
                inner=self.inner,
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
                acc_epsilon_literal=acc_epsilon_literal,
                acc_dtype=acc_dtype,
                use_kahan=use_kahan,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, self.shape),
            (self.scale, self.scale_shape),
        ]
        if self.bias is not None and self.bias_shape is not None:
            inputs.append((self.bias, self.bias_shape))
        return tuple(inputs)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (self.output, self.shape, self.dtype)
        ]
        if self.mean_output is not None:
            mean_shape = self.shape[: self.axis] + (1,) * len(self.normalized_shape)
            outputs.append((self.mean_output, mean_shape, self.dtype))
        if self.invstd_output is not None:
            invstd_shape = self.shape[: self.axis] + (1,) * len(self.normalized_shape)
            outputs.append((self.invstd_output, invstd_shape, self.dtype))
        return tuple(outputs)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["mean_variance_norm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                loop_vars=loop_vars,
                axes=self.axes,
                non_axes=self.non_axes,
                reduce_count=self.reduce_count,
                epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
                dtype=self.dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, self.shape),)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("output", self.output),
            ]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        prefix_loop_vars = loop_vars[: self.axis]
        norm_loop_vars = loop_vars[self.axis :]
        scale_index_vars = [
            "0" if dim == 1 else var
            for dim, var in zip(self.scale_shape, norm_loop_vars)
        ]
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        scale_suffix = emitter.param_array_suffix(self.scale_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["scale"], c_type, scale_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["rms_norm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                prefix_shape=shape[: self.axis],
                norm_shape=shape[self.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                inner=self.inner,
                epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
                dtype=self.dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, self.shape), (self.scale, self.scale_shape))


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        shape = CEmitterCompat.codegen_shape(self.shape)
        loop_vars = CEmitterCompat.loop_vars(shape)
        input_suffix = emitter.param_array_suffix(shape)
        output_suffix = emitter.param_array_suffix(shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["lrn"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                channels=self.channels,
                half=self.half,
                loop_vars=loop_vars,
                zero_literal=zero_literal,
                alpha_div_size_literal=emitter.format_floating(
                    self.alpha / self.size, self.dtype
                ),
                beta_literal=emitter.format_floating(self.beta, self.dtype),
                bias_literal=emitter.format_floating(self.bias, self.dtype),
                dtype=self.dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.shape


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output_y or self.output_y_h)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [
                ("input_x", self.input_x),
                ("input_w", self.input_w),
                ("input_r", self.input_r),
                ("input_b", self.input_b),
                ("input_sequence_lens", self.input_sequence_lens),
                ("input_initial_h", self.input_initial_h),
                ("output_y", self.output_y),
                ("output_y_h", self.output_y_h),
            ]
        )
        input_x_shape = (
            (self.seq_length, self.batch_size, self.input_size)
            if self.layout == 0
            else (self.batch_size, self.seq_length, self.input_size)
        )
        w_shape = (self.num_directions, 3 * self.hidden_size, self.input_size)
        r_shape = (self.num_directions, 3 * self.hidden_size, self.hidden_size)
        b_shape = (
            (self.num_directions, 6 * self.hidden_size)
            if self.input_b is not None
            else None
        )
        seq_shape = (self.batch_size,) if self.input_sequence_lens is not None else None
        state_shape = (
            (self.num_directions, self.batch_size, self.hidden_size)
            if self.layout == 0
            else (self.batch_size, self.num_directions, self.hidden_size)
        )
        h_shape = (
            state_shape
            if self.input_initial_h is not None or self.output_y_h is not None
            else None
        )
        y_shape = (
            (self.seq_length, self.num_directions, self.batch_size, self.hidden_size)
            if self.layout == 0
            else (
                self.batch_size,
                self.seq_length,
                self.num_directions,
                self.hidden_size,
            )
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_x"],
                    c_type,
                    emitter.param_array_suffix(input_x_shape),
                    True,
                ),
                (
                    params["input_w"],
                    c_type,
                    emitter.param_array_suffix(w_shape),
                    True,
                ),
                (
                    params["input_r"],
                    c_type,
                    emitter.param_array_suffix(r_shape),
                    True,
                ),
                (
                    (
                        params["input_b"],
                        c_type,
                        emitter.param_array_suffix(b_shape),
                        True,
                    )
                    if params["input_b"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_sequence_lens"],
                        (self.sequence_lens_dtype or ScalarType.I64).c_type,
                        emitter.param_array_suffix(seq_shape),
                        True,
                    )
                    if params["input_sequence_lens"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_h"],
                        c_type,
                        emitter.param_array_suffix(h_shape),
                        True,
                    )
                    if params["input_initial_h"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["output_y"],
                        c_type,
                        emitter.param_array_suffix(y_shape),
                        False,
                    )
                    if params["output_y"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_h"],
                        c_type,
                        emitter.param_array_suffix(h_shape),
                        False,
                    )
                    if params["output_y_h"]
                    else (None, "", "", False)
                ),
            ]
        )
        activation_functions = tuple(
            emitter.rnn_activation_function_name(kind, alpha, beta, self.dtype)
            for kind, alpha, beta in zip(
                self.activation_kinds,
                self.activation_alphas,
                self.activation_betas,
            )
        )
        rendered = (
            state.templates["gru"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_x=params["input_x"],
                input_w=params["input_w"],
                input_r=params["input_r"],
                input_b=params["input_b"],
                input_sequence_lens=params["input_sequence_lens"],
                input_initial_h=params["input_initial_h"],
                output_y=params["output_y"],
                output_y_h=params["output_y_h"],
                params=param_decls,
                c_type=c_type,
                seq_c_type=(self.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(self.dtype, 1),
                clip_literal=(
                    emitter.format_floating(self.clip, self.dtype)
                    if self.clip is not None
                    else emitter.format_literal(self.dtype, 0)
                ),
                use_clip=int(self.clip is not None and self.clip > 0),
                seq_length=self.seq_length,
                batch_size=self.batch_size,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_directions=self.num_directions,
                layout=self.layout,
                direction=self.direction,
                linear_before_reset=self.linear_before_reset,
                activation_functions=activation_functions,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = []
        if self.output_y is not None:
            if self.layout == 0:
                y_shape = (
                    self.seq_length,
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                y_shape = (
                    self.batch_size,
                    self.seq_length,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y, y_shape, self.dtype))
        if self.output_y_h is not None:
            if self.layout == 0:
                state_shape = (
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                state_shape = (
                    self.batch_size,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y_h, state_shape, self.dtype))
        return tuple(outputs)


@dataclass(frozen=True)
class RnnOp(RenderableOpBase):
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
    clip: float | None
    activation_kinds: tuple[int, ...]
    activation_alphas: tuple[float, ...]
    activation_betas: tuple[float, ...]
    dtype: ScalarType
    sequence_lens_dtype: ScalarType | None

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output_y or self.output_y_h)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [
                ("input_x", self.input_x),
                ("input_w", self.input_w),
                ("input_r", self.input_r),
                ("input_b", self.input_b),
                ("input_sequence_lens", self.input_sequence_lens),
                ("input_initial_h", self.input_initial_h),
                ("output_y", self.output_y),
                ("output_y_h", self.output_y_h),
            ]
        )
        input_x_shape = (
            (self.seq_length, self.batch_size, self.input_size)
            if self.layout == 0
            else (self.batch_size, self.seq_length, self.input_size)
        )
        w_shape = (self.num_directions, self.hidden_size, self.input_size)
        r_shape = (self.num_directions, self.hidden_size, self.hidden_size)
        b_shape = (
            (self.num_directions, 2 * self.hidden_size)
            if self.input_b is not None
            else None
        )
        seq_shape = (self.batch_size,) if self.input_sequence_lens is not None else None
        state_shape = (
            (self.num_directions, self.batch_size, self.hidden_size)
            if self.layout == 0
            else (self.batch_size, self.num_directions, self.hidden_size)
        )
        h_shape = (
            state_shape
            if self.input_initial_h is not None or self.output_y_h is not None
            else None
        )
        y_shape = (
            (self.seq_length, self.num_directions, self.batch_size, self.hidden_size)
            if self.layout == 0
            else (
                self.batch_size,
                self.seq_length,
                self.num_directions,
                self.hidden_size,
            )
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_x"],
                    c_type,
                    emitter.param_array_suffix(input_x_shape),
                    True,
                ),
                (
                    params["input_w"],
                    c_type,
                    emitter.param_array_suffix(w_shape),
                    True,
                ),
                (
                    params["input_r"],
                    c_type,
                    emitter.param_array_suffix(r_shape),
                    True,
                ),
                (
                    (
                        params["input_b"],
                        c_type,
                        emitter.param_array_suffix(b_shape),
                        True,
                    )
                    if params["input_b"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_sequence_lens"],
                        (self.sequence_lens_dtype or ScalarType.I64).c_type,
                        emitter.param_array_suffix(seq_shape),
                        True,
                    )
                    if params["input_sequence_lens"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_h"],
                        c_type,
                        emitter.param_array_suffix(h_shape),
                        True,
                    )
                    if params["input_initial_h"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["output_y"],
                        c_type,
                        emitter.param_array_suffix(y_shape),
                        False,
                    )
                    if params["output_y"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_h"],
                        c_type,
                        emitter.param_array_suffix(h_shape),
                        False,
                    )
                    if params["output_y_h"]
                    else (None, "", "", False)
                ),
            ]
        )
        activation_functions = tuple(
            emitter.rnn_activation_function_name(kind, alpha, beta, self.dtype)
            for kind, alpha, beta in zip(
                self.activation_kinds,
                self.activation_alphas,
                self.activation_betas,
            )
        )
        rendered = (
            state.templates["rnn"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_x=params["input_x"],
                input_w=params["input_w"],
                input_r=params["input_r"],
                input_b=params["input_b"],
                input_sequence_lens=params["input_sequence_lens"],
                input_initial_h=params["input_initial_h"],
                output_y=params["output_y"],
                output_y_h=params["output_y_h"],
                params=param_decls,
                c_type=c_type,
                seq_c_type=(self.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                clip_literal=(
                    emitter.format_floating(self.clip, self.dtype)
                    if self.clip is not None
                    else emitter.format_literal(self.dtype, 0)
                ),
                use_clip=int(self.clip is not None and self.clip > 0),
                seq_length=self.seq_length,
                batch_size=self.batch_size,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_directions=self.num_directions,
                layout=self.layout,
                direction=self.direction,
                activation_functions=activation_functions,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = []
        if self.output_y is not None:
            if self.layout == 0:
                y_shape = (
                    self.seq_length,
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                y_shape = (
                    self.batch_size,
                    self.seq_length,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y, y_shape, self.dtype))
        if self.output_y_h is not None:
            if self.layout == 0:
                state_shape = (
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                state_shape = (
                    self.batch_size,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y_h, state_shape, self.dtype))
        return tuple(outputs)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.ctx_dtype(
            self.output_y or self.output_y_h or self.output_y_c
        )
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [
                ("input_x", self.input_x),
                ("input_w", self.input_w),
                ("input_r", self.input_r),
                ("input_b", self.input_b),
                ("input_sequence_lens", self.input_sequence_lens),
                ("input_initial_h", self.input_initial_h),
                ("input_initial_c", self.input_initial_c),
                ("input_p", self.input_p),
                ("output_y", self.output_y),
                ("output_y_h", self.output_y_h),
                ("output_y_c", self.output_y_c),
            ]
        )
        input_x_shape = emitter.ctx_shape(self.input_x)
        input_x_dim_names = emitter.dim_names_for(self.input_x)
        input_x_dims = CEmitterCompat.shape_dim_exprs(input_x_shape, input_x_dim_names)
        seq_length = input_x_dims[0] if self.layout == 0 else input_x_dims[1]
        batch_size = input_x_dims[1] if self.layout == 0 else input_x_dims[0]
        w_shape = emitter.ctx_shape(self.input_w)
        r_shape = emitter.ctx_shape(self.input_r)
        b_shape = emitter.ctx_shape(self.input_b) if self.input_b is not None else None
        seq_shape = (
            emitter.ctx_shape(self.input_sequence_lens)
            if self.input_sequence_lens is not None
            else None
        )
        h_name = self.input_initial_h or self.output_y_h
        h_shape = emitter.ctx_shape(h_name) if h_name is not None else None
        c_name = self.input_initial_c or self.output_y_c
        c_shape = emitter.ctx_shape(c_name) if c_name is not None else None
        p_shape = emitter.ctx_shape(self.input_p) if self.input_p is not None else None
        y_shape = (
            emitter.ctx_shape(self.output_y) if self.output_y is not None else None
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_x"],
                    c_type,
                    emitter.param_array_suffix(input_x_shape, input_x_dim_names),
                    True,
                ),
                (
                    params["input_w"],
                    c_type,
                    emitter.param_array_suffix(
                        w_shape, emitter.dim_names_for(self.input_w)
                    ),
                    True,
                ),
                (
                    params["input_r"],
                    c_type,
                    emitter.param_array_suffix(
                        r_shape, emitter.dim_names_for(self.input_r)
                    ),
                    True,
                ),
                (
                    (
                        params["input_b"],
                        c_type,
                        emitter.param_array_suffix(
                            b_shape, emitter.dim_names_for(self.input_b)
                        ),
                        True,
                    )
                    if params["input_b"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_sequence_lens"],
                        (self.sequence_lens_dtype or ScalarType.I64).c_type,
                        emitter.param_array_suffix(
                            seq_shape, emitter.dim_names_for(self.input_sequence_lens)
                        ),
                        True,
                    )
                    if params["input_sequence_lens"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_h"],
                        c_type,
                        emitter.param_array_suffix(
                            h_shape, emitter.dim_names_for(self.input_initial_h)
                        ),
                        True,
                    )
                    if params["input_initial_h"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_c"],
                        c_type,
                        emitter.param_array_suffix(
                            c_shape, emitter.dim_names_for(self.input_initial_c)
                        ),
                        True,
                    )
                    if params["input_initial_c"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_p"],
                        c_type,
                        emitter.param_array_suffix(
                            p_shape, emitter.dim_names_for(self.input_p)
                        ),
                        True,
                    )
                    if params["input_p"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["output_y"],
                        c_type,
                        emitter.param_array_suffix(
                            y_shape, emitter.dim_names_for(self.output_y)
                        ),
                        False,
                    )
                    if params["output_y"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_h"],
                        c_type,
                        emitter.param_array_suffix(
                            h_shape, emitter.dim_names_for(self.output_y_h)
                        ),
                        False,
                    )
                    if params["output_y_h"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_c"],
                        c_type,
                        emitter.param_array_suffix(
                            c_shape, emitter.dim_names_for(self.output_y_c)
                        ),
                        False,
                    )
                    if params["output_y_c"]
                    else (None, "", "", False)
                ),
            ]
        )
        activation_functions = tuple(
            emitter.rnn_activation_function_name(kind, alpha, beta, self.dtype)
            for kind, alpha, beta in zip(
                self.activation_kinds,
                self.activation_alphas,
                self.activation_betas,
            )
        )
        rendered = (
            state.templates["lstm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_x=params["input_x"],
                input_w=params["input_w"],
                input_r=params["input_r"],
                input_b=params["input_b"],
                input_sequence_lens=params["input_sequence_lens"],
                input_initial_h=params["input_initial_h"],
                input_initial_c=params["input_initial_c"],
                input_p=params["input_p"],
                output_y=params["output_y"],
                output_y_h=params["output_y_h"],
                output_y_c=params["output_y_c"],
                dim_args=dim_args,
                params=param_decls,
                c_type=c_type,
                seq_c_type=(self.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(self.dtype, 1),
                clip_literal=(
                    emitter.format_floating(self.clip, self.dtype)
                    if self.clip is not None
                    else emitter.format_literal(self.dtype, 0)
                ),
                use_clip=int(self.clip is not None and self.clip > 0),
                seq_length=seq_length,
                batch_size=batch_size,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_directions=self.num_directions,
                layout=self.layout,
                direction=self.direction,
                input_forget=self.input_forget,
                activation_functions=activation_functions,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = []
        if self.output_y is not None:
            if self.layout == 0:
                y_shape = (
                    self.seq_length,
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                y_shape = (
                    self.batch_size,
                    self.seq_length,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y, y_shape, self.dtype))
        if self.output_y_h is not None:
            outputs.append(
                (
                    self.output_y_h,
                    (self.num_directions, self.batch_size, self.hidden_size),
                    self.dtype,
                )
            )
        if self.output_y_c is not None:
            outputs.append(
                (
                    self.output_y_c,
                    (self.num_directions, self.batch_size, self.hidden_size),
                    self.dtype,
                )
            )
        return tuple(outputs)


@dataclass(frozen=True)
class DynamicQuantizeLstmOp(RenderableOpBase):
    __io_inputs__ = (
        "input_x",
        "input_w",
        "input_r",
        "input_b",
        "input_sequence_lens",
        "input_initial_h",
        "input_initial_c",
        "input_p",
        "input_w_scale",
        "input_w_zero_point",
        "input_r_scale",
        "input_r_zero_point",
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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.ctx_dtype(
            self.output_y or self.output_y_h or self.output_y_c
        )
        c_type = output_dtype.c_type
        w_c_type = self.weight_dtype.c_type
        zero_literal = output_dtype.zero_literal
        params = emitter.shared_param_map(
            [
                ("input_x", self.input_x),
                ("input_w", self.input_w),
                ("input_r", self.input_r),
                ("input_b", self.input_b),
                ("input_sequence_lens", self.input_sequence_lens),
                ("input_initial_h", self.input_initial_h),
                ("input_initial_c", self.input_initial_c),
                ("input_p", self.input_p),
                ("input_w_scale", self.input_w_scale),
                ("input_w_zero_point", self.input_w_zero_point),
                ("input_r_scale", self.input_r_scale),
                ("input_r_zero_point", self.input_r_zero_point),
                ("output_y", self.output_y),
                ("output_y_h", self.output_y_h),
                ("output_y_c", self.output_y_c),
            ]
        )
        input_x_shape = emitter.ctx_shape(self.input_x)
        input_x_dim_names = emitter.dim_names_for(self.input_x)
        input_x_dims = CEmitterCompat.shape_dim_exprs(input_x_shape, input_x_dim_names)
        seq_length = input_x_dims[0] if self.layout == 0 else input_x_dims[1]
        batch_size = input_x_dims[1] if self.layout == 0 else input_x_dims[0]
        w_shape = emitter.ctx_shape(self.input_w)
        r_shape = emitter.ctx_shape(self.input_r)
        b_shape = emitter.ctx_shape(self.input_b) if self.input_b is not None else None
        seq_shape = (
            emitter.ctx_shape(self.input_sequence_lens)
            if self.input_sequence_lens is not None
            else None
        )
        h_name = self.input_initial_h or self.output_y_h
        h_shape = emitter.ctx_shape(h_name) if h_name is not None else None
        c_name = self.input_initial_c or self.output_y_c
        c_shape = emitter.ctx_shape(c_name) if c_name is not None else None
        p_shape = emitter.ctx_shape(self.input_p) if self.input_p is not None else None
        y_shape = (
            emitter.ctx_shape(self.output_y) if self.output_y is not None else None
        )
        w_scale_shape = emitter.ctx_shape(self.input_w_scale)
        w_zp_shape = emitter.ctx_shape(self.input_w_zero_point)
        r_scale_shape = emitter.ctx_shape(self.input_r_scale)
        r_zp_shape = emitter.ctx_shape(self.input_r_zero_point)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_x"],
                    c_type,
                    emitter.param_array_suffix(input_x_shape, input_x_dim_names),
                    True,
                ),
                (
                    params["input_w"],
                    w_c_type,
                    emitter.param_array_suffix(
                        w_shape, emitter.dim_names_for(self.input_w)
                    ),
                    True,
                ),
                (
                    params["input_r"],
                    w_c_type,
                    emitter.param_array_suffix(
                        r_shape, emitter.dim_names_for(self.input_r)
                    ),
                    True,
                ),
                (
                    (
                        params["input_b"],
                        c_type,
                        emitter.param_array_suffix(
                            b_shape, emitter.dim_names_for(self.input_b)
                        ),
                        True,
                    )
                    if params["input_b"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_sequence_lens"],
                        (self.sequence_lens_dtype or ScalarType.I64).c_type,
                        emitter.param_array_suffix(
                            seq_shape,
                            emitter.dim_names_for(self.input_sequence_lens),
                        ),
                        True,
                    )
                    if params["input_sequence_lens"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_h"],
                        c_type,
                        emitter.param_array_suffix(
                            h_shape, emitter.dim_names_for(self.input_initial_h)
                        ),
                        True,
                    )
                    if params["input_initial_h"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_initial_c"],
                        c_type,
                        emitter.param_array_suffix(
                            c_shape, emitter.dim_names_for(self.input_initial_c)
                        ),
                        True,
                    )
                    if params["input_initial_c"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_p"],
                        c_type,
                        emitter.param_array_suffix(
                            p_shape, emitter.dim_names_for(self.input_p)
                        ),
                        True,
                    )
                    if params["input_p"]
                    else (None, "", "", True)
                ),
                (
                    params["input_w_scale"],
                    c_type,
                    emitter.param_array_suffix(
                        w_scale_shape, emitter.dim_names_for(self.input_w_scale)
                    ),
                    True,
                ),
                (
                    params["input_w_zero_point"],
                    w_c_type,
                    emitter.param_array_suffix(
                        w_zp_shape, emitter.dim_names_for(self.input_w_zero_point)
                    ),
                    True,
                ),
                (
                    params["input_r_scale"],
                    c_type,
                    emitter.param_array_suffix(
                        r_scale_shape, emitter.dim_names_for(self.input_r_scale)
                    ),
                    True,
                ),
                (
                    params["input_r_zero_point"],
                    w_c_type,
                    emitter.param_array_suffix(
                        r_zp_shape, emitter.dim_names_for(self.input_r_zero_point)
                    ),
                    True,
                ),
                (
                    (
                        params["output_y"],
                        c_type,
                        emitter.param_array_suffix(
                            y_shape, emitter.dim_names_for(self.output_y)
                        ),
                        False,
                    )
                    if params["output_y"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_h"],
                        c_type,
                        emitter.param_array_suffix(
                            h_shape, emitter.dim_names_for(self.output_y_h)
                        ),
                        False,
                    )
                    if params["output_y_h"]
                    else (None, "", "", False)
                ),
                (
                    (
                        params["output_y_c"],
                        c_type,
                        emitter.param_array_suffix(
                            c_shape, emitter.dim_names_for(self.output_y_c)
                        ),
                        False,
                    )
                    if params["output_y_c"]
                    else (None, "", "", False)
                ),
            ]
        )
        activation_functions = tuple(
            emitter.rnn_activation_function_name(kind, alpha, beta, self.dtype)
            for kind, alpha, beta in zip(
                self.activation_kinds,
                self.activation_alphas,
                self.activation_betas,
            )
        )
        rendered = (
            state.templates["dynamic_quantize_lstm"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input_x=params["input_x"],
                input_w=params["input_w"],
                input_r=params["input_r"],
                input_b=params["input_b"],
                input_sequence_lens=params["input_sequence_lens"],
                input_initial_h=params["input_initial_h"],
                input_initial_c=params["input_initial_c"],
                input_p=params["input_p"],
                input_w_scale=params["input_w_scale"],
                input_w_zero_point=params["input_w_zero_point"],
                input_r_scale=params["input_r_scale"],
                input_r_zero_point=params["input_r_zero_point"],
                output_y=params["output_y"],
                output_y_h=params["output_y_h"],
                output_y_c=params["output_y_c"],
                dim_args=dim_args,
                params=param_decls,
                c_type=c_type,
                seq_c_type=(self.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                one_literal=emitter.format_literal(self.dtype, 1),
                clip_literal=(
                    emitter.format_floating(self.clip, self.dtype)
                    if self.clip is not None
                    else emitter.format_literal(self.dtype, 0)
                ),
                use_clip=int(self.clip is not None and self.clip > 0),
                seq_length=seq_length,
                batch_size=batch_size,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_directions=self.num_directions,
                layout=self.layout,
                direction=self.direction,
                input_forget=self.input_forget,
                activation_functions=activation_functions,
                per_channel_weights=self.per_channel_weights,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = []
        if self.output_y is not None:
            if self.layout == 0:
                y_shape = (
                    self.seq_length,
                    self.num_directions,
                    self.batch_size,
                    self.hidden_size,
                )
            else:
                y_shape = (
                    self.batch_size,
                    self.seq_length,
                    self.num_directions,
                    self.hidden_size,
                )
            outputs.append((self.output_y, y_shape, self.dtype))
        if self.output_y_h is not None:
            outputs.append(
                (
                    self.output_y_h,
                    (self.num_directions, self.batch_size, self.hidden_size),
                    self.dtype,
                )
            )
        if self.output_y_c is not None:
            outputs.append(
                (
                    self.output_y_c,
                    (self.num_directions, self.batch_size, self.hidden_size),
                    self.dtype,
                )
            )
        return tuple(outputs)


@dataclass(frozen=True)
class AdamOp(RenderableOpBase):
    __io_inputs__ = (
        "rate",
        "timestep",
        "inputs",
        "gradients",
        "velocities",
        "accumulators",
    )
    __io_outputs__ = ("outputs", "velocity_outputs", "accumulator_outputs")
    rate: str
    timestep: str
    inputs: tuple[str, ...]
    gradients: tuple[str, ...]
    velocities: tuple[str, ...]
    accumulators: tuple[str, ...]
    outputs: tuple[str, ...]
    velocity_outputs: tuple[str, ...]
    accumulator_outputs: tuple[str, ...]
    rate_shape: tuple[int, ...]
    timestep_shape: tuple[int, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtype: ScalarType
    rate_dtype: ScalarType
    timestep_dtype: ScalarType
    alpha: float
    beta: float
    epsilon: float
    norm_coefficient: float
    norm_coefficient_post: float

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def call_args(self) -> tuple[str, ...]:
        args = [self.rate, self.timestep]
        for index in range(len(self.inputs)):
            args.extend(
                [
                    self.inputs[index],
                    self.gradients[index],
                    self.velocities[index],
                    self.accumulators[index],
                    self.outputs[index],
                    self.velocity_outputs[index],
                    self.accumulator_outputs[index],
                ]
            )
        return tuple(args)

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.outputs[0])
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("rate", self.rate),
                ("timestep", self.timestep),
                *((f"input{idx}", name) for idx, name in enumerate(self.inputs)),
                *((f"grad{idx}", name) for idx, name in enumerate(self.gradients)),
                *((f"vel{idx}", name) for idx, name in enumerate(self.velocities)),
                *((f"acc{idx}", name) for idx, name in enumerate(self.accumulators)),
                *((f"output{idx}", name) for idx, name in enumerate(self.outputs)),
                *(
                    (f"vel_output{idx}", name)
                    for idx, name in enumerate(self.velocity_outputs)
                ),
                *(
                    (f"acc_output{idx}", name)
                    for idx, name in enumerate(self.accumulator_outputs)
                ),
            ]
        )
        rate_suffix = emitter.param_array_suffix(
            self.rate_shape, emitter.dim_names_for(self.rate)
        )
        timestep_suffix = emitter.param_array_suffix(
            self.timestep_shape, emitter.dim_names_for(self.timestep)
        )
        param_specs = [
            (params["rate"], self.rate_dtype.c_type, rate_suffix, True),
            (
                params["timestep"],
                self.timestep_dtype.c_type,
                timestep_suffix,
                True,
            ),
        ]
        tensor_specs = []
        for idx, shape in enumerate(self.output_shapes):
            input_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.inputs[idx])
            )
            grad_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.gradients[idx])
            )
            vel_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.velocities[idx])
            )
            acc_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.accumulators[idx])
            )
            output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx], emitter.dim_names_for(self.outputs[idx])
            )
            vel_output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx],
                emitter.dim_names_for(self.velocity_outputs[idx]),
            )
            acc_output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx],
                emitter.dim_names_for(self.accumulator_outputs[idx]),
            )
            param_specs.extend(
                [
                    (params[f"input{idx}"], c_type, input_suffix, True),
                    (params[f"grad{idx}"], c_type, grad_suffix, True),
                    (params[f"vel{idx}"], c_type, vel_suffix, True),
                    (params[f"acc{idx}"], c_type, acc_suffix, True),
                    (params[f"output{idx}"], c_type, output_suffix, False),
                    (params[f"vel_output{idx}"], c_type, vel_output_suffix, False),
                    (params[f"acc_output{idx}"], c_type, acc_output_suffix, False),
                ]
            )
            output_dim_names = emitter.dim_names_for(self.outputs[idx])
            shape_exprs = CEmitterCompat.shape_dim_exprs(shape, output_dim_names)
            loop_vars = CEmitterCompat.loop_vars(shape)
            index_suffix = "".join(f"[{var}]" for var in loop_vars)
            tensor_specs.append(
                {
                    "shape": shape_exprs,
                    "loop_vars": loop_vars,
                    "input_expr": f"{params[f'input{idx}']}{index_suffix}",
                    "grad_expr": f"{params[f'grad{idx}']}{index_suffix}",
                    "vel_expr": f"{params[f'vel{idx}']}{index_suffix}",
                    "acc_expr": f"{params[f'acc{idx}']}{index_suffix}",
                    "output_expr": f"{params[f'output{idx}']}{index_suffix}",
                    "vel_output_expr": f"{params[f'vel_output{idx}']}{index_suffix}",
                    "acc_output_expr": f"{params[f'acc_output{idx}']}{index_suffix}",
                }
            )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["adam"]
            .render(
                model_name=model.name,
                op_name=op_name,
                rate=params["rate"],
                timestep=params["timestep"],
                params=param_decls,
                c_type=c_type,
                one_literal=emitter.format_literal(self.dtype, 1),
                alpha_literal=emitter.format_floating(self.alpha, self.dtype),
                beta_literal=emitter.format_floating(self.beta, self.dtype),
                epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
                norm_coefficient_literal=emitter.format_floating(
                    self.norm_coefficient, self.dtype
                ),
                norm_coefficient_post_literal=emitter.format_floating(
                    self.norm_coefficient_post, self.dtype
                ),
                dtype=self.dtype,
                tensors=tensor_specs,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs = [
            (name, shape, self.dtype)
            for name, shape in zip(self.outputs, self.output_shapes)
        ]
        outputs.extend(
            (name, shape, self.dtype)
            for name, shape in zip(self.velocity_outputs, self.output_shapes)
        )
        outputs.extend(
            (name, shape, self.dtype)
            for name, shape in zip(self.accumulator_outputs, self.output_shapes)
        )
        return tuple(outputs)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.outputs[0])
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("rate", self.rate),
                ("timestep", self.timestep),
                *((f"input{idx}", name) for idx, name in enumerate(self.inputs)),
                *((f"grad{idx}", name) for idx, name in enumerate(self.gradients)),
                *((f"acc{idx}", name) for idx, name in enumerate(self.accumulators)),
                *((f"output{idx}", name) for idx, name in enumerate(self.outputs)),
                *(
                    (f"acc_output{idx}", name)
                    for idx, name in enumerate(self.accumulator_outputs)
                ),
            ]
        )
        rate_suffix = emitter.param_array_suffix(
            self.rate_shape, emitter.dim_names_for(self.rate)
        )
        timestep_suffix = emitter.param_array_suffix(
            self.timestep_shape, emitter.dim_names_for(self.timestep)
        )
        param_specs = [
            (params["rate"], self.rate_dtype.c_type, rate_suffix, True),
            (
                params["timestep"],
                self.timestep_dtype.c_type,
                timestep_suffix,
                True,
            ),
        ]
        tensor_specs = []
        for idx, shape in enumerate(self.output_shapes):
            input_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.inputs[idx])
            )
            grad_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.gradients[idx])
            )
            acc_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.accumulators[idx])
            )
            output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx], emitter.dim_names_for(self.outputs[idx])
            )
            acc_output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx],
                emitter.dim_names_for(self.accumulator_outputs[idx]),
            )
            param_specs.extend(
                [
                    (params[f"input{idx}"], c_type, input_suffix, True),
                    (params[f"grad{idx}"], c_type, grad_suffix, True),
                    (params[f"acc{idx}"], c_type, acc_suffix, True),
                    (params[f"output{idx}"], c_type, output_suffix, False),
                    (
                        params[f"acc_output{idx}"],
                        c_type,
                        acc_output_suffix,
                        False,
                    ),
                ]
            )
            output_dim_names = emitter.dim_names_for(self.outputs[idx])
            shape_exprs = CEmitterCompat.shape_dim_exprs(shape, output_dim_names)
            loop_vars = CEmitterCompat.loop_vars(shape)
            index_suffix = "".join(f"[{var}]" for var in loop_vars)
            tensor_specs.append(
                {
                    "shape": shape_exprs,
                    "loop_vars": loop_vars,
                    "input_expr": f"{params[f'input{idx}']}{index_suffix}",
                    "grad_expr": f"{params[f'grad{idx}']}{index_suffix}",
                    "acc_expr": f"{params[f'acc{idx}']}{index_suffix}",
                    "output_expr": f"{params[f'output{idx}']}{index_suffix}",
                    "acc_output_expr": f"{params[f'acc_output{idx}']}{index_suffix}",
                }
            )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["adagrad"]
            .render(
                model_name=model.name,
                op_name=op_name,
                rate=params["rate"],
                timestep=params["timestep"],
                params=param_decls,
                c_type=c_type,
                one_literal=emitter.format_literal(self.dtype, 1),
                decay_factor_literal=emitter.format_floating(
                    self.decay_factor, self.dtype
                ),
                norm_coefficient_literal=emitter.format_floating(
                    self.norm_coefficient, self.dtype
                ),
                epsilon_literal=emitter.format_floating(self.epsilon, self.dtype),
                dtype=self.dtype,
                tensors=tensor_specs,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs = [
            (name, shape, self.dtype)
            for name, shape in zip(self.outputs, self.output_shapes)
        ]
        outputs.extend(
            (name, shape, self.dtype)
            for name, shape in zip(self.accumulator_outputs, self.output_shapes)
        )
        return tuple(outputs)


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

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.outputs[0])
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("rate", self.rate),
                ("timestep", self.timestep),
                *((f"input{idx}", name) for idx, name in enumerate(self.inputs)),
                *((f"grad{idx}", name) for idx, name in enumerate(self.gradients)),
                *((f"vel{idx}", name) for idx, name in enumerate(self.velocities)),
                *((f"output{idx}", name) for idx, name in enumerate(self.outputs)),
                *(
                    (f"vel_output{idx}", name)
                    for idx, name in enumerate(self.velocity_outputs)
                ),
            ]
        )
        rate_suffix = emitter.param_array_suffix(
            self.rate_shape, emitter.dim_names_for(self.rate)
        )
        timestep_suffix = emitter.param_array_suffix(
            self.timestep_shape, emitter.dim_names_for(self.timestep)
        )
        param_specs = [
            (params["rate"], self.rate_dtype.c_type, rate_suffix, True),
            (
                params["timestep"],
                self.timestep_dtype.c_type,
                timestep_suffix,
                True,
            ),
        ]
        tensor_specs = []
        for idx, shape in enumerate(self.output_shapes):
            input_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.inputs[idx])
            )
            grad_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.gradients[idx])
            )
            vel_suffix = emitter.param_array_suffix(
                self.tensor_shapes[idx], emitter.dim_names_for(self.velocities[idx])
            )
            output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx], emitter.dim_names_for(self.outputs[idx])
            )
            vel_output_suffix = emitter.param_array_suffix(
                self.output_shapes[idx],
                emitter.dim_names_for(self.velocity_outputs[idx]),
            )
            param_specs.extend(
                [
                    (params[f"input{idx}"], c_type, input_suffix, True),
                    (params[f"grad{idx}"], c_type, grad_suffix, True),
                    (params[f"vel{idx}"], c_type, vel_suffix, True),
                    (params[f"output{idx}"], c_type, output_suffix, False),
                    (
                        params[f"vel_output{idx}"],
                        c_type,
                        vel_output_suffix,
                        False,
                    ),
                ]
            )
            output_dim_names = emitter.dim_names_for(self.outputs[idx])
            shape_exprs = CEmitterCompat.shape_dim_exprs(shape, output_dim_names)
            loop_vars = CEmitterCompat.loop_vars(shape)
            index_suffix = "".join(f"[{var}]" for var in loop_vars)
            tensor_specs.append(
                {
                    "shape": shape_exprs,
                    "loop_vars": loop_vars,
                    "input_expr": f"{params[f'input{idx}']}{index_suffix}",
                    "grad_expr": f"{params[f'grad{idx}']}{index_suffix}",
                    "vel_expr": f"{params[f'vel{idx}']}{index_suffix}",
                    "output_expr": f"{params[f'output{idx}']}{index_suffix}",
                    "vel_output_expr": f"{params[f'vel_output{idx}']}{index_suffix}",
                }
            )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["momentum"]
            .render(
                model_name=model.name,
                op_name=op_name,
                rate=params["rate"],
                timestep=params["timestep"],
                params=param_decls,
                c_type=c_type,
                one_literal=emitter.format_literal(self.dtype, 1),
                norm_coefficient_literal=emitter.format_floating(
                    self.norm_coefficient, self.dtype
                ),
                alpha_literal=emitter.format_floating(self.alpha, self.dtype),
                beta_literal=emitter.format_floating(self.beta, self.dtype),
                mode=self.mode,
                tensors=tensor_specs,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs = [
            (name, shape, self.dtype)
            for name, shape in zip(self.outputs, self.output_shapes)
        ]
        outputs.extend(
            (name, shape, self.dtype)
            for name, shape in zip(self.velocity_outputs, self.output_shapes)
        )
        return tuple(outputs)


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

    _INT_TYPES = frozenset(
        {ScalarType.I64, ScalarType.I32, ScalarType.I16, ScalarType.I8}
    )

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = set()
        if self.dtype.is_float:
            includes.add("#include <math.h>")
        if self.dtype in self._INT_TYPES:
            includes.add("#include <limits.h>")
        return includes

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        if self.indices_dtype is not None:
            return {self.indices_dtype}
        return set()

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        min_literal = output_dtype.min_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output", self.output),
                ("indices", self.indices),
            ]
        )
        input_shape = (self.batch, self.channels, *self.in_spatial)
        output_shape = (self.batch, self.channels, *self.out_spatial)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        indices_c_type = (
            self.indices_dtype.c_type
            if self.indices is not None and self.indices_dtype is not None
            else None
        )
        input_shape_expr = CEmitterCompat.shape_dim_exprs(input_shape, input_dim_names)
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        indices_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
                (
                    (
                        params["indices"],
                        indices_c_type or ScalarType.I64.c_type,
                        indices_suffix,
                        False,
                    )
                    if params["indices"]
                    else (None, "", "", False)
                ),
            ]
        )
        rendered = (
            state.templates["maxpool"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                indices=params["indices"],
                params=param_decls,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                indices_suffix=indices_suffix,
                indices_c_type=indices_c_type,
                dtype=self.dtype,
                batch=input_shape_expr[0],
                channels=input_shape_expr[1],
                spatial_rank=self.spatial_rank,
                in_spatial=input_shape_expr[2:],
                out_spatial=output_shape_expr[2:],
                kernel_shape=self.kernel_shape,
                strides=self.strides,
                pads=self.pads,
                dilations=self.dilations,
                ceil_mode=int(self.ceil_mode),
                storage_order=self.storage_order,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, self.channels, *self.out_spatial)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        shape = self.computed_output_shape(emitter)
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (self.output, shape, self.dtype)
        ]
        if self.indices is not None and self.indices_dtype is not None:
            outputs.append((self.indices, shape, self.indices_dtype))
        return tuple(outputs)


@dataclass(frozen=True)
class NhwcMaxPoolOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
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
    dtype: ScalarType

    _INT_TYPES = frozenset(
        {ScalarType.I64, ScalarType.I32, ScalarType.I16, ScalarType.I8}
    )

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = set()
        if self.dtype.is_float:
            includes.add("#include <math.h>")
        if self.dtype in self._INT_TYPES:
            includes.add("#include <limits.h>")
        return includes

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        min_literal = output_dtype.min_literal
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output", self.output),
            ]
        )
        input_shape = (self.batch, *self.in_spatial, self.channels)
        output_shape = (self.batch, *self.out_spatial, self.channels)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape_expr = CEmitterCompat.shape_dim_exprs(input_shape, input_dim_names)
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["nhwc_maxpool"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                dtype=self.dtype,
                batch=input_shape_expr[0],
                channels=input_shape_expr[-1],
                spatial_rank=self.spatial_rank,
                in_spatial=input_shape_expr[1:-1],
                out_spatial=output_shape_expr[1:-1],
                kernel_shape=self.kernel_shape,
                strides=self.strides,
                pads=self.pads,
                dilations=self.dilations,
                ceil_mode=int(self.ceil_mode),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return (self.batch, *self.out_spatial, self.channels)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        shape = self.computed_output_shape(emitter)
        return ((self.output, shape, self.dtype),)


@dataclass(frozen=True)
class MaxUnpoolOp(RenderableOpBase):
    __io_inputs__ = ("input0", "indices")
    __io_outputs__ = ("output",)
    input0: str
    indices: str
    output_shape: str | None
    output: str
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    inferred_out_spatial: tuple[int, ...]
    dtype: ScalarType
    indices_dtype: ScalarType

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        indices_c_type = self.indices_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("indices", self.indices),
                ("output", self.output),
            ]
        )
        input_shape = (self.batch, self.channels, *self.in_spatial)
        output_shape = (self.batch, self.channels, *self.out_spatial)
        input_suffix = emitter.param_array_suffix(input_shape)
        indices_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["indices"], indices_c_type, indices_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["maxunpool"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                indices=params["indices"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=output_dtype.zero_literal,
                indices_c_type=indices_c_type,
                dtype=self.dtype,
                batch=self.batch,
                channels=self.channels,
                spatial_rank=self.spatial_rank,
                in_spatial=self.in_spatial,
                out_spatial=self.out_spatial,
                inferred_out_spatial=self.inferred_out_spatial,
                input_suffix=input_suffix,
                indices_suffix=indices_suffix,
                output_suffix=output_suffix,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        shape = (self.batch, self.channels, *self.out_spatial)
        return ((self.output, shape, self.dtype),)


@dataclass(frozen=True)
class RoiAlignOp(RenderableOpBase):
    __io_inputs__ = ("input0", "rois", "batch_indices")
    __io_outputs__ = ("output",)
    input0: str
    rois: str
    batch_indices: str
    output: str
    num_rois: int
    channels: int
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    sampling_ratio: int
    spatial_scale: float
    mode: str
    coordinate_transformation_mode: str
    dtype: ScalarType

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        output_dtype = emitter.ctx_dtype(self.output)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("rois", self.rois),
                ("batch_indices", self.batch_indices),
                ("output", self.output),
            ]
        )
        input_shape = emitter.ctx_shape(self.input0)
        rois_shape = emitter.ctx_shape(self.rois)
        batch_indices_shape = emitter.ctx_shape(self.batch_indices)
        output_shape = emitter.ctx_shape(self.output)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    c_type,
                    emitter.param_array_suffix(input_shape),
                    True,
                ),
                (
                    params["rois"],
                    c_type,
                    emitter.param_array_suffix(rois_shape),
                    True,
                ),
                (
                    params["batch_indices"],
                    ScalarType.I64.c_type,
                    emitter.param_array_suffix(batch_indices_shape),
                    True,
                ),
                (
                    params["output"],
                    c_type,
                    emitter.param_array_suffix(output_shape),
                    False,
                ),
            ]
        )
        rendered = (
            state.templates["roi_align"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                input0=params["input0"],
                rois=params["rois"],
                batch_indices=params["batch_indices"],
                output=params["output"],
                c_type=c_type,
                dtype=self.dtype,
                num_rois=self.num_rois,
                channels=self.channels,
                input_height=self.input_height,
                input_width=self.input_width,
                output_height=self.output_height,
                output_width=self.output_width,
                sampling_ratio=self.sampling_ratio,
                spatial_scale=emitter.format_double(self.spatial_scale),
                mode=self.mode,
                coordinate_transformation_mode=self.coordinate_transformation_mode,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)
