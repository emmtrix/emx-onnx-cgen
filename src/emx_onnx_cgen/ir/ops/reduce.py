from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..op_base import CEmitterCompat, Emitter, EmitContext, ReduceOpBase
from ..op_context import OpContext

from ...errors import CodegenError


@dataclass(frozen=True)
class ReduceOp(ReduceOpBase):
    __io_inputs__ = ("input0", "axes_input")
    input0: str
    output: str
    axes: tuple[int, ...]
    axes_input: str | None
    keepdims: bool
    noop_with_empty_axes: bool
    reduce_kind: str
    reduce_count: int | None

    _INT_TYPES = frozenset(
        {ScalarType.I64, ScalarType.I32, ScalarType.I16, ScalarType.I8}
    )

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = set()
        if self.axes_input is not None:
            includes.add("#include <stdbool.h>")
        if self.reduce_kind in {"l1", "l2", "logsum", "logsumexp"}:
            includes.add("#include <math.h>")
        if self.reduce_kind in {"min", "max"}:
            output_dtype = ctx.dtype(self.output)
            if output_dtype.is_float:
                includes.add("#include <math.h>")
            if output_dtype in self._INT_TYPES:
                includes.add("#include <limits.h>")
        return includes

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if self.axes_input is None:
            axes = self.normalize_axes(self.axes, len(input_shape))
            output_shape = self.reduced_shape(input_shape, axes, keepdims=self.keepdims)
        else:
            axes = self.axes
            output_shape = ctx.shape(self.output)
        ctx.set_shape(self.output, output_shape)
        ctx.set_derived(self, "axes", axes)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, emitter.ctx_shape(self.input0)),)


@dataclass(frozen=True)
class DetOp(ReduceOpBase):
    input0: str
    output: str

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if len(input_shape) < 2:
            raise CodegenError(f"Det expects rank >= 2 input, got shape {input_shape}")
        if input_shape[-1] != input_shape[-2]:
            raise CodegenError(f"Det expects square matrices, got shape {input_shape}")
        ctx.set_shape(self.output, input_shape[:-2])

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        if input_dtype != output_dtype:
            raise CodegenError(
                f"Det expects matching input/output dtypes, got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
            )
        matrix_dim = input_shape_raw[-1]
        if matrix_dim < 0:
            raise CodegenError("Det requires a static matrix dimension")
        batch_shape = input_shape_raw[:-2]
        batch_shape_codegen = CEmitterCompat.codegen_shape(batch_shape)
        batch_loop_vars = CEmitterCompat.loop_vars(batch_shape_codegen)
        input_batch_index_expr = (
            "".join(f"[{var}]" for var in batch_loop_vars) if batch_shape else ""
        )
        batch_index_expr = "".join(f"[{var}]" for var in batch_loop_vars)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_suffix = emitter.param_array_suffix(input_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["det"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                input0=params["input0"],
                output=params["output"],
                c_type=input_dtype.c_type,
                matrix_dim=matrix_dim,
                batch_shape=batch_shape_codegen,
                batch_loop_vars=batch_loop_vars,
                input_batch_index_expr=input_batch_index_expr,
                batch_index_expr=batch_index_expr,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class InverseOp(ReduceOpBase):
    input0: str
    output: str

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if len(input_shape) < 2:
            raise CodegenError(
                f"Inverse expects rank >= 2 input, got shape {input_shape}"
            )
        if input_shape[-1] != input_shape[-2]:
            raise CodegenError(
                f"Inverse expects square matrices, got shape {input_shape}"
            )
        ctx.set_shape(self.output, input_shape)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        if input_dtype != output_dtype:
            raise CodegenError(
                f"Inverse expects matching input/output dtypes, got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
            )
        matrix_dim = input_shape_raw[-1]
        if matrix_dim < 0:
            raise CodegenError("Inverse requires a static matrix dimension")
        batch_shape = input_shape_raw[:-2]
        batch_shape_codegen = CEmitterCompat.codegen_shape(batch_shape)
        batch_loop_vars = CEmitterCompat.loop_vars(batch_shape_codegen)
        input_batch_index_expr = (
            "".join(f"[{var}]" for var in batch_loop_vars) if batch_shape else ""
        )
        batch_index_expr = (
            "".join(f"[{var}]" for var in batch_loop_vars) if batch_shape else ""
        )
        # Use float32 for f16/bf16/f64 to match ORT's implementation behavior.
        compute_type = (
            "float"
            if input_dtype in {ScalarType.F16, ScalarType.BF16, ScalarType.F64}
            else input_dtype.c_type
        )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_suffix = emitter.param_array_suffix(input_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["inverse"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                input0=params["input0"],
                output=params["output"],
                c_type=input_dtype.c_type,
                compute_type=compute_type,
                matrix_dim=matrix_dim,
                batch_shape=batch_shape_codegen,
                batch_loop_vars=batch_loop_vars,
                input_batch_index_expr=input_batch_index_expr,
                batch_index_expr=batch_index_expr,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class CDistOp(ReduceOpBase):
    """Pairwise distance computation between two sets of vectors (com.microsoft::CDist)."""

    __io_inputs__ = ("input_a", "input_b")

    input_a: str
    input_b: str
    output: str
    metric: str  # "euclidean" or "sqeuclidean"

    def required_includes(self, ctx: OpContext) -> set[str]:
        if self.metric == "euclidean":
            return {"#include <math.h>"}
        return set()

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input_a)
        ctx.dtype(self.input_b)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        shape_a = ctx.shape(self.input_a)  # [M, K]
        shape_b = ctx.shape(self.input_b)  # [N, K]
        ctx.set_shape(self.output, (shape_a[0], shape_b[0]))  # [M, N]

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        shape_a = emitter.ctx_shape(self.input_a)  # [M, K]
        shape_b = emitter.ctx_shape(self.input_b)  # [N, K]
        output_shape = emitter.ctx_shape(self.output)  # [M, N]
        dtype_a = emitter.ctx_dtype(self.input_a)
        dtype_b = emitter.ctx_dtype(self.input_b)
        output_dtype = emitter.ctx_dtype(self.output)
        if shape_a[1] < 0 or shape_b[0] < 0 or shape_b[1] < 0 or shape_a[0] < 0:
            raise CodegenError(
                "CDist requires static shapes; export with static shapes"
            )
        compute_type = dtype_a.c_type
        sqrt_fn = "sqrtf" if dtype_a == ScalarType.F32 else "sqrt"
        params = emitter.shared_param_map(
            [("input_a", self.input_a), ("input_b", self.input_b), ("output", self.output)]
        )
        suffix_a = emitter.param_array_suffix(shape_a)
        suffix_b = emitter.param_array_suffix(shape_b)
        suffix_out = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input_a"], dtype_a.c_type, suffix_a, True),
                (params["input_b"], dtype_b.c_type, suffix_b, True),
                (params["output"], output_dtype.c_type, suffix_out, False),
            ]
        )
        rendered = (
            state.templates["cdist"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                input_a=params["input_a"],
                input_b=params["input_b"],
                output=params["output"],
                c_type=dtype_a.c_type,
                compute_type=compute_type,
                M=shape_a[0],
                N=shape_b[0],
                K=shape_a[1],
                metric=self.metric,
                sqrt_fn=sqrt_fn,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class DynamicTimeWarpingOp(ReduceOpBase):
    """com.microsoft::DynamicTimeWarping — DTW path computation."""

    __io_inputs__ = ("input0",)

    input0: str
    output: str
    rows: int  # M
    cols: int  # N
    path_len: int  # declared output path length

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.set_dtype(self.output, ScalarType.I32)

    def infer_shapes(self, ctx: OpContext) -> None:
        ctx.set_shape(self.output, (2, self.path_len))

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["dynamic_time_warping"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input=params["input0"],
                output=params["output"],
                cost_type=input_dtype.c_type,
                M=self.rows,
                N=self.cols,
                max_path_len=self.rows + self.cols - 1,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ArgReduceOp(ReduceOpBase):
    input0: str
    output: str
    axis: int
    keepdims: bool
    select_last_index: bool
    reduce_kind: str

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        return {ctx.dtype(self.input0), ctx.dtype(self.output)}

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axes = self.normalize_axes((self.axis,), len(input_shape))
        output_shape = self.reduced_shape(input_shape, axes, keepdims=self.keepdims)
        ctx.set_shape(self.output, output_shape)
        ctx.set_derived(self, "axis", axes[0])

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        axis = emitter.derived(self, "axis")
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        reduce_var = "r0"
        reduce_dim = input_shape[axis]
        if self.keepdims:
            input_indices = [
                reduce_var if axis_index == axis else output_loop_vars[axis_index]
                for axis_index in range(len(input_shape))
            ]
        else:
            kept_axes = [
                axis_index
                for axis_index in range(len(input_shape))
                if axis_index != axis
            ]
            input_indices = [
                (
                    reduce_var
                    if axis_index == axis
                    else output_loop_vars[kept_axes.index(axis_index)]
                )
                for axis_index in range(len(input_shape))
            ]
        init_indices = [
            "0" if axis_index == axis else input_indices[axis_index]
            for axis_index in range(len(input_shape))
        ]
        input_index_expr = "".join(f"[{var}]" for var in input_indices)
        init_index_expr = "".join(f"[{var}]" for var in init_indices)
        output_index_expr = "".join(f"[{var}]" for var in output_loop_vars)
        if self.reduce_kind == "max":
            compare_op = ">=" if self.select_last_index else ">"
        elif self.reduce_kind == "min":
            compare_op = "<=" if self.select_last_index else "<"
        else:
            raise CodegenError(f"Unsupported arg reduce kind {self.reduce_kind}")
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["arg_reduce"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_var=reduce_var,
                reduce_dim=reduce_dim,
                input_index_expr=input_index_expr,
                init_index_expr=init_index_expr,
                output_index_expr=output_index_expr,
                compare_op=compare_op,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class TopKOp(ReduceOpBase):
    __io_inputs__ = ("input0", "k_input")
    __io_outputs__ = ("output_values", "output_indices")
    input0: str
    k_input: str
    output_values: str
    output_indices: str
    axis: int
    k: int
    largest: bool
    sorted: bool

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        return {
            ctx.dtype(self.input0),
            ctx.dtype(self.output_values),
            ctx.dtype(self.output_indices),
        }

    def call_args(self) -> tuple[str, ...]:
        return (self.input0, self.output_values, self.output_indices)

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output_values)
        ctx.dtype(self.output_indices)

    def infer_shapes(self, ctx: OpContext) -> None:
        output_shape = ctx.shape(self.output_values)
        ctx.set_shape(self.output_values, output_shape)
        ctx.set_shape(self.output_indices, ctx.shape(self.output_indices))

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output_values)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_values_dtype = emitter.ctx_dtype(self.output_values)
        output_indices_dtype = emitter.ctx_dtype(self.output_indices)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output_values", self.output_values),
                ("output_indices", self.output_indices),
            ]
        )
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        outer_shape = tuple(
            dim for axis, dim in enumerate(output_shape) if axis != self.axis
        )
        outer_loop_vars = CEmitterCompat.loop_vars(outer_shape)
        reduce_var = "r0"
        k_var = "k0"
        input_indices: list[str] = []
        output_indices: list[str] = []
        outer_index = 0
        for axis in range(len(input_shape)):
            if axis == self.axis:
                input_indices.append(reduce_var)
                output_indices.append(k_var)
            else:
                input_indices.append(outer_loop_vars[outer_index])
                output_indices.append(outer_loop_vars[outer_index])
                outer_index += 1
        input_index_expr = "".join(f"[{var}]" for var in input_indices)
        output_index_expr = "".join(f"[{var}]" for var in output_indices)
        compare_expr = (
            "(a > b) || ((a == b) && (ai < bi))"
            if self.largest
            else "(a < b) || ((a == b) && (ai < bi))"
        )
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (
                    params["output_values"],
                    output_values_dtype.c_type,
                    output_suffix,
                    False,
                ),
                (
                    params["output_indices"],
                    output_indices_dtype.c_type,
                    output_suffix,
                    False,
                ),
            ]
        )
        rendered = (
            state.templates["topk"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output_values=params["output_values"],
                output_indices=params["output_indices"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                output_values_c_type=output_values_dtype.c_type,
                output_indices_c_type=output_indices_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                outer_shape=outer_shape,
                outer_loop_vars=outer_loop_vars,
                reduce_var=reduce_var,
                k_var=k_var,
                axis_dim=input_shape[self.axis],
                k=self.k,
                input_index_expr=input_index_expr,
                output_index_expr=output_index_expr,
                compare_expr=compare_expr,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        shape = self.computed_output_shape(emitter)
        return (
            (self.output_values, shape, emitter.ctx_dtype(self.output_values)),
            (self.output_indices, shape, emitter.ctx_dtype(self.output_indices)),
        )
