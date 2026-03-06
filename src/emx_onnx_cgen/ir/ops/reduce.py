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
