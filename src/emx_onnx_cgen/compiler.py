from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Callable, Mapping, TypeVar

import numpy as np
import onnx

from shared.scalar_types import ScalarType

from .onnxruntime_utils import make_deterministic_session_options
from .codegen.c_emitter import (
    CEmitter,
    ConstTensor,
    LoweredModel,
    ModelHeader,
    NodeInfo,
)
from .errors import ShapeInferenceError, UnsupportedOpError
from .invariants import (
    check_graph_integrity,
    check_inferred_shapes,
    check_inferred_types,
    check_lowered_ops,
)
from .ir.context import GraphContext
from .ir.model import (
    Graph,
    Initializer,
    Node,
    SequenceType,
    TensorType,
    Value,
    ValueType,
)
from .ir.op_base import OpBase
from .ir.op_context import OpContext
from .lowering import load_lowering_registry
from .lowering.common import ensure_supported_dtype, shape_product
from .lowering.registry import get_lowering_registry
from .onnx_import import import_onnx, prepare_onnx_model


def _value_requires_explicit_shape_concretization(
    value: Value,
    *,
    allow_top_level_tensor_dims: bool,
    check_sequence_dims: bool,
    allowed_tensor_dim_params: set[str] | None = None,
) -> str | None:
    value_type = value.type
    if isinstance(value_type, SequenceType):
        if check_sequence_dims and any(
            dim_param is not None for dim_param in value_type.elem.dim_params
        ):
            return (
                f"sequence '{value.name}' has dynamic element dimensions "
                f"{value_type.elem.dim_params}"
            )
        return None
    if allow_top_level_tensor_dims:
        return None
    unresolved_dim_params = tuple(
        dim_param
        for dim_param in value_type.dim_params
        if dim_param is not None
        and (allowed_tensor_dim_params is None or dim_param not in allowed_tensor_dim_params)
    )
    if unresolved_dim_params:
        return f"tensor '{value.name}' has dynamic dimensions {value_type.dim_params}"
    return None


@dataclass(frozen=True)
class CompilerOptions:
    template_dir: Path | None = None
    model_name: str = "model"
    emit_testbench: bool = False
    command_line: str | None = None
    model_checksum: str | None = None
    restrict_arrays: bool = True
    fp32_accumulation_strategy: str = "simple"
    fp16_accumulation_strategy: str = "fp32"
    testbench_optional_inputs: Mapping[str, bool] | None = None
    testbench_output_format: str = "json"
    shape_inference_inputs: Mapping[str, np.ndarray] | None = None
    truncate_weights_after: int | None = None
    large_temp_threshold_bytes: int = 1024
    large_weight_threshold: int = 100 * 1024
    replicate_ort_bugs: bool = False
    timings: dict[str, float] | None = None


_T = TypeVar("_T")


def _onnx_elem_type(dtype: np.dtype) -> int:
    for elem_type, info in onnx._mapping.TENSOR_TYPE_MAP.items():
        if info.np_dtype == dtype:
            return elem_type
    raise UnsupportedOpError(f"Unsupported dtype {dtype} for ONNX output")


def _optional_flag_name(name: str) -> str:
    return f"{name}_present"


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions()
        self._options = options
        self._emitter = CEmitter(
            options.template_dir,
            restrict_arrays=options.restrict_arrays,
            fp32_accumulation_strategy=options.fp32_accumulation_strategy,
            fp16_accumulation_strategy=options.fp16_accumulation_strategy,
            truncate_weights_after=options.truncate_weights_after,
            large_temp_threshold_bytes=options.large_temp_threshold_bytes,
            large_weight_threshold=options.large_weight_threshold,
            replicate_ort_bugs=options.replicate_ort_bugs,
        )
        load_lowering_registry()

    @dataclass(frozen=True)
    class _CompileContext:
        lowered: LoweredModel
        variable_dim_inputs: dict[int, dict[int, str]]
        variable_dim_outputs: dict[int, dict[int, str]]

    def _try_lower_without_shape_concretization(
        self,
        model: onnx.ModelProto,
        graph: Graph,
    ) -> LoweredModel | None:
        try:
            return self._lower_model(model, graph)
        except (ShapeInferenceError, UnsupportedOpError):
            return None

    def _time_step(self, label: str, func: Callable[[], _T]) -> _T:
        timings = self._options.timings
        if timings is None:
            return func()
        started = time.perf_counter()
        result = func()
        timings[label] = time.perf_counter() - started
        return result

    def _build_compile_context(self, model: onnx.ModelProto) -> _CompileContext:
        prepared_model = self._time_step(
            "prepare_onnx_model", lambda: prepare_onnx_model(model)
        )
        graph = self._time_step(
            "import_onnx",
            lambda: import_onnx(prepared_model, _prepared=True),
        )
        missing_shape_reason = self._shape_concretization_requirement_reason(graph)
        if (
            missing_shape_reason is not None
            and not self._options.shape_inference_inputs
        ):
            lowered = self._time_step(
                "lower_model_without_concretize",
                lambda: self._try_lower_without_shape_concretization(
                    prepared_model, graph
                ),
            )
            if lowered is not None:
                variable_dim_inputs, variable_dim_outputs = self._time_step(
                    "collect_variable_dims", lambda: self._collect_variable_dims(graph)
                )
                return Compiler._CompileContext(
                    lowered=lowered,
                    variable_dim_inputs=variable_dim_inputs,
                    variable_dim_outputs=variable_dim_outputs,
                )
            raise UnsupportedOpError(
                "Code generation needs explicit shape concretization, but no "
                "--shape-inference-shapes were provided. "
                f"Reason: {missing_shape_reason}. "
                "Hint: pass --shape-inference-shapes with explicit input specs "
                "(for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, "
                "or export the model with static shapes."
            )
        graph = self._time_step(
            "concretize_shapes",
            lambda: self._concretize_graph_shapes(prepared_model, graph),
        )
        unresolved_shape_reason = self._unresolved_shape_concretization_reason(graph)
        if unresolved_shape_reason is not None:
            raise UnsupportedOpError(
                "Code generation still has unresolved dynamic shapes after "
                "shape concretization. "
                f"Reason: {unresolved_shape_reason}. "
                "Hint: provide more representative --shape-inference-shapes or "
                "export the model with static shapes."
            )
        variable_dim_inputs, variable_dim_outputs = self._time_step(
            "collect_variable_dims", lambda: self._collect_variable_dims(graph)
        )
        lowered = self._time_step(
            "lower_model", lambda: self._lower_model(prepared_model, graph)
        )
        return Compiler._CompileContext(
            lowered=lowered,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )

    def compile(self, model: onnx.ModelProto) -> str:
        ctx = self._build_compile_context(model)
        return self._time_step(
            "emit_model",
            lambda: self._emitter.emit_model(
                ctx.lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                testbench_output_format=self._options.testbench_output_format,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )

    def compile_testbench(self, model: onnx.ModelProto) -> str:
        ctx = self._build_compile_context(model)
        return self._time_step(
            "emit_testbench",
            lambda: self._emitter.emit_testbench(
                ctx.lowered,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                testbench_output_format=self._options.testbench_output_format,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )

    def compile_testbench_declarations(self, model: onnx.ModelProto) -> str:
        ctx = self._build_compile_context(model)
        return self._time_step(
            "emit_testbench_decls",
            lambda: self._emitter.emit_testbench_declarations(
                ctx.lowered,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        ctx = self._build_compile_context(model)
        return self._time_step(
            "emit_model_with_data_file",
            lambda: self._emitter.emit_model_with_data_file(
                ctx.lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                testbench_output_format=self._options.testbench_output_format,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )

    def compile_with_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, bytes | None]:
        ctx = self._build_compile_context(model)
        generated = self._time_step(
            "emit_model",
            lambda: self._emitter.emit_model(
                ctx.lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                testbench_output_format=self._options.testbench_output_format,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )
        weight_data = self._time_step(
            "collect_weight_data",
            lambda: self._emitter.collect_weight_data(ctx.lowered.constants),
        )
        return generated, weight_data

    def compile_with_data_file_and_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, str, bytes | None]:
        ctx = self._build_compile_context(model)
        generated, data_source = self._time_step(
            "emit_model_with_data_file",
            lambda: self._emitter.emit_model_with_data_file(
                ctx.lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                testbench_output_format=self._options.testbench_output_format,
                variable_dim_inputs=ctx.variable_dim_inputs,
                variable_dim_outputs=ctx.variable_dim_outputs,
            ),
        )
        weight_data = self._time_step(
            "collect_weight_data",
            lambda: self._emitter.collect_weight_data(ctx.lowered.constants),
        )
        return generated, data_source, weight_data

    @staticmethod
    def _shape_concretization_requirement_reason(graph: Graph) -> str | None:
        allowed_internal_dim_params = Compiler._allowed_internal_dim_params(graph)
        for value in graph.inputs:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=True,
                check_sequence_dims=True,
            )
            if reason is not None:
                return reason
        for value in graph.values:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=False,
                check_sequence_dims=True,
                allowed_tensor_dim_params=allowed_internal_dim_params,
            )
            if reason is not None:
                return reason
        for value in graph.outputs:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=True,
                check_sequence_dims=True,
            )
            if reason is not None:
                return reason
        return None

    @staticmethod
    def _unresolved_shape_concretization_reason(graph: Graph) -> str | None:
        allowed_internal_dim_params = Compiler._allowed_internal_dim_params(graph)
        for value in graph.inputs:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=True,
                check_sequence_dims=True,
            )
            if reason is not None:
                return reason
        for value in graph.values:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=False,
                check_sequence_dims=False,
                allowed_tensor_dim_params=allowed_internal_dim_params,
            )
            if reason is not None:
                return reason
        for value in graph.outputs:
            reason = _value_requires_explicit_shape_concretization(
                value,
                allow_top_level_tensor_dims=True,
                check_sequence_dims=False,
            )
            if reason is not None:
                return reason
        return None

    @staticmethod
    def _allowed_internal_dim_params(graph: Graph) -> set[str]:
        dim_params: set[str] = set()
        for value in graph.inputs + graph.outputs:
            if isinstance(value.type, TensorType):
                dim_params.update(
                    dim_param for dim_param in value.type.dim_params if dim_param
                )
            elif isinstance(value.type, SequenceType):
                dim_params.update(
                    dim_param for dim_param in value.type.elem.dim_params if dim_param
                )
        return dim_params

    @classmethod
    def requires_explicit_shape_inference_inputs(cls, model: onnx.ModelProto) -> bool:
        prepared_model = prepare_onnx_model(model)
        graph = import_onnx(prepared_model, _prepared=True)
        if cls._shape_concretization_requirement_reason(graph) is None:
            return False
        compiler = cls(CompilerOptions())
        return compiler._try_lower_without_shape_concretization(
            prepared_model, graph
        ) is None

    @staticmethod
    def _collect_variable_dims(
        graph: Graph,
    ) -> tuple[dict[int, dict[int, str]], dict[int, dict[int, str]]]:
        def collect(values: tuple[Value, ...]) -> dict[int, dict[int, str]]:
            dim_map: dict[int, dict[int, str]] = {}
            for index, value in enumerate(values):
                if not isinstance(value.type, TensorType):
                    continue
                dims = {
                    dim_index: dim_param
                    for dim_index, dim_param in enumerate(value.type.dim_params)
                    if dim_param
                }
                if dims:
                    dim_map[index] = dims
            return dim_map

        return collect(graph.inputs), collect(graph.outputs)

    def _lower_model(self, model: onnx.ModelProto, graph: Graph) -> LoweredModel:
        graph = self._materialize_initializer_outputs(graph)
        ctx = GraphContext(graph)
        constants = _lowered_constants(ctx)
        self._validate_graph(graph)
        (
            input_names,
            input_optional_names,
            input_shapes,
            input_dtypes,
            input_types,
            output_names,
            output_optional_names,
            output_shapes,
            output_dtypes,
            output_types,
        ) = self._collect_io_specs(graph)
        ops, node_infos = self._lower_nodes(ctx)
        check_lowered_ops(ops)
        op_ctx = OpContext(ctx)
        for op in ops:
            op.validate(op_ctx)
        for op in ops:
            op.infer_types(op_ctx)
        check_inferred_types(ops, op_ctx)
        for op in ops:
            op.infer_shapes(op_ctx)
        check_inferred_shapes(ops, op_ctx)
        refreshed_output_types: list[ValueType] = []
        refreshed_output_shapes: list[tuple[int, ...]] = []
        refreshed_output_dtypes: list[ScalarType] = []
        for value, original_type, original_shape, original_dtype in zip(
            graph.outputs,
            output_types,
            output_shapes,
            output_dtypes,
        ):
            if isinstance(original_type, TensorType):
                inferred_shape = op_ctx.shape(value.name)
                inferred_dtype = op_ctx.dtype(value.name)
                refreshed_output_types.append(
                    TensorType(
                        dtype=inferred_dtype,
                        shape=inferred_shape,
                        dim_params=(None,) * len(inferred_shape),
                        is_optional=original_type.is_optional,
                    )
                )
                refreshed_output_shapes.append(inferred_shape)
                refreshed_output_dtypes.append(inferred_dtype)
                continue
            refreshed_output_types.append(original_type)
            refreshed_output_shapes.append(original_shape)
            refreshed_output_dtypes.append(original_dtype)
        output_types = tuple(refreshed_output_types)
        output_shapes = tuple(refreshed_output_shapes)
        output_dtypes = tuple(refreshed_output_dtypes)
        header = self._build_header(model, graph)
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            input_optional_names=input_optional_names,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            input_types=input_types,
            output_names=output_names,
            output_optional_names=output_optional_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            output_types=output_types,
            constants=constants,
            ops=tuple(ops),
            node_infos=tuple(node_infos),
            header=header,
            op_context=op_ctx,
        )

    def _materialize_initializer_outputs(self, graph: Graph) -> Graph:
        if graph.nodes:
            return graph
        if not graph.outputs:
            return graph

        initializer_by_name = {
            initializer.name: initializer for initializer in graph.initializers
        }
        output_names = {value.name for value in graph.outputs}
        if not all(name in initializer_by_name for name in output_names):
            return graph

        reserved_names = {
            *(value.name for value in graph.inputs),
            *(value.name for value in graph.outputs),
            *(value.name for value in graph.values),
            *(initializer.name for initializer in graph.initializers),
        }
        renamed_initializers: list[Initializer] = []
        initializer_name_map: dict[str, str] = {}
        for initializer in graph.initializers:
            if initializer.name not in output_names:
                renamed_initializers.append(initializer)
                continue
            base_name = f"{initializer.name}__const"
            new_name = base_name
            suffix = 0
            while new_name in reserved_names:
                suffix += 1
                new_name = f"{base_name}_{suffix}"
            reserved_names.add(new_name)
            initializer_name_map[initializer.name] = new_name
            renamed_initializers.append(
                Initializer(
                    name=new_name,
                    type=initializer.type,
                    data=initializer.data,
                )
            )

        identity_nodes = tuple(
            Node(
                op_type="Identity",
                name=f"materialize_output_{index}",
                inputs=(initializer_name_map[output.name],),
                outputs=(output.name,),
                attrs={},
            )
            for index, output in enumerate(graph.outputs)
        )

        return Graph(
            inputs=graph.inputs,
            outputs=graph.outputs,
            nodes=identity_nodes,
            initializers=tuple(renamed_initializers),
            values=graph.values,
            opset_imports=graph.opset_imports,
        )

    def _concretize_graph_shapes(self, model: onnx.ModelProto, graph: Graph) -> Graph:
        shape_inputs = self._options.shape_inference_inputs
        if not shape_inputs:
            return graph
        if not any(
            (isinstance(value.type, TensorType) and bool(value.type.dim_params))
            or (
                isinstance(value.type, SequenceType)
                and bool(value.type.elem.dim_params)
            )
            for value in graph.values + graph.inputs + graph.outputs
        ):
            return graph
        try:
            import onnxruntime as ort
        except Exception:
            return graph
        has_sequence_insert = any(
            node.op_type == "SequenceInsert" for node in graph.nodes
        )
        sequence_elem_shapes_by_name: dict[str, tuple[int, ...]] = {}
        if not has_sequence_insert:
            for value in graph.inputs:
                if not isinstance(value.type, SequenceType):
                    continue
                array = shape_inputs.get(value.name)
                if not isinstance(array, np.ndarray) or array.ndim < 1:
                    continue
                sequence_elem_shapes_by_name[value.name] = tuple(
                    int(dim) for dim in array.shape[1:]
                )

        ort_inputs: dict[str, np.ndarray] = {}
        output_names: list[str] = []
        output_arrays: list[object] = []
        try:
            model_with_outputs = onnx.ModelProto()
            model_with_outputs.CopyFrom(model)
            existing_outputs = {
                output.name for output in model_with_outputs.graph.output
            }
            value_info_by_name = {
                value_info.name: value_info
                for value_info in model_with_outputs.graph.value_info
            }
            for value in graph.values:
                if value.name in existing_outputs:
                    continue
                value_info = value_info_by_name.get(value.name)
                if value_info is None:
                    dims: list[int | str | None] = []
                    if not isinstance(value.type, TensorType):
                        continue
                    for index, dim in enumerate(value.type.shape):
                        dim_param = None
                        if index < len(value.type.dim_params):
                            dim_param = value.type.dim_params[index]
                        dims.append(dim_param if dim_param else None)
                    elem_type = _onnx_elem_type(value.type.dtype.np_dtype)
                    value_info = onnx.helper.make_tensor_value_info(
                        value.name, elem_type, dims
                    )
                model_with_outputs.graph.output.append(value_info)
                existing_outputs.add(value.name)
            output_names = [output.name for output in model_with_outputs.graph.output]
            sess_options = make_deterministic_session_options(ort)
            sess = ort.InferenceSession(
                model_with_outputs.SerializeToString(),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            graph_tensor_inputs = {
                value.name
                for value in graph.inputs
                if isinstance(value.type, TensorType)
            }
            ort_inputs = {
                name: array
                for name, array in shape_inputs.items()
                if name in graph_tensor_inputs and isinstance(array, np.ndarray)
            }
            if ort_inputs:
                output_arrays = sess.run(None, ort_inputs)
            else:
                output_arrays = []
                output_names = []
        except Exception:
            if not sequence_elem_shapes_by_name:
                return graph

        shapes_by_name: dict[str, tuple[int, ...]] = {}
        for name, array in zip(output_names, output_arrays):
            if isinstance(array, np.ndarray):
                shapes_by_name[name] = tuple(int(dim) for dim in array.shape)
        for name, array in ort_inputs.items():
            if isinstance(array, np.ndarray):
                shapes_by_name[name] = tuple(int(dim) for dim in array.shape)

        def concretize_value(value: Value) -> Value:
            if not isinstance(value.type, TensorType):
                if isinstance(value.type, SequenceType):
                    elem_shape = sequence_elem_shapes_by_name.get(value.name)
                    if elem_shape is None:
                        return value
                    elem = value.type.elem
                    return Value(
                        name=value.name,
                        type=SequenceType(
                            elem=TensorType(
                                dtype=elem.dtype,
                                shape=elem_shape,
                                dim_params=(None,) * len(elem_shape),
                                is_optional=elem.is_optional,
                            ),
                            is_optional=value.type.is_optional,
                        ),
                    )
                return value
            shape = shapes_by_name.get(value.name)
            if shape is None:
                return value
            return Value(
                name=value.name,
                type=TensorType(
                    dtype=value.type.dtype,
                    shape=shape,
                    dim_params=(None,) * len(shape),
                    is_optional=value.type.is_optional,
                ),
            )

        return Graph(
            inputs=tuple(concretize_value(value) for value in graph.inputs),
            outputs=tuple(concretize_value(value) for value in graph.outputs),
            nodes=graph.nodes,
            initializers=graph.initializers,
            values=tuple(concretize_value(value) for value in graph.values),
            opset_imports=graph.opset_imports,
        )

    def _validate_graph(self, graph: Graph) -> None:
        check_graph_integrity(graph)
        for value in graph.outputs:
            if isinstance(value.type, TensorType):
                shape_product(value.type.shape)

    def _collect_io_specs(
        self, graph: Graph
    ) -> tuple[
        tuple[str, ...],
        tuple[str | None, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
        tuple[ValueType, ...],
        tuple[str, ...],
        tuple[str | None, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
        tuple[ValueType, ...],
    ]:
        def tensor_type(value_name: str, value_type: ValueType) -> TensorType:
            if isinstance(value_type, TensorType):
                return value_type
            elem = value_type.elem
            if elem.shape and all(dim_param is None for dim_param in elem.dim_params):
                return elem

            def _split_chunk_size(split_name: str | None, axis_size: int) -> int:
                if split_name is None:
                    return 1
                initializer = next(
                    (item for item in graph.initializers if item.name == split_name),
                    None,
                )
                if initializer is None:
                    # Runtime-provided split can span up to the full axis extent.
                    return axis_size
                values = initializer.data.reshape(-1).tolist()
                if not values:
                    return axis_size
                if initializer.data.ndim == 0:
                    scalar_split = int(values[0])
                    if scalar_split <= 0:
                        return axis_size
                    return min(axis_size, scalar_split)
                chunk_sizes = [int(size) for size in values if int(size) > 0]
                if not chunk_sizes:
                    return axis_size
                return min(axis_size, max(chunk_sizes))

            if isinstance(value_type, SequenceType):
                for node in graph.nodes:
                    if node.op_type != "SplitToSequence":
                        continue
                    if value_name != node.outputs[0]:
                        continue
                    input_name = node.inputs[0]
                    try:
                        input_value = graph.find_value(input_name)
                    except KeyError:
                        continue
                    if not isinstance(input_value.type, TensorType):
                        continue
                    input_shape = input_value.type.shape
                    if not input_shape:
                        continue
                    axis = int(node.attrs.get("axis", 0))
                    if axis < 0:
                        axis += len(input_shape)
                    if axis < 0 or axis >= len(input_shape):
                        continue
                    keepdims = bool(int(node.attrs.get("keepdims", 1)))
                    split_name = None
                    if len(node.inputs) > 1 and node.inputs[1]:
                        split_name = node.inputs[1]
                        keepdims = True
                    if keepdims:
                        output_shape = list(input_shape)
                        output_shape[axis] = _split_chunk_size(
                            split_name, input_shape[axis]
                        )
                    else:
                        output_shape = list(input_shape)
                        del output_shape[axis]
                    return TensorType(
                        dtype=elem.dtype,
                        shape=tuple(output_shape),
                        dim_params=(None,) * len(output_shape),
                        is_optional=elem.is_optional,
                    )

            for node in graph.nodes:
                if node.op_type != "SequenceInsert":
                    continue
                if value_name in {node.inputs[0], node.outputs[0]}:
                    tensor_name = node.inputs[1]
                    try:
                        tensor_value = graph.find_value(tensor_name)
                    except KeyError:
                        continue
                    if isinstance(tensor_value.type, TensorType):
                        return TensorType(
                            dtype=elem.dtype,
                            shape=tensor_value.type.shape,
                            dim_params=tensor_value.type.dim_params,
                            is_optional=elem.is_optional,
                        )
            return elem

        input_names = tuple(value.name for value in graph.inputs)

        def is_optional_value_type(value_type: ValueType) -> bool:
            return bool(getattr(value_type, "is_optional", False))

        input_optional_names = tuple(
            (
                _optional_flag_name(value.name)
                if is_optional_value_type(value.type)
                else None
            )
            for value in graph.inputs
        )
        input_types = tuple(value.type for value in graph.inputs)
        input_shapes = tuple(
            tensor_type(value.name, value.type).shape for value in graph.inputs
        )
        input_dtypes = tuple(
            tensor_type(value.name, value.type).dtype for value in graph.inputs
        )
        output_names = tuple(value.name for value in graph.outputs)
        output_optional_names = tuple(
            (
                _optional_flag_name(value.name)
                if is_optional_value_type(value.type)
                else None
            )
            for value in graph.outputs
        )
        output_types = tuple(value.type for value in graph.outputs)
        output_shapes = tuple(
            tensor_type(value.name, value.type).shape for value in graph.outputs
        )
        output_dtypes = tuple(
            tensor_type(value.name, value.type).dtype for value in graph.outputs
        )
        return (
            input_names,
            input_optional_names,
            input_shapes,
            input_dtypes,
            input_types,
            output_names,
            output_optional_names,
            output_shapes,
            output_dtypes,
            output_types,
        )

    def _lower_nodes(self, ctx: GraphContext) -> tuple[list[OpBase], list[NodeInfo]]:
        ops: list[OpBase] = []
        node_infos: list[NodeInfo] = []
        registry = get_lowering_registry()
        for node in ctx.nodes:
            lowering = registry.get(node.op_type)
            if lowering is None:
                raise UnsupportedOpError(f"Unsupported op {node.op_type}")
            ops.append(lowering(ctx, node))
            node_infos.append(
                NodeInfo(
                    op_type=node.op_type,
                    name=node.name,
                    inputs=tuple(node.inputs),
                    outputs=tuple(node.outputs),
                    attrs=dict(node.attrs),
                )
            )
        return ops, node_infos

    def _build_header(self, model: onnx.ModelProto, graph: Graph) -> ModelHeader:
        metadata_props = tuple((prop.key, prop.value) for prop in model.metadata_props)
        opset_imports = tuple(
            (opset.domain, opset.version) for opset in model.opset_import
        )
        checksum = self._options.model_checksum
        if checksum is None:
            checksum = hashlib.sha256(model.SerializeToString()).hexdigest()
        codegen_settings = [
            ("emit_testbench", str(self._options.emit_testbench)),
            ("restrict_arrays", str(self._options.restrict_arrays)),
            (
                "fp32_accumulation_strategy",
                self._options.fp32_accumulation_strategy,
            ),
            (
                "fp16_accumulation_strategy",
                self._options.fp16_accumulation_strategy,
            ),
        ]
        if self._options.truncate_weights_after is not None:
            codegen_settings.append(
                ("truncate_weights_after", str(self._options.truncate_weights_after))
            )
        codegen_settings.extend(
            [
                (
                    "large_temp_threshold",
                    str(self._options.large_temp_threshold_bytes),
                ),
                (
                    "large_weight_threshold",
                    str(self._options.large_weight_threshold),
                ),
            ]
        )
        return ModelHeader(
            generator="Generated by emmtrix ONNX-to-C Code Generator (emx-onnx-cgen)",
            model_checksum=checksum,
            model_name=self._options.model_name,
            graph_name=model.graph.name or None,
            description=model.doc_string or None,
            graph_description=model.graph.doc_string or None,
            producer_name=model.producer_name or None,
            producer_version=model.producer_version or None,
            domain=model.domain or None,
            model_version=model.model_version or None,
            ir_version=model.ir_version or None,
            opset_imports=opset_imports,
            metadata_props=metadata_props,
            input_count=len(graph.inputs),
            output_count=len(graph.outputs),
            node_count=len(graph.nodes),
            initializer_count=len(graph.initializers),
            codegen_settings=tuple(codegen_settings),
        )


def _lowered_constants(graph: Graph | GraphContext) -> tuple[ConstTensor, ...]:
    used_initializers = {value.name for value in graph.outputs}
    for node in graph.nodes:
        used_initializers.update(node.inputs)
    constants: list[ConstTensor] = []
    for initializer in graph.initializers:
        if initializer.name not in used_initializers:
            continue
        dtype = ensure_supported_dtype(initializer.type.dtype)
        data_array = initializer.data.astype(dtype.np_dtype, copy=False)
        data_tuple = tuple(data_array.ravel().tolist())
        constants.append(
            ConstTensor(
                name=initializer.name,
                shape=initializer.type.shape,
                data=data_tuple,
                dtype=dtype,
            )
        )
    return tuple(constants)
