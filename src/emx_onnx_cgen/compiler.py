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
from .dtypes import dtype_info
from .errors import CodegenError, UnsupportedOpError
from .ir.context import GraphContext
from .ir.model import Graph, TensorType, Value, ValueType
from .ir.op_base import OpBase
from .ir.op_context import OpContext
from .lowering import load_lowering_registry
from .lowering.common import ensure_supported_dtype, shape_product, value_dtype
from .lowering.registry import get_lowering_registry
from .onnx_import import import_onnx


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
    testbench_inputs: Mapping[str, np.ndarray] | None = None
    testbench_optional_inputs: Mapping[str, bool] | None = None
    truncate_weights_after: int | None = None
    large_temp_threshold_bytes: int = 1024
    large_weight_threshold: int = 100 * 1024
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
        )
        load_lowering_registry()

    def _time_step(self, label: str, func: Callable[[], _T]) -> _T:
        timings = self._options.timings
        if timings is None:
            return func()
        started = time.perf_counter()
        result = func()
        timings[label] = time.perf_counter() - started
        return result

    def compile(self, model: onnx.ModelProto) -> str:
        graph = self._time_step("import_onnx", lambda: import_onnx(model))
        graph = self._time_step(
            "concretize_shapes",
            lambda: self._concretize_graph_shapes(model, graph),
        )
        testbench_inputs = self._time_step(
            "resolve_testbench_inputs", lambda: self._resolve_testbench_inputs(graph)
        )
        variable_dim_inputs, variable_dim_outputs = self._time_step(
            "collect_variable_dims", lambda: self._collect_variable_dims(graph)
        )
        lowered = self._time_step(
            "lower_model", lambda: self._lower_model(model, graph)
        )
        return self._time_step(
            "emit_model",
            lambda: self._emitter.emit_model(
                lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_inputs=testbench_inputs,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                variable_dim_inputs=variable_dim_inputs,
                variable_dim_outputs=variable_dim_outputs,
            ),
        )

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        graph = self._time_step("import_onnx", lambda: import_onnx(model))
        graph = self._time_step(
            "concretize_shapes",
            lambda: self._concretize_graph_shapes(model, graph),
        )
        testbench_inputs = self._time_step(
            "resolve_testbench_inputs", lambda: self._resolve_testbench_inputs(graph)
        )
        variable_dim_inputs, variable_dim_outputs = self._time_step(
            "collect_variable_dims", lambda: self._collect_variable_dims(graph)
        )
        lowered = self._time_step(
            "lower_model", lambda: self._lower_model(model, graph)
        )
        return self._time_step(
            "emit_model_with_data_file",
            lambda: self._emitter.emit_model_with_data_file(
                lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_inputs=testbench_inputs,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                variable_dim_inputs=variable_dim_inputs,
                variable_dim_outputs=variable_dim_outputs,
            ),
        )

    def compile_with_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, bytes | None]:
        graph = self._time_step("import_onnx", lambda: import_onnx(model))
        graph = self._time_step(
            "concretize_shapes",
            lambda: self._concretize_graph_shapes(model, graph),
        )
        testbench_inputs = self._time_step(
            "resolve_testbench_inputs", lambda: self._resolve_testbench_inputs(graph)
        )
        variable_dim_inputs, variable_dim_outputs = self._time_step(
            "collect_variable_dims", lambda: self._collect_variable_dims(graph)
        )
        lowered = self._time_step(
            "lower_model", lambda: self._lower_model(model, graph)
        )
        generated = self._time_step(
            "emit_model",
            lambda: self._emitter.emit_model(
                lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_inputs=testbench_inputs,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                variable_dim_inputs=variable_dim_inputs,
                variable_dim_outputs=variable_dim_outputs,
            ),
        )
        weight_data = self._time_step(
            "collect_weight_data",
            lambda: self._emitter.collect_weight_data(lowered.constants),
        )
        return generated, weight_data

    def compile_with_data_file_and_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, str, bytes | None]:
        graph = self._time_step("import_onnx", lambda: import_onnx(model))
        graph = self._time_step(
            "concretize_shapes",
            lambda: self._concretize_graph_shapes(model, graph),
        )
        testbench_inputs = self._time_step(
            "resolve_testbench_inputs", lambda: self._resolve_testbench_inputs(graph)
        )
        variable_dim_inputs, variable_dim_outputs = self._time_step(
            "collect_variable_dims", lambda: self._collect_variable_dims(graph)
        )
        lowered = self._time_step(
            "lower_model", lambda: self._lower_model(model, graph)
        )
        generated, data_source = self._time_step(
            "emit_model_with_data_file",
            lambda: self._emitter.emit_model_with_data_file(
                lowered,
                emit_testbench=self._options.emit_testbench,
                testbench_inputs=testbench_inputs,
                testbench_optional_inputs=self._options.testbench_optional_inputs,
                variable_dim_inputs=variable_dim_inputs,
                variable_dim_outputs=variable_dim_outputs,
            ),
        )
        weight_data = self._time_step(
            "collect_weight_data",
            lambda: self._emitter.collect_weight_data(lowered.constants),
        )
        return generated, data_source, weight_data

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
        op_ctx = OpContext(ctx)
        for op in ops:
            op.validate(op_ctx)
        for op in ops:
            op.infer_types(op_ctx)
        for op in ops:
            op.infer_shapes(op_ctx)
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

    def _resolve_testbench_inputs(
        self, graph: Graph
    ) -> Mapping[str, tuple[float | int | bool, ...]] | None:
        if not self._options.testbench_inputs:
            return None
        input_specs = {value.name: value for value in graph.inputs}
        unknown_inputs = sorted(
            name for name in self._options.testbench_inputs if name not in input_specs
        )
        if unknown_inputs:
            raise CodegenError(
                "Testbench inputs include unknown inputs: " + ", ".join(unknown_inputs)
            )
        for name, values in self._options.testbench_inputs.items():
            if not isinstance(values, np.ndarray):
                raise CodegenError(f"Testbench input {name} must be a numpy array")
            input_value = input_specs[name]
            if not isinstance(input_value.type, TensorType):
                raise CodegenError(f"Testbench input {name} must be a tensor value")
            dtype = value_dtype(graph, name)
            info = dtype_info(dtype)
            expected_shape = input_value.type.shape
            expected_count = shape_product(expected_shape)
            array = values.astype(info.np_dtype, copy=False)
            if array.size != expected_count:
                raise CodegenError(
                    "Testbench input "
                    f"{name} has {array.size} elements, expected {expected_count}"
                )
        return None

    def _concretize_graph_shapes(self, model: onnx.ModelProto, graph: Graph) -> Graph:
        if not self._options.testbench_inputs:
            return graph
        if not any(
            isinstance(value.type, TensorType) and value.type.dim_params
            for value in graph.values + graph.inputs + graph.outputs
        ):
            return graph
        try:
            import onnxruntime as ort
        except Exception:
            return graph
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
            output_arrays = sess.run(None, self._options.testbench_inputs)
        except Exception:
            return graph

        shapes_by_name: dict[str, tuple[int, ...]] = {
            name: tuple(int(dim) for dim in array.shape)
            for name, array in zip(output_names, output_arrays)
        }
        for name, array in self._options.testbench_inputs.items():
            shapes_by_name[name] = tuple(int(dim) for dim in array.shape)

        def concretize_value(value: Value) -> Value:
            shape = shapes_by_name.get(value.name)
            if shape is None:
                return value
            if not isinstance(value.type, TensorType):
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
        if not graph.outputs:
            raise UnsupportedOpError("Graph must have at least one output")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        for value in graph.outputs:
            if isinstance(value.type, TensorType):
                shape_product(value.type.shape)

    def _collect_io_specs(self, graph: Graph) -> tuple[
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
        def tensor_type(value_type: ValueType) -> TensorType:
            if isinstance(value_type, TensorType):
                return value_type
            return value_type.elem

        input_names = tuple(value.name for value in graph.inputs)
        input_optional_names = tuple(
            (
                _optional_flag_name(value.name)
                if isinstance(value.type, TensorType) and value.type.is_optional
                else None
            )
            for value in graph.inputs
        )
        input_types = tuple(value.type for value in graph.inputs)
        input_shapes = tuple(tensor_type(value.type).shape for value in graph.inputs)
        input_dtypes = tuple(tensor_type(value.type).dtype for value in graph.inputs)
        output_names = tuple(value.name for value in graph.outputs)
        output_optional_names = tuple(
            (
                _optional_flag_name(value.name)
                if isinstance(value.type, TensorType) and value.type.is_optional
                else None
            )
            for value in graph.outputs
        )
        output_types = tuple(value.type for value in graph.outputs)
        output_shapes = tuple(tensor_type(value.type).shape for value in graph.outputs)
        output_dtypes = tuple(tensor_type(value.type).dtype for value in graph.outputs)
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
