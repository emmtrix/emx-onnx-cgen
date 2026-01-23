from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
from typing import Mapping

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
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.context import GraphContext
from .ir.model import Graph, TensorType, Value
from .ir.op_base import OpBase
from .ir.op_context import OpContext
from .ir.ops import ExpandOp, GatherOp, MultiInputBinaryOp
from .lowering import load_lowering_registry
from .lowering.common import ensure_supported_dtype, shape_product, value_dtype
from .lowering.registry import get_lowering_registry
from .onnx_import import import_onnx
from .runtime.evaluator import Evaluator


@dataclass(frozen=True)
class CompilerOptions:
    template_dir: Path | None = None
    model_name: str = "model"
    emit_testbench: bool = False
    command_line: str | None = None
    model_checksum: str | None = None
    restrict_arrays: bool = True
    testbench_inputs: Mapping[str, np.ndarray] | None = None
    truncate_weights_after: int | None = None
    large_temp_threshold_bytes: int = 1024
    large_weight_threshold: int = 1024 * 1024


def _onnx_elem_type(dtype: np.dtype) -> int:
    for elem_type, info in onnx._mapping.TENSOR_TYPE_MAP.items():
        if info.np_dtype == dtype:
            return elem_type
    raise UnsupportedOpError(f"Unsupported dtype {dtype} for ONNX output")


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions()
        self._options = options
        self._emitter = CEmitter(
            options.template_dir,
            restrict_arrays=options.restrict_arrays,
            truncate_weights_after=options.truncate_weights_after,
            large_temp_threshold_bytes=options.large_temp_threshold_bytes,
            large_weight_threshold=options.large_weight_threshold,
        )
        load_lowering_registry()

    def compile(self, model: onnx.ModelProto) -> str:
        graph = import_onnx(model)
        graph = self._concretize_graph_shapes(model, graph)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        graph = import_onnx(model)
        graph = self._concretize_graph_shapes(model, graph)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model_with_data_file(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )

    def compile_with_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, bytes | None]:
        graph = import_onnx(model)
        graph = self._concretize_graph_shapes(model, graph)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        generated = self._emitter.emit_model(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )
        weight_data = self._emitter.collect_weight_data(lowered.constants)
        return generated, weight_data

    def compile_with_data_file_and_weight_data(
        self, model: onnx.ModelProto
    ) -> tuple[str, str, bytes | None]:
        graph = import_onnx(model)
        graph = self._concretize_graph_shapes(model, graph)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        generated, data_source = self._emitter.emit_model_with_data_file(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )
        weight_data = self._emitter.collect_weight_data(lowered.constants)
        return generated, data_source, weight_data

    @staticmethod
    def _collect_variable_dims(
        graph: Graph,
    ) -> tuple[dict[int, dict[int, str]], dict[int, dict[int, str]]]:
        def collect(values: tuple[Value, ...]) -> dict[int, dict[int, str]]:
            dim_map: dict[int, dict[int, str]] = {}
            for index, value in enumerate(values):
                dims = {
                    dim_index: dim_param
                    for dim_index, dim_param in enumerate(
                        value.type.dim_params
                    )
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
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        ) = self._collect_io_specs(graph)
        ops, node_infos = self._lower_nodes(ctx)
        op_ctx = OpContext(ctx)
        for op, node_info in zip(ops, node_infos):
            field_names = {field.name for field in fields(op)}
            if "dtype" in field_names:
                dtype = getattr(op, "dtype")
                for field in fields(op):
                    if not field.name.startswith("output"):
                        continue
                    value = getattr(op, field.name)
                    if isinstance(value, str):
                        op_ctx.set_dtype(value, dtype)
                for name in node_info.outputs:
                    op_ctx.set_dtype(name, dtype)
            if "outputs" in field_names:
                dtype = getattr(op, "dtype", None)
                if dtype is not None:
                    for name in getattr(op, "outputs"):
                        op_ctx.set_dtype(name, dtype)
            if "output_dtype" in field_names and "output" in field_names:
                output_name = getattr(op, "output")
                if isinstance(output_name, str):
                    op_ctx.set_dtype(output_name, getattr(op, "output_dtype"))
            if "output_values_dtype" in field_names:
                op_ctx.set_dtype(
                    getattr(op, "output_values"),
                    getattr(op, "output_values_dtype"),
                )
            if "output_indices_dtype" in field_names:
                op_ctx.set_dtype(
                    getattr(op, "output_indices"),
                    getattr(op, "output_indices_dtype"),
                )
            if isinstance(op, MultiInputBinaryOp) and op.inputs:
                op_ctx.set_dtype(op.output, op_ctx.dtype(op.inputs[0]))
            if isinstance(op, GatherOp):
                op_ctx.set_dtype(op.output, op_ctx.dtype(op.data))
            if isinstance(op, ExpandOp):
                op_ctx.set_dtype(op.output, op_ctx.dtype(op.input0))
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
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
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
            name
            for name in self._options.testbench_inputs
            if name not in input_specs
        )
        if unknown_inputs:
            raise CodegenError(
                "Testbench inputs include unknown inputs: "
                + ", ".join(unknown_inputs)
            )
        resolved: dict[str, tuple[float | int | bool, ...]] = {}
        for name, values in self._options.testbench_inputs.items():
            if not isinstance(values, np.ndarray):
                raise CodegenError(
                    f"Testbench input {name} must be a numpy array"
                )
            input_value = input_specs[name]
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
            array = array.reshape(expected_shape)
            resolved[name] = tuple(array.ravel().tolist())
        return resolved

    def _concretize_graph_shapes(
        self, model: onnx.ModelProto, graph: Graph
    ) -> Graph:
        if not self._options.testbench_inputs:
            return graph
        if not any(value.type.dim_params for value in graph.values):
            if not any(value.type.dim_params for value in graph.inputs):
                if not any(value.type.dim_params for value in graph.outputs):
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
            return Value(
                name=value.name,
                type=TensorType(
                    dtype=value.type.dtype,
                    shape=shape,
                    dim_params=(None,) * len(shape),
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
            shape_product(value.type.shape)

    def _collect_io_specs(
        self, graph: Graph
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
    ]:
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.inputs
        )
        output_names = tuple(value.name for value in graph.outputs)
        output_shapes = tuple(value.type.shape for value in graph.outputs)
        output_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.outputs
        )
        return (
            input_names,
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        )

    def _lower_nodes(
        self, ctx: GraphContext
    ) -> tuple[list[OpBase], list[NodeInfo]]:
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
        metadata_props = tuple(
            (prop.key, prop.value) for prop in model.metadata_props
        )
        opset_imports = tuple(
            (opset.domain, opset.version) for opset in model.opset_import
        )
        checksum = self._options.model_checksum
        if checksum is None:
            checksum = hashlib.sha256(model.SerializeToString()).hexdigest()
        return ModelHeader(
            generator="Generated by emmtrix ONNX-to-C Code Generator (emx-onnx-cgen)",
            command_line=self._options.command_line,
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
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        evaluator = Evaluator(graph)
        return evaluator.run(feeds)


def _lowered_constants(graph: Graph | GraphContext) -> tuple[ConstTensor, ...]:
    used_initializers = {value.name for value in graph.outputs}
    for node in graph.nodes:
        used_initializers.update(node.inputs)
    constants: list[ConstTensor] = []
    for initializer in graph.initializers:
        if initializer.name not in used_initializers:
            continue
        dtype = ensure_supported_dtype(initializer.type.dtype)
        constants.append(
            ConstTensor(
                name=initializer.name,
                shape=initializer.type.shape,
                data=tuple(
                    dtype.np_dtype.type(value)
                    for value in initializer.data.ravel()
                ),
                dtype=dtype,
            )
        )
    return tuple(constants)
