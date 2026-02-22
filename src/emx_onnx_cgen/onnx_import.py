from __future__ import annotations

from typing import Iterable, Mapping

import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference

from shared.scalar_types import ScalarType

from .dtypes import scalar_type_from_onnx
from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import (
    Graph,
    Initializer,
    Node,
    SequenceType,
    TensorType,
    Value,
    ValueType,
)


def _normalize_initializer_data(dtype: ScalarType, data: object) -> np.ndarray:
    if isinstance(data, (onnx.TensorProto, onnx.SparseTensorProto)):
        array = numpy_helper.to_array(data)
    elif isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data)
    return array.astype(dtype.np_dtype, copy=False)


def _format_elem_type(elem_type: int) -> str:
    try:
        name = onnx.TensorProto.DataType.Name(elem_type)
    except ValueError:
        name = "UNKNOWN"
    return f"{elem_type} ({name})"


def _unsupported_value_type(value_info: onnx.ValueInfoProto) -> UnsupportedOpError:
    value_kind = value_info.type.WhichOneof("value")
    if value_kind is None:
        value_kind = "unknown"
    return UnsupportedOpError(
        f"Unsupported value type '{value_kind}' for '{value_info.name}'. "
        "Hint: export the model with tensor inputs/outputs."
    )


def _tensor_type_from_proto(
    tensor_type: onnx.TypeProto.Tensor,
    name: str,
    *,
    dim_param_override: tuple[str | None, ...] | None = None,
) -> TensorType:
    if not tensor_type.HasField("elem_type"):
        raise ShapeInferenceError(f"Missing elem_type for tensor '{name}'")
    dtype = scalar_type_from_onnx(tensor_type.elem_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(tensor_type.elem_type)} for tensor '{name}'."
        )
    shape = []
    dim_params = []
    for dim_index, dim in enumerate(tensor_type.shape.dim):
        dim_param = dim.dim_param if dim.HasField("dim_param") else ""
        if (
            dim_param_override is not None
            and dim_index < len(dim_param_override)
            and dim_param_override[dim_index]
        ):
            dim_param = dim_param_override[dim_index] or ""
        dim_params.append(dim_param or None)
        if not dim.HasField("dim_value"):
            if dim_param:
                shape.append(1)
                continue
            synthetic_dim_param = f"{name}_dim_{dim_index}"
            dim_params[-1] = synthetic_dim_param
            shape.append(1)
            continue
        shape.append(dim.dim_value)
    return TensorType(
        dtype=dtype,
        shape=tuple(shape),
        dim_params=tuple(dim_params),
    )


def _value_type(
    value_info: onnx.ValueInfoProto,
    *,
    dim_param_override: tuple[str | None, ...] | None = None,
) -> ValueType:
    value_kind = value_info.type.WhichOneof("value")
    if value_kind == "tensor_type":
        return _tensor_type_from_proto(
            value_info.type.tensor_type,
            value_info.name,
            dim_param_override=dim_param_override,
        )
    if value_kind == "optional_type":
        elem_type = value_info.type.optional_type.elem_type
        elem_kind = elem_type.WhichOneof("value")
        if elem_kind != "tensor_type":
            raise UnsupportedOpError(
                f"Unsupported optional element type '{elem_kind}' for '{value_info.name}'. "
                "Hint: export the model with optional tensor inputs/outputs."
            )
        tensor_type = _tensor_type_from_proto(
            elem_type.tensor_type,
            value_info.name,
            dim_param_override=dim_param_override,
        )
        return TensorType(
            dtype=tensor_type.dtype,
            shape=tensor_type.shape,
            dim_params=tensor_type.dim_params,
            is_optional=True,
        )
    if value_kind == "sequence_type":
        elem_type = value_info.type.sequence_type.elem_type
        if elem_type.WhichOneof("value") != "tensor_type":
            raise UnsupportedOpError(
                f"Unsupported sequence element type '{elem_type.WhichOneof('value')}' for '{value_info.name}'. "
                "Hint: export the model with sequence<tensor<...>> inputs/outputs."
            )
        return SequenceType(
            elem=_tensor_type_from_proto(
                elem_type.tensor_type,
                value_info.name,
                dim_param_override=dim_param_override,
            )
        )
    raise _unsupported_value_type(value_info)


def _values(
    value_infos: Iterable[onnx.ValueInfoProto],
    *,
    dim_param_by_name: Mapping[str, tuple[str | None, ...]] | None = None,
) -> tuple[Value, ...]:
    dim_param_by_name = dim_param_by_name or {}
    return tuple(
        Value(
            name=vi.name,
            type=_value_type(vi, dim_param_override=dim_param_by_name.get(vi.name)),
        )
        for vi in value_infos
    )


def _collect_dim_params(
    value_infos: Iterable[onnx.ValueInfoProto],
) -> dict[str, tuple[str | None, ...]]:
    dim_params: dict[str, tuple[str | None, ...]] = {}
    for value_info in value_infos:
        value_kind = value_info.type.WhichOneof("value")
        if value_kind == "tensor_type":
            tensor_type = value_info.type.tensor_type
        elif value_kind == "optional_type":
            elem_type = value_info.type.optional_type.elem_type
            if elem_type.WhichOneof("value") != "tensor_type":
                continue
            tensor_type = elem_type.tensor_type
        else:
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dim_param = dim.dim_param if dim.HasField("dim_param") else ""
            dims.append(dim_param or None)
        if any(dims):
            dim_params[value_info.name] = tuple(dims)
    return dim_params


def _value_info_complete(value_info: onnx.ValueInfoProto) -> bool:
    value_kind = value_info.type.WhichOneof("value")
    if value_kind == "tensor_type":
        tensor_type = value_info.type.tensor_type
    elif value_kind == "optional_type":
        elem_type = value_info.type.optional_type.elem_type
        if elem_type.WhichOneof("value") != "tensor_type":
            return False
        tensor_type = elem_type.tensor_type
    else:
        return False
    if not tensor_type.HasField("elem_type"):
        return False
    if not tensor_type.HasField("shape"):
        return False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            continue
        if dim.HasField("dim_param"):
            continue
        return False
    return True


def _needs_shape_inference(model: onnx.ModelProto) -> bool:
    graph = model.graph
    value_info_by_name = {
        value_info.name: value_info for value_info in graph.value_info
    }
    output_names = {value_info.name for value_info in graph.output}
    initializer_names = {initializer.name for initializer in graph.initializer}
    initializer_names.update(
        sparse_init.name for sparse_init in graph.sparse_initializer
    )
    for node in graph.node:
        for output in node.output:
            if not output:
                continue
            if output in output_names or output in value_info_by_name:
                continue
            return True
    for value_info in graph.value_info:
        if not _value_info_complete(value_info):
            return True
    for value_info in graph.output:
        if not _value_info_complete(value_info):
            return True
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        if not _value_info_complete(value_info):
            return True
    return False


def _initializer(value: onnx.TensorProto) -> Initializer:
    dtype = scalar_type_from_onnx(value.data_type)
    if dtype is None:
        raise UnsupportedOpError(
            "Unsupported elem_type "
            f"{_format_elem_type(value.data_type)} for initializer '{value.name}'. "
            "Hint: export the model with float32 initializers."
        )
    data = _normalize_initializer_data(dtype, value)
    return Initializer(
        name=value.name,
        type=TensorType(
            dtype=dtype,
            shape=tuple(data.shape),
            dim_params=(None,) * len(data.shape),
        ),
        data=data,
    )


def _node_attrs(node: onnx.NodeProto) -> dict[str, object]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _find_value_info(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto | None:
    for value_info in graph.input:
        if value_info.name == name:
            return value_info
    for value_info in graph.value_info:
        if value_info.name == name:
            return value_info
    for value_info in graph.output:
        if value_info.name == name:
            return value_info
    return None


def _tensor_shape_from_value_info(graph: onnx.GraphProto, name: str) -> tuple[int, ...]:
    value_info = _find_value_info(graph, name)
    if value_info is None:
        for initializer in graph.initializer:
            if initializer.name == name:
                return tuple(int(dim) for dim in initializer.dims)
        raise ShapeInferenceError(
            f"Missing shape for '{name}' in Scan expansion. "
            "Hint: run ONNX shape inference or export with static shapes."
        )
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ShapeInferenceError(
            f"Missing shape for '{name}' in Scan expansion. "
            "Hint: run ONNX shape inference or export with static shapes."
        )
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ShapeInferenceError(
                f"Dynamic dim for '{name}' in Scan expansion. "
                "Hint: export with static shapes."
            )
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _tensor_elem_type_from_value_info(graph: onnx.GraphProto, name: str) -> int:
    value_info = _find_value_info(graph, name)
    if value_info is not None:
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField("elem_type"):
            return int(tensor_type.elem_type)
    for initializer in graph.initializer:
        if initializer.name == name:
            return int(initializer.data_type)
    raise ShapeInferenceError(
        f"Missing elem_type for '{name}' in Gradient expansion. "
        "Hint: run ONNX shape inference or export with static shapes."
    )


def _scan_attr_ints(
    attrs: dict[str, object],
    key: str,
    *,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    value = attrs.get(key)
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _onnx_opset_version(model: onnx.ModelProto) -> int | None:
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            return int(opset.version)
    return None


def _scan_expected_axis(is_opset8: bool) -> int:
    return 1 if is_opset8 else 0


def _scan_axes_and_directions(
    attrs: dict[str, object],
    *,
    num_scan_inputs: int,
    scan_output_count: int,
    is_opset8: bool,
) -> None:
    default_axis = _scan_expected_axis(is_opset8)
    scan_input_axes = _scan_attr_ints(
        attrs,
        "scan_input_axes",
        default=(default_axis,) * num_scan_inputs,
    )
    scan_output_axes = _scan_attr_ints(
        attrs,
        "scan_output_axes",
        default=(default_axis,) * scan_output_count,
    )
    scan_input_directions = _scan_attr_ints(
        attrs,
        "scan_input_directions",
        default=(0,) * num_scan_inputs,
    )
    scan_output_directions = _scan_attr_ints(
        attrs,
        "scan_output_directions",
        default=(0,) * scan_output_count,
    )
    if any(axis != default_axis for axis in scan_input_axes):
        raise UnsupportedOpError(f"Scan only supports scan_input_axes={default_axis}")
    if any(axis != default_axis for axis in scan_output_axes):
        raise UnsupportedOpError(f"Scan only supports scan_output_axes={default_axis}")
    if any(direction != 0 for direction in scan_input_directions):
        raise UnsupportedOpError("Scan only supports scan_input_directions=0")
    if any(direction != 0 for direction in scan_output_directions):
        raise UnsupportedOpError("Scan only supports scan_output_directions=0")


def _scan_sequence_length(
    graph: onnx.GraphProto,
    scan_input_names: list[str],
    *,
    is_opset8: bool,
) -> tuple[int, int | None]:
    scan_input_shapes = [
        _tensor_shape_from_value_info(graph, name) for name in scan_input_names
    ]
    if not scan_input_shapes:
        raise UnsupportedOpError("Scan requires scan inputs")
    if is_opset8:
        if any(len(shape) < 2 for shape in scan_input_shapes):
            raise UnsupportedOpError(
                "Scan opset 8 inputs must include batch and sequence dims"
            )
        batch_size = scan_input_shapes[0][0]
        sequence_len = scan_input_shapes[0][1]
        if batch_size != 1:
            raise UnsupportedOpError(
                "Scan opset 8 currently supports batch size 1 only"
            )
        if sequence_len <= 0:
            raise UnsupportedOpError("Scan requires positive sequence length")
        if any(
            shape[0] != batch_size or shape[1] != sequence_len
            for shape in scan_input_shapes
        ):
            raise UnsupportedOpError(
                "Scan inputs must share the same batch and sequence length"
            )
        return sequence_len, batch_size
    sequence_len = scan_input_shapes[0][0]
    if sequence_len <= 0:
        raise UnsupportedOpError("Scan requires positive sequence length")
    if any(shape[0] != sequence_len for shape in scan_input_shapes):
        raise UnsupportedOpError("Scan inputs must share the same sequence length")
    return sequence_len, None


def _scan_body_initializers(
    body: onnx.GraphProto,
    *,
    prefix: str,
    new_initializers: list[onnx.TensorProto],
) -> dict[str, str]:
    initializer_map: dict[str, str] = {}
    for initializer in body.initializer:
        new_name = f"{prefix}_init_{initializer.name}"
        initializer_map[initializer.name] = new_name
        array = numpy_helper.to_array(initializer)
        new_initializers.append(numpy_helper.from_array(array, name=new_name))
    return initializer_map


def _scan_state_inputs(
    graph: onnx.GraphProto,
    *,
    prefix: str,
    state_input_names: list[str],
    new_nodes: list[onnx.NodeProto],
    is_opset8: bool,
    batch_size: int | None,
) -> list[str]:
    state_names = list(state_input_names)
    if is_opset8 and state_input_names:
        for state_index, state_name in enumerate(state_input_names):
            state_shape = _tensor_shape_from_value_info(graph, state_name)
            if not state_shape:
                raise UnsupportedOpError("Scan opset 8 state inputs must be tensors")
            if batch_size is not None and state_shape[0] != batch_size:
                raise UnsupportedOpError(
                    "Scan opset 8 state inputs must match batch size"
                )
            squeezed_name = f"{prefix}_state{state_index}_squeezed"
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[state_name],
                    outputs=[squeezed_name],
                    name=f"{squeezed_name}_node",
                    axes=[0],
                )
            )
            state_names[state_index] = squeezed_name
    return state_names


def _scan_iteration_inputs(
    *,
    prefix: str,
    iter_index: int,
    scan_input_names: list[str],
    new_nodes: list[onnx.NodeProto],
    is_opset8: bool,
) -> list[str]:
    scan_iter_inputs: list[str] = []
    slice_axis = _scan_expected_axis(is_opset8)
    squeeze_axes = [0, 1] if is_opset8 else [0]
    for scan_index, scan_name in enumerate(scan_input_names):
        slice_out = f"{prefix}_iter{iter_index}_scan{scan_index}_slice"
        squeeze_out = f"{prefix}_iter{iter_index}_scan{scan_index}_value"
        new_nodes.append(
            helper.make_node(
                "Slice",
                inputs=[scan_name],
                outputs=[slice_out],
                name=f"{slice_out}_node",
                starts=[iter_index],
                ends=[iter_index + 1],
                axes=[slice_axis],
            )
        )
        new_nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=[slice_out],
                outputs=[squeeze_out],
                name=f"{squeeze_out}_node",
                axes=squeeze_axes,
            )
        )
        scan_iter_inputs.append(squeeze_out)
    return scan_iter_inputs


def _expand_scan_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    graph = model.graph
    opset_version = _onnx_opset_version(model)
    if opset_version is None:
        return model, False

    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    scan_index = 0
    expanded = False
    is_opset8 = opset_version <= 8

    for node in graph.node:
        if node.op_type != "Scan":
            new_nodes.append(node)
            continue

        expanded = True
        scan_index += 1
        attrs = _node_attrs(node)
        body = attrs.get("body")
        if not isinstance(body, onnx.GraphProto):
            raise UnsupportedOpError("Scan requires a body graph")
        num_scan_inputs = int(attrs.get("num_scan_inputs", 0))
        if num_scan_inputs <= 0:
            raise UnsupportedOpError("Scan requires num_scan_inputs")
        input_names = list(node.input)
        if is_opset8:
            if not input_names:
                raise UnsupportedOpError("Scan in opset 8 requires inputs")
            sequence_lens = input_names.pop(0)
            if sequence_lens:
                raise UnsupportedOpError("Scan sequence_lens input is not supported")
        num_state_inputs = len(input_names) - num_scan_inputs
        if num_state_inputs < 0:
            raise UnsupportedOpError("Scan input count is invalid")
        if len(body.input) != num_state_inputs + num_scan_inputs:
            raise UnsupportedOpError(
                "Scan body input count must match state and scan inputs"
            )
        if len(body.output) != len(node.output):
            raise UnsupportedOpError("Scan body output count must match Scan outputs")
        scan_output_count = len(node.output) - num_state_inputs
        _scan_axes_and_directions(
            attrs,
            num_scan_inputs=num_scan_inputs,
            scan_output_count=scan_output_count,
            is_opset8=is_opset8,
        )

        state_input_names = input_names[:num_state_inputs]
        scan_input_names = input_names[num_state_inputs:]
        sequence_len, batch_size = _scan_sequence_length(
            graph,
            scan_input_names,
            is_opset8=is_opset8,
        )

        prefix = node.name or f"scan_{scan_index}"
        initializer_map = _scan_body_initializers(
            body,
            prefix=prefix,
            new_initializers=new_initializers,
        )

        state_names = _scan_state_inputs(
            graph,
            prefix=prefix,
            state_input_names=state_input_names,
            new_nodes=new_nodes,
            is_opset8=is_opset8,
            batch_size=batch_size,
        )
        scan_output_buffers: list[list[str]] = [[] for _ in range(scan_output_count)]

        for iter_index in range(sequence_len):
            scan_iter_inputs = _scan_iteration_inputs(
                prefix=prefix,
                iter_index=iter_index,
                scan_input_names=scan_input_names,
                new_nodes=new_nodes,
                is_opset8=is_opset8,
            )
            name_map: dict[str, str] = {}
            for index, value in enumerate(body.input[:num_state_inputs]):
                name_map[value.name] = state_names[index]
            for index, value in enumerate(
                body.input[num_state_inputs : num_state_inputs + num_scan_inputs]
            ):
                name_map[value.name] = scan_iter_inputs[index]
            for original, mapped in initializer_map.items():
                name_map[original] = mapped

            for body_node in body.node:
                body_attrs = _node_attrs(body_node)
                mapped_inputs = [
                    name_map.get(input_name, input_name)
                    for input_name in body_node.input
                ]
                mapped_outputs: list[str] = []
                for output_name in body_node.output:
                    if not output_name:
                        mapped_outputs.append("")
                        continue
                    mapped_name = f"{prefix}_iter{iter_index}_{output_name}"
                    name_map[output_name] = mapped_name
                    mapped_outputs.append(mapped_name)
                new_nodes.append(
                    helper.make_node(
                        body_node.op_type,
                        inputs=mapped_inputs,
                        outputs=mapped_outputs,
                        name=(
                            f"{prefix}_iter{iter_index}_{body_node.name}"
                            if body_node.name
                            else ""
                        ),
                        domain=body_node.domain,
                        **body_attrs,
                    )
                )

            for index, output in enumerate(body.output[:num_state_inputs]):
                mapped_output = name_map.get(output.name)
                if mapped_output is None:
                    raise UnsupportedOpError(
                        "Scan body did not produce a required state output"
                    )
                state_names[index] = mapped_output

            for output_index, output in enumerate(
                body.output[num_state_inputs : num_state_inputs + scan_output_count]
            ):
                mapped_output = name_map.get(output.name)
                if mapped_output is None:
                    raise UnsupportedOpError(
                        "Scan body did not produce a required scan output"
                    )
                unsqueeze_out = f"{prefix}_iter{iter_index}_scanout{output_index}"
                unsqueeze_axes = [0, 1] if is_opset8 else [0]
                new_nodes.append(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[mapped_output],
                        outputs=[unsqueeze_out],
                        name=f"{unsqueeze_out}_node",
                        axes=unsqueeze_axes,
                    )
                )
                scan_output_buffers[output_index].append(unsqueeze_out)

        for index, output_name in enumerate(node.output[:num_state_inputs]):
            state_value = state_names[index]
            if is_opset8:
                expanded_state = f"{prefix}_state_output_{index}_expanded"
                new_nodes.append(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[state_value],
                        outputs=[expanded_state],
                        name=f"{expanded_state}_node",
                        axes=[0],
                    )
                )
                state_value = expanded_state
            if state_value == output_name:
                continue
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[state_value],
                    outputs=[output_name],
                    name=f"{prefix}_state_output_{index}",
                )
            )

        for output_index, output_name in enumerate(
            node.output[num_state_inputs : num_state_inputs + scan_output_count]
        ):
            buffer = scan_output_buffers[output_index]
            concat_axis = _scan_expected_axis(is_opset8)
            if len(buffer) == 1:
                new_nodes.append(
                    helper.make_node(
                        "Identity",
                        inputs=buffer,
                        outputs=[output_name],
                        name=f"{prefix}_scan_output_{output_index}",
                    )
                )
            else:
                new_nodes.append(
                    helper.make_node(
                        "Concat",
                        inputs=buffer,
                        outputs=[output_name],
                        name=f"{prefix}_scan_output_{output_index}",
                        axis=concat_axis,
                    )
                )

    if expanded:
        del graph.node[:]
        graph.node.extend(new_nodes)
        if new_initializers:
            graph.initializer.extend(new_initializers)
    return model, expanded


def _if_graph_attrs(
    node: onnx.NodeProto,
) -> tuple[onnx.GraphProto, onnx.GraphProto] | None:
    then_branch: onnx.GraphProto | None = None
    else_branch: onnx.GraphProto | None = None
    for attr in node.attribute:
        if attr.name == "then_branch" and attr.HasField("g"):
            then_branch = attr.g
        if attr.name == "else_branch" and attr.HasField("g"):
            else_branch = attr.g
    if then_branch is None or else_branch is None:
        return None
    return then_branch, else_branch


def _is_tensor_output(graph: onnx.GraphProto, name: str) -> bool:
    value_info = _find_value_info(graph, name)
    return (
        value_info is not None and value_info.type.WhichOneof("value") == "tensor_type"
    )


def _is_sequence_output(graph: onnx.GraphProto, name: str) -> bool:
    value_info = _find_value_info(graph, name)
    return (
        value_info is not None
        and value_info.type.WhichOneof("value") == "sequence_type"
    )


def _sequence_construct_inputs(
    branch: onnx.GraphProto,
    output_name: str,
) -> tuple[str, ...] | None:
    for branch_node in branch.node:
        if output_name not in branch_node.output:
            continue
        if branch_node.op_type != "SequenceConstruct":
            return None
        return tuple(branch_node.input)
    return None


def _inline_if_branch(
    branch: onnx.GraphProto,
    *,
    prefix: str,
    new_initializers: list[onnx.TensorProto],
) -> tuple[list[onnx.NodeProto], dict[str, str]]:
    name_map: dict[str, str] = {}
    for initializer in branch.initializer:
        new_name = f"{prefix}_init_{initializer.name}"
        name_map[initializer.name] = new_name
        array = numpy_helper.to_array(initializer)
        new_initializers.append(numpy_helper.from_array(array, name=new_name))

    inlined_nodes: list[onnx.NodeProto] = []
    for node_index, branch_node in enumerate(branch.node):
        inlined = onnx.NodeProto()
        inlined.CopyFrom(branch_node)
        if inlined.name:
            inlined.name = f"{prefix}_{inlined.name}"
        else:
            inlined.name = f"{prefix}_node_{node_index}"

        del inlined.input[:]
        inlined.input.extend(name_map.get(name, name) for name in branch_node.input)

        mapped_outputs: list[str] = []
        for output_index, output_name in enumerate(branch_node.output):
            if not output_name:
                mapped_outputs.append(output_name)
                continue
            mapped = f"{prefix}_value_{node_index}_{output_index}"
            name_map[output_name] = mapped
            mapped_outputs.append(mapped)
        del inlined.output[:]
        inlined.output.extend(mapped_outputs)
        inlined_nodes.append(inlined)
    return inlined_nodes, name_map


def _expand_if_node(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_index: int,
    *,
    new_initializers: list[onnx.TensorProto],
) -> list[onnx.NodeProto] | None:
    if len(node.input) != 1:
        return None
    branches = _if_graph_attrs(node)
    if branches is None:
        return None
    then_branch, else_branch = branches
    if then_branch.input or else_branch.input:
        return None
    if len(node.output) != len(then_branch.output) or len(node.output) != len(
        else_branch.output
    ):
        return None
    output_is_tensor = tuple(
        _is_tensor_output(graph, output_name) for output_name in node.output
    )
    output_is_sequence = tuple(
        _is_sequence_output(graph, output_name) for output_name in node.output
    )
    if any(
        not (is_tensor or is_sequence)
        for is_tensor, is_sequence in zip(output_is_tensor, output_is_sequence)
    ):
        return None
    if any(is_tensor != output_is_tensor[0] for is_tensor in output_is_tensor):
        return None

    prefix_base = node.name or f"if_{node_index}"
    then_nodes, then_map = _inline_if_branch(
        then_branch,
        prefix=f"{prefix_base}_then",
        new_initializers=new_initializers,
    )
    else_nodes, else_map = _inline_if_branch(
        else_branch,
        prefix=f"{prefix_base}_else",
        new_initializers=new_initializers,
    )

    select_nodes: list[onnx.NodeProto] = []
    cond_name = node.input[0]
    if output_is_tensor[0]:
        for output_index, output_name in enumerate(node.output):
            then_name = then_map.get(then_branch.output[output_index].name)
            else_name = else_map.get(else_branch.output[output_index].name)
            if then_name is None or else_name is None:
                return None
            select_nodes.append(
                helper.make_node(
                    "Where",
                    inputs=[cond_name, then_name, else_name],
                    outputs=[output_name],
                    name=f"{prefix_base}_select_{output_index}",
                )
            )
        return [*then_nodes, *else_nodes, *select_nodes]

    for output_index, output_name in enumerate(node.output):
        then_output_name = then_branch.output[output_index].name
        else_output_name = else_branch.output[output_index].name
        then_inputs = _sequence_construct_inputs(then_branch, then_output_name)
        else_inputs = _sequence_construct_inputs(else_branch, else_output_name)
        if then_inputs is None or else_inputs is None:
            return None
        if len(then_inputs) != len(else_inputs):
            return None

        selected_elements: list[str] = []
        for elem_index, (then_elem, else_elem) in enumerate(
            zip(then_inputs, else_inputs)
        ):
            then_name = then_map.get(then_elem)
            else_name = else_map.get(else_elem)
            if then_name is None or else_name is None:
                return None
            selected_name = f"{prefix_base}_sequence_{output_index}_{elem_index}"
            select_nodes.append(
                helper.make_node(
                    "Where",
                    inputs=[cond_name, then_name, else_name],
                    outputs=[selected_name],
                    name=f"{prefix_base}_select_{output_index}_{elem_index}",
                )
            )
            selected_elements.append(selected_name)

        select_nodes.append(
            helper.make_node(
                "SequenceConstruct",
                inputs=selected_elements,
                outputs=[output_name],
                name=f"{prefix_base}_sequence_construct_{output_index}",
            )
        )

    return [*then_nodes, *else_nodes, *select_nodes]


def _expand_if_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    graph = model.graph
    expanded = False
    while True:
        changed = False
        new_nodes: list[onnx.NodeProto] = []
        new_initializers: list[onnx.TensorProto] = []
        for node_index, node in enumerate(graph.node):
            if node.op_type != "If":
                new_nodes.append(node)
                continue
            replacement = _expand_if_node(
                graph,
                node,
                node_index,
                new_initializers=new_initializers,
            )
            if replacement is None:
                new_nodes.append(node)
                continue
            new_nodes.extend(replacement)
            changed = True

        if not changed:
            break

        expanded = True
        del graph.node[:]
        graph.node.extend(new_nodes)
        if new_initializers:
            graph.initializer.extend(new_initializers)

    return model, expanded


def _expand_gradient_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    graph = model.graph
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    expanded = False

    for node_index, node in enumerate(graph.node):
        if not (
            node.op_type == "Gradient" and node.domain == "ai.onnx.preview.training"
        ):
            new_nodes.append(node)
            continue

        attrs = _node_attrs(node)
        xs_attr = attrs.get("xs")
        y_attr = attrs.get("y")
        if not isinstance(xs_attr, (list, tuple)):
            raise UnsupportedOpError(
                "Gradient requires 'xs' attribute listing input tensor names"
            )
        if not isinstance(y_attr, (bytes, str)):
            raise UnsupportedOpError("Gradient requires string 'y' attribute")
        xs = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in xs_attr
        )
        y_name = y_attr.decode("utf-8") if isinstance(y_attr, bytes) else str(y_attr)
        if len(node.output) != len(xs):
            raise UnsupportedOpError(
                "Gradient output count must match the number of tensors in 'xs'"
            )

        grad_nodes: list[onnx.NodeProto] = []
        grad_by_value: dict[str, str] = {}

        y_shape = _tensor_shape_from_value_info(graph, y_name)
        y_elem_type = _tensor_elem_type_from_value_info(graph, y_name)
        y_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(y_elem_type))
        seed_name = f"__grad_seed_{node_index}_{y_name}"
        new_initializers.append(
            numpy_helper.from_array(np.ones(y_shape, dtype=y_dtype), name=seed_name)
        )
        grad_by_value[y_name] = seed_name

        forward_nodes = [n for n in new_nodes if any(output for output in n.output)]
        for reverse_index, forward_node in enumerate(reversed(forward_nodes)):
            node_outputs = [output for output in forward_node.output if output]
            if len(node_outputs) != 1:
                if node_outputs:
                    raise UnsupportedOpError(
                        f"Gradient only supports single-output ops, got {forward_node.op_type}"
                    )
                continue
            grad_output_name = grad_by_value.get(node_outputs[0])
            if grad_output_name is None:
                continue

            if forward_node.op_type in {"Identity"}:
                contribution_names = [grad_output_name]
            elif forward_node.op_type in {"Add", "Sub"}:
                if len(forward_node.input) != 2:
                    raise UnsupportedOpError(
                        f"Gradient expects two inputs for {forward_node.op_type}"
                    )
                contribution_names = [grad_output_name, grad_output_name]
                if forward_node.op_type == "Sub":
                    neg_name = f"__grad_neg_{node_index}_{reverse_index}"
                    grad_nodes.append(
                        helper.make_node(
                            "Neg",
                            inputs=[grad_output_name],
                            outputs=[neg_name],
                            name=neg_name,
                        )
                    )
                    contribution_names[1] = neg_name
            elif forward_node.op_type == "Mul":
                if len(forward_node.input) != 2:
                    raise UnsupportedOpError("Gradient expects two inputs for Mul")
                left_name, right_name = forward_node.input
                left_grad = f"__grad_mul_l_{node_index}_{reverse_index}"
                right_grad = f"__grad_mul_r_{node_index}_{reverse_index}"
                grad_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[grad_output_name, right_name],
                        outputs=[left_grad],
                        name=left_grad,
                    )
                )
                grad_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[grad_output_name, left_name],
                        outputs=[right_grad],
                        name=right_grad,
                    )
                )
                contribution_names = [left_grad, right_grad]
            else:
                raise UnsupportedOpError(
                    "Gradient currently supports Add/Sub/Mul/Identity; "
                    f"got {forward_node.op_type}"
                )

            for input_name, contribution_name in zip(
                forward_node.input,
                contribution_names,
                strict=False,
            ):
                existing_name = grad_by_value.get(input_name)
                if existing_name is None:
                    grad_by_value[input_name] = contribution_name
                    continue
                accum_name = f"__grad_acc_{node_index}_{reverse_index}_{input_name}"
                grad_nodes.append(
                    helper.make_node(
                        "Add",
                        inputs=[existing_name, contribution_name],
                        outputs=[accum_name],
                        name=accum_name,
                    )
                )
                grad_by_value[input_name] = accum_name

        for output_name, x_name in zip(node.output, xs, strict=False):
            grad_name = grad_by_value.get(x_name)
            if grad_name is None:
                zero_name = f"__grad_zero_{node_index}_{x_name}"
                grad_nodes.append(
                    helper.make_node(
                        "Sub",
                        inputs=[x_name, x_name],
                        outputs=[zero_name],
                        name=zero_name,
                    )
                )
                grad_name = zero_name
            grad_nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[grad_name],
                    outputs=[output_name],
                    name=f"__grad_out_{node_index}_{output_name}",
                )
            )

        new_nodes.extend(grad_nodes)
        expanded = True

    if not expanded:
        return model, False

    del graph.node[:]
    graph.node.extend(new_nodes)
    if new_initializers:
        graph.initializer.extend(new_initializers)
    return model, True


def _constant_initializer(node: onnx.NodeProto) -> Initializer:
    if len(node.output) != 1:
        raise UnsupportedOpError("Constant must have exactly one output")
    attrs = _node_attrs(node)
    output_name = node.output[0]
    if "value" in attrs:
        tensor = attrs["value"]
        dtype = scalar_type_from_onnx(tensor.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=dtype,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "sparse_value" in attrs:
        tensor = attrs["sparse_value"]
        dtype = scalar_type_from_onnx(tensor.values.data_type)
        if dtype is None:
            raise UnsupportedOpError(
                "Unsupported elem_type "
                f"{_format_elem_type(tensor.values.data_type)} for Constant '{output_name}'."
            )
        data = _normalize_initializer_data(dtype, tensor)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=dtype,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "value_float" in attrs or "value_floats" in attrs:
        values = attrs.get("value_floats", attrs.get("value_float"))
        data = _normalize_initializer_data(ScalarType.F32, values)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=ScalarType.F32,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "value_int" in attrs or "value_ints" in attrs:
        values = attrs.get("value_ints", attrs.get("value_int"))
        data = _normalize_initializer_data(ScalarType.I64, values)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=ScalarType.I64,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    if "value_string" in attrs or "value_strings" in attrs:
        values = attrs.get("value_strings", attrs.get("value_string"))
        if isinstance(values, (bytes, str)):
            data = _normalize_initializer_data(ScalarType.STRING, [values])
        else:
            data = _normalize_initializer_data(ScalarType.STRING, values)
        return Initializer(
            name=output_name,
            type=TensorType(
                dtype=ScalarType.STRING,
                shape=tuple(data.shape),
                dim_params=(None,) * len(data.shape),
            ),
            data=data,
        )
    raise UnsupportedOpError(f"Constant '{output_name}' requires a value attribute")


def import_onnx(model: onnx.ModelProto) -> Graph:
    model, _ = _expand_scan_nodes(model)
    model, _ = _expand_if_nodes(model)
    model, _ = _expand_gradient_nodes(model)
    dim_param_by_name = _collect_dim_params(
        tuple(model.graph.input) + tuple(model.graph.output)
    )
    opset_imports = tuple((opset.domain, opset.version) for opset in model.opset_import)
    if _needs_shape_inference(model):
        try:
            model = shape_inference.infer_shapes(model, data_prop=True)
        except Exception as exc:  # pragma: no cover - onnx inference errors
            raise ShapeInferenceError("ONNX shape inference failed") from exc
    graph = model.graph
    base_initializers = [_initializer(value) for value in graph.initializer]
    constant_initializers: list[Initializer] = []
    input_names = {value_info.name for value_info in graph.input}
    output_names = {value_info.name for value_info in graph.output}
    nodes: list[Node] = []
    for node in graph.node:
        if node.op_type == "Constant":
            constant_initializers.append(_constant_initializer(node))
            continue
        nodes.append(
            Node(
                op_type=node.op_type,
                name=node.name or None,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attrs=_node_attrs(node),
            )
        )
    initializers = tuple(base_initializers + constant_initializers)
    initializer_names = {initializer.name for initializer in initializers}
    inputs = _values(
        (
            value_info
            for value_info in graph.input
            if value_info.name not in initializer_names
        ),
        dim_param_by_name=dim_param_by_name,
    )
    outputs = _values(graph.output, dim_param_by_name=dim_param_by_name)
    values = _values(
        value_info
        for value_info in graph.value_info
        if value_info.name not in initializer_names | input_names | output_names
    )
    return Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers,
        values=values,
        opset_imports=opset_imports,
    )
