"""Explicit padding: moves the implicit spatial padding of Conv operators into a Pad node.

Conv kernels with ``pads`` check in their innermost loop whether a kernel tap falls into the
padding. With an explicit ``Pad`` in front of them, the operator runs with ``pads=0`` on the
padded tensor and its loop nest has no bounds check, which compilers and vectorizers handle
much better. The cost is one additional tensor per padded operator.

Pooling operators keep their padding: their padding value depends on the operator (the lowest
value for MaxPool, none for AveragePool without ``count_include_pad``), ``ceil_mode`` can drop
windows that start in the end padding, and their windows are small, so that removing their
bounds checks did not pay off in measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import onnx
from onnx import helper, numpy_helper

_DEFAULT_DOMAINS = ("", "ai.onnx")
_MICROSOFT_DOMAIN = "com.microsoft"
# Pad takes pads and constant_value as inputs since opset 11; before, they are attributes, and
# Pad-1 names them differently (`paddings`), which the pass does not emit
_PAD_INPUTS_OPSET = 11
_PAD_ATTRIBUTES_OPSET = 2
_FLOAT_TYPES = (
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.BFLOAT16,
)


@dataclass(frozen=True)
class _GraphInfo:
    """Lookups into the (shape-inferred) top-level graph that the rules need."""

    elem_types: dict[str, int]
    shapes: dict[str, tuple[int | None, ...]]
    value_infos: dict[str, onnx.ValueInfoProto]
    initializers: dict[str, onnx.TensorProto]

    @staticmethod
    def of(graph: onnx.GraphProto) -> "_GraphInfo":
        elem_types: dict[str, int] = {}
        shapes: dict[str, tuple[int | None, ...]] = {}
        value_infos: dict[str, onnx.ValueInfoProto] = {}
        for value in (*graph.input, *graph.value_info, *graph.output):
            tensor_type = value.type.tensor_type
            if not tensor_type.elem_type:
                continue
            elem_types[value.name] = tensor_type.elem_type
            if tensor_type.HasField("shape"):
                shapes[value.name] = tuple(
                    dim.dim_value if dim.HasField("dim_value") else None
                    for dim in tensor_type.shape.dim
                )
                value_infos[value.name] = value
        initializers = {init.name: init for init in graph.initializer}
        for init in graph.initializer:
            elem_types.setdefault(init.name, init.data_type)
            shapes.setdefault(init.name, tuple(init.dims))
        return _GraphInfo(elem_types, shapes, value_infos, initializers)

    def scalar_initializer(self, name: str) -> np.ndarray | None:
        init = self.initializers.get(name)
        if init is None:
            return None
        array = numpy_helper.to_array(init)
        return array.reshape(()) if array.size == 1 else None


def _attribute(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def _set_attribute(node: onnx.NodeProto, name: str, value) -> None:
    for index, attr in enumerate(node.attribute):
        if attr.name == name:
            del node.attribute[index]
            break
    node.attribute.append(helper.make_attribute(name, value))


def _zero_pad_value(node: onnx.NodeProto, info: _GraphInfo) -> float | int | None:
    return 0


def _x_zero_point_pad_value(
    node: onnx.NodeProto, info: _GraphInfo
) -> float | int | None:
    """ConvInteger / QLinearConv pad with the zero point of the input: it is the quantized 0."""
    zero_point_index = 2
    if len(node.input) <= zero_point_index or not node.input[zero_point_index]:
        return 0 if node.op_type == "ConvInteger" else None
    zero_point = info.scalar_initializer(node.input[zero_point_index])
    return None if zero_point is None else int(zero_point)


@dataclass(frozen=True)
class _PaddingRule:
    """The value that the padding of an operator stands for, `None` if it cannot be made explicit."""

    domains: tuple[str, ...]
    pad_value: Callable[[onnx.NodeProto, _GraphInfo], float | int | None]


_RULES: dict[str, _PaddingRule] = {
    "Conv": _PaddingRule(_DEFAULT_DOMAINS, _zero_pad_value),
    "FusedConv": _PaddingRule((_MICROSOFT_DOMAIN,), _zero_pad_value),
    "ConvInteger": _PaddingRule(_DEFAULT_DOMAINS, _x_zero_point_pad_value),
    "QLinearConv": _PaddingRule(_DEFAULT_DOMAINS, _x_zero_point_pad_value),
}


def _kernel_shape(node: onnx.NodeProto, info: _GraphInfo) -> list[int]:
    """The `kernel_shape` attribute, or the spatial dimensions of the weights if it is absent."""
    kernel_shape = list(_attribute(node, "kernel_shape", []))
    if kernel_shape:
        return kernel_shape
    weights_index = 3 if node.op_type == "QLinearConv" else 1
    weights_shape = (
        info.shapes.get(node.input[weights_index])
        if len(node.input) > weights_index
        else None
    )
    if weights_shape is None or any(dim is None for dim in weights_shape[2:]):
        return []
    return list(weights_shape[2:])


def _explicit_pads(
    node: onnx.NodeProto,
    input_shape: tuple[int | None, ...],
    kernel_shape: list[int],
) -> list[int] | None:
    """The spatial pads `[begin..., end...]` of `node`, resolving `auto_pad`; `None` if unknown."""
    spatial_rank = len(input_shape) - 2
    auto_pad = _attribute(node, "auto_pad", b"NOTSET")
    auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
    if auto_pad == "NOTSET":
        pads = list(_attribute(node, "pads", [0] * (2 * spatial_rank)))
        return pads if len(pads) == 2 * spatial_rank else None
    if auto_pad == "VALID":
        return [0] * (2 * spatial_rank)
    if auto_pad not in ("SAME_UPPER", "SAME_LOWER"):
        return None
    spatial = input_shape[2:]
    if len(kernel_shape) != spatial_rank or any(dim is None for dim in spatial):
        return None
    strides = list(_attribute(node, "strides", [1] * spatial_rank))
    dilations = list(_attribute(node, "dilations", [1] * spatial_rank))
    begins: list[int] = []
    ends: list[int] = []
    for size, kernel, stride, dilation in zip(
        spatial, kernel_shape, strides, dilations
    ):
        out = -(-size // stride)
        total = max((out - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
        small = total // 2
        begins.append(small if auto_pad == "SAME_UPPER" else total - small)
        ends.append(total - begins[-1])
    return begins + ends


def _graph_names(graph: onnx.GraphProto) -> set[str]:
    """All value and node names of `graph` and of its subgraphs (e.g. Loop bodies), which a new
    top-level name must not collide with."""
    names = {
        value.name
        for value in (
            *graph.input,
            *graph.output,
            *graph.value_info,
            *graph.initializer,
        )
    }
    for node in graph.node:
        names.update((*node.input, *node.output, node.name))
        for attr in node.attribute:
            for subgraph in (*([attr.g] if attr.HasField("g") else []), *attr.graphs):
                names |= _graph_names(subgraph)
    return names


def _padded_value_info(
    value_info: onnx.ValueInfoProto, name: str, pads: list[int]
) -> onnx.ValueInfoProto | None:
    """The value info of the padded tensor, `None` if a padded dimension is not static."""
    spatial_rank = len(pads) // 2
    padded = onnx.ValueInfoProto()
    padded.CopyFrom(value_info)
    padded.name = name
    dims = padded.type.tensor_type.shape.dim
    for axis in range(spatial_rank):
        total = pads[axis] + pads[spatial_rank + axis]
        dim = dims[2 + axis]
        if total == 0:
            continue
        if not dim.HasField("dim_value"):
            return None
        dim.dim_value += total
    return padded


class ExplicitPaddingPass:
    """Inserts a Pad node in front of every padded Conv node of the top-level graph."""

    def __init__(self, model: onnx.ModelProto) -> None:
        self._model = model
        self._info = _GraphInfo.of(model.graph)
        self._opset = next(
            (op.version for op in model.opset_import if op.domain in _DEFAULT_DOMAINS),
            0,
        )
        self._used_names = _graph_names(model.graph)

    def _unique_name(self, base: str) -> str:
        name = base
        counter = 1
        while name in self._used_names:
            counter += 1
            name = f"{base}_{counter}"
        self._used_names.add(name)
        return name

    def _pad_node(
        self,
        node: onnx.NodeProto,
        pads: list[int],
        value: float | int,
        elem_type: int,
        new_initializers: list[onnx.TensorProto],
    ) -> onnx.NodeProto | None:
        spatial_rank = len(pads) // 2
        # Pad pads all axes: none for batch and channels
        all_pads = [0, 0, *pads[:spatial_rank], 0, 0, *pads[spatial_rank:]]
        padded = self._unique_name(f"{node.input[0]}_padded")
        name = self._unique_name(f"{node.name or node.op_type}_explicit_pad")
        if self._opset < _PAD_ATTRIBUTES_OPSET:
            return None
        if self._opset >= _PAD_INPUTS_OPSET:
            pads_name = self._unique_name(f"{name}_pads")
            value_name = self._unique_name(f"{name}_value")
            np_type = helper.tensor_dtype_to_np_dtype(elem_type)
            new_initializers.append(
                numpy_helper.from_array(np.array(all_pads, dtype=np.int64), pads_name)
            )
            new_initializers.append(
                numpy_helper.from_array(np.array(value, dtype=np_type), value_name)
            )
            return helper.make_node(
                "Pad", [node.input[0], pads_name, value_name], [padded], name=name
            )
        if elem_type not in _FLOAT_TYPES:
            return None
        return helper.make_node(
            "Pad",
            [node.input[0]],
            [padded],
            name=name,
            mode="constant",
            pads=all_pads,
            value=float(value),
        )

    def _rewrite(
        self,
        node: onnx.NodeProto,
        new_initializers: list[onnx.TensorProto],
        new_value_infos: list[onnx.ValueInfoProto],
    ) -> onnx.NodeProto | None:
        """Makes the padding of `node` explicit and returns the Pad node, `None` if it keeps it."""
        rule = _RULES.get(node.op_type)
        if rule is None or node.domain not in rule.domains:
            return None
        input_shape = self._info.shapes.get(node.input[0])
        input_info = self._info.value_infos.get(node.input[0])
        elem_type = self._info.elem_types.get(node.input[0])
        if input_shape is None or len(input_shape) < 3 or elem_type is None:
            return None
        if input_info is None:
            return None
        pads = _explicit_pads(node, input_shape, _kernel_shape(node, self._info))
        if pads is None or not any(pads):
            return None
        value = rule.pad_value(node, self._info)
        if value is None:
            return None
        # Probe the shape first, so that a skipped node reserves no names
        if _padded_value_info(input_info, "", pads) is None:
            return None
        pad_node = self._pad_node(node, pads, value, elem_type, new_initializers)
        if pad_node is None:
            return None

        new_value_infos.append(_padded_value_info(input_info, pad_node.output[0], pads))
        node.input[0] = pad_node.output[0]
        _set_attribute(node, "pads", [0] * len(pads))
        _set_attribute(node, "auto_pad", "NOTSET")
        return pad_node

    def run(self) -> int:
        """Rewrites the model in place and returns the number of inserted Pad nodes."""
        graph = self._model.graph
        new_nodes: list[onnx.NodeProto] = []
        new_initializers: list[onnx.TensorProto] = []
        new_value_infos: list[onnx.ValueInfoProto] = []
        for node in graph.node:
            pad_node = self._rewrite(node, new_initializers, new_value_infos)
            if pad_node is not None:
                new_nodes.append(pad_node)
            new_nodes.append(node)

        count = len(new_nodes) - len(graph.node)
        if count:
            del graph.node[:]
            graph.node.extend(new_nodes)
            graph.initializer.extend(new_initializers)
            graph.value_info.extend(new_value_infos)
        return count


def make_padding_explicit(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """A copy of the prepared `model` with explicit Pad nodes (see the module description) and the
    value infos of their outputs, together with their number; `model` itself if there are none.
    """
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    count = ExplicitPaddingPass(rewritten).run()
    return (rewritten, count) if count else (model, 0)
