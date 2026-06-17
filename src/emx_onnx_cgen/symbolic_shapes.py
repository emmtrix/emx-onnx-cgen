"""Symbolic shape inference with output-constraint solving.

ONNX shape inference (even with data propagation) leaves a tensor's shape
unresolved whenever it is computed from the *values* of a runtime integer input
-- for example the ``size`` input of ``AffineGrid`` or the target ``shape`` input
of ``CenterCropPad``. The decomposed (``*_expanded``) reference functions for
those operators build their internal shapes from such inputs, so the resulting
graph keeps dynamic dimensions that block static C generation.

This module recovers those dimensions the principled way: it evaluates the
shape arithmetic *forward* with the runtime integer inputs treated as symbols,
collects equations by matching each symbolic shape against the statically
declared shape of the corresponding value (graph outputs and any value_info that
already carries concrete dims), solves the system, and writes the resulting
concrete dimensions back into the graph's value_info.

The pass is deliberately conservative:

* Symbols are only created for the elements of 1-D integer *inputs*. Genuinely
  dynamic tensor dimensions (e.g. a dynamic batch axis on a float input) never
  become symbols, so models that are meant to stay dynamic are left untouched.
* It only ever fills in dimensions that are currently dynamic, and never
  overwrites an existing concrete dimension with a different value.
* Any failure to evaluate or solve simply leaves the graph unchanged.
"""

from __future__ import annotations

import onnx
from onnx import helper, numpy_helper

try:  # sympy is a transitive dependency via onnxruntime tooling
    import sympy
except Exception:  # pragma: no cover - sympy unavailable
    sympy = None  # type: ignore[assignment]


_MAX_DEPTH = 256


def _static_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _has_dynamic_dims(graph: onnx.GraphProto) -> bool:
    for value_info in (*graph.value_info, *graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                return True
    return False


class _SymbolicEvaluator:
    def __init__(self, graph: onnx.GraphProto) -> None:
        self._graph = graph
        self._initializers = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        self._producer: dict[str, onnx.NodeProto] = {}
        for node in graph.node:
            for output in node.output:
                if output:
                    self._producer[output] = node
        self._input_info = {value_info.name: value_info for value_info in graph.input}
        self._static_shapes: dict[str, tuple[int, ...]] = {}
        for value_info in (*graph.input, *graph.value_info, *graph.output):
            shape = _static_shape(value_info)
            if shape is not None:
                self._static_shapes[value_info.name] = shape
        self._val_cache: dict[str, list | None] = {}
        self._shape_cache: dict[str, tuple | None] = {}
        self._symbol_count = 0
        self.symbols: dict[str, list] = {}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _attrs(node: onnx.NodeProto) -> dict[str, object]:
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    def _fresh_symbol(self, tag: str):
        self._symbol_count += 1
        return sympy.Symbol(f"{tag}__sym{self._symbol_count}", integer=True)

    # -- integer value evaluation ---------------------------------------
    def eval_value(self, name: str, depth: int = 0) -> list | None:
        if name in self._val_cache:
            return self._val_cache[name]
        if not name or depth > _MAX_DEPTH:
            return None
        self._val_cache[name] = None
        result: list | None = None
        if name in self._initializers:
            array = self._initializers[name]
            if array.dtype.kind in {"i", "u"} and array.ndim <= 1:
                result = [sympy.Integer(int(v)) for v in array.reshape(-1)]
        elif name in self._input_info and name not in self._initializers:
            result = self._symbolic_input(name)
        else:
            node = self._producer.get(name)
            if node is not None:
                result = self._eval_value_node(node, name, depth)
        self._val_cache[name] = result
        return result

    def _symbolic_input(self, name: str) -> list | None:
        tensor_type = self._input_info[name].type.tensor_type
        if tensor_type.elem_type not in {
            onnx.TensorProto.INT64,
            onnx.TensorProto.INT32,
        }:
            return None
        shape = self._static_shapes.get(name)
        if shape is None or len(shape) != 1:
            return None
        symbols = [self._fresh_symbol(name) for _ in range(shape[0])]
        self.symbols[name] = symbols
        return symbols

    def _eval_value_node(
        self, node: onnx.NodeProto, name: str, depth: int
    ) -> list | None:
        op = node.op_type
        attrs = self._attrs(node)
        if op == "Constant":
            if "value" in attrs:
                array = numpy_helper.to_array(attrs["value"])
                if array.dtype.kind in {"i", "u"} and array.ndim <= 1:
                    return [sympy.Integer(int(v)) for v in array.reshape(-1)]
                return None
            if "value_ints" in attrs:
                return [sympy.Integer(int(v)) for v in attrs["value_ints"]]
            if "value_int" in attrs:
                return [sympy.Integer(int(attrs["value_int"]))]
            return None
        if op in {"Cast", "Identity", "Flatten"}:
            return self.eval_value(node.input[0], depth + 1)
        if op == "Shape":
            shape = self.eval_shape(node.input[0], depth + 1)
            if shape is None:
                return None
            rank = len(shape)
            start = int(attrs.get("start", 0))
            end = attrs.get("end")
            end = rank if end is None else int(end)
            if start < 0:
                start += rank
            if end < 0:
                end += rank
            return list(shape[max(0, start) : max(0, end)])
        if op == "Size":
            shape = self.eval_shape(node.input[0], depth + 1)
            if shape is None:
                return None
            product = sympy.Integer(1)
            for dim in shape:
                product = product * dim
            return [product]
        if op == "Concat":
            values: list = []
            for input_name in node.input:
                part = self.eval_value(input_name, depth + 1)
                if part is None:
                    return None
                values.extend(part)
            return values
        if op == "Gather":
            return self._eval_gather(node, depth)
        if op == "Slice":
            return self._eval_slice_values(node, depth)
        if op in {"Squeeze", "Unsqueeze"}:
            # Reshaping a 1-D shape vector leaves its element order unchanged.
            return self.eval_value(node.input[0], depth + 1)
        if op == "Split":
            return self._eval_split(node, name, depth)
        if op in {"Add", "Sub", "Mul", "Div", "Mod", "Max", "Min"}:
            return self._eval_binary(op, node, depth)
        return None

    def _eval_gather(self, node: onnx.NodeProto, depth: int) -> list | None:
        data = self.eval_value(node.input[0], depth + 1)
        indices = self.eval_value(node.input[1], depth + 1)
        if data is None or indices is None:
            return None
        gathered: list = []
        for index in indices:
            if not index.is_Integer:
                return None
            position = int(index)
            if position < 0:
                position += len(data)
            if position < 0 or position >= len(data):
                return None
            gathered.append(data[position])
        return gathered

    def _eval_slice_values(self, node: onnx.NodeProto, depth: int) -> list | None:
        data = self.eval_value(node.input[0], depth + 1)
        if data is None:
            return None
        starts = self.eval_value(node.input[1], depth + 1)
        ends = self.eval_value(node.input[2], depth + 1)
        if starts is None or ends is None or len(starts) != 1 or len(ends) != 1:
            return None
        axes = (
            self.eval_value(node.input[3], depth + 1)
            if len(node.input) > 3 and node.input[3]
            else [sympy.Integer(0)]
        )
        steps = (
            self.eval_value(node.input[4], depth + 1)
            if len(node.input) > 4 and node.input[4]
            else [sympy.Integer(1)]
        )
        if axes is None or steps is None:
            return None
        if [int(a) for a in axes] != [0] or [int(s) for s in steps] != [1]:
            return None
        if not (starts[0].is_Integer and ends[0].is_Integer):
            return None
        start = int(starts[0])
        end = min(int(ends[0]), len(data))
        return data[start:end]

    def _eval_split(self, node: onnx.NodeProto, name: str, depth: int) -> list | None:
        data = self.eval_value(node.input[0], depth + 1)
        if data is None:
            return None
        outputs = list(node.output)
        if len(node.input) > 1 and node.input[1]:
            sizes_values = self.eval_value(node.input[1], depth + 1)
            if sizes_values is None or any(not s.is_Integer for s in sizes_values):
                return None
            sizes = [int(s) for s in sizes_values]
        else:
            count = len(outputs)
            if count == 0 or len(data) % count != 0:
                return None
            sizes = [len(data) // count] * count
        if sum(sizes) != len(data):
            return None
        offset = 0
        for output_name, size in zip(outputs, sizes):
            self._val_cache[output_name] = data[offset : offset + size]
            offset += size
        return self._val_cache.get(name)

    def _eval_binary(self, op: str, node: onnx.NodeProto, depth: int) -> list | None:
        left = self.eval_value(node.input[0], depth + 1)
        right = self.eval_value(node.input[1], depth + 1)
        if left is None or right is None:
            return None
        if len(left) == 1 and len(right) != 1:
            left = left * len(right)
        if len(right) == 1 and len(left) != 1:
            right = right * len(left)
        if len(left) != len(right):
            return None

        def divide(a, b):
            if a.is_Integer and b.is_Integer:
                return sympy.Integer(int(a) // int(b))
            return a / b

        ops = {
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mul": lambda a, b: a * b,
            "Div": divide,
            "Mod": lambda a, b: a % b,
            "Max": sympy.Max,
            "Min": sympy.Min,
        }
        func = ops[op]
        return [func(a, b) for a, b in zip(left, right)]

    # -- shape evaluation ------------------------------------------------
    def eval_shape(self, name: str, depth: int = 0) -> tuple | None:
        if name in self._shape_cache:
            return self._shape_cache[name]
        if not name or depth > _MAX_DEPTH:
            return None
        self._shape_cache[name] = None
        result: tuple | None = None
        if name in self._initializers:
            result = tuple(sympy.Integer(d) for d in self._initializers[name].shape)
        elif name in self._input_info and name in self._static_shapes:
            result = tuple(sympy.Integer(d) for d in self._static_shapes[name])
        else:
            node = self._producer.get(name)
            if node is not None:
                result = self._eval_shape_node(node, depth)
            elif name in self._static_shapes:
                result = tuple(sympy.Integer(d) for d in self._static_shapes[name])
        self._shape_cache[name] = result
        return result

    _SHAPE_PRESERVING = frozenset(
        {
            "Cast",
            "CastLike",
            "Identity",
            "Relu",
            "Neg",
            "Abs",
            "Sqrt",
            "Reciprocal",
            "Sigmoid",
            "Tanh",
            "Exp",
            "Log",
            "Erf",
            "Clip",
            "Elu",
            "LeakyRelu",
        }
    )

    def _eval_shape_node(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        op = node.op_type
        attrs = self._attrs(node)
        if op == "Reshape":
            return self._eval_reshape_shape(node, depth)
        if op == "ConstantOfShape":
            values = self.eval_value(node.input[0], depth + 1)
            return tuple(values) if values is not None else None
        if op in self._SHAPE_PRESERVING:
            return self.eval_shape(node.input[0], depth + 1)
        if op == "Transpose":
            return self._eval_transpose_shape(node, attrs, depth)
        if op == "Unsqueeze":
            return self._eval_unsqueeze_shape(node, attrs, depth)
        if op == "Squeeze":
            return self._eval_squeeze_shape(node, attrs, depth)
        if op == "Slice":
            return self._eval_slice_shape(node, depth)
        if op == "Pad":
            return self._eval_pad_shape(node, depth)
        if op == "MatMul":
            left = self.eval_shape(node.input[0], depth + 1)
            right = self.eval_shape(node.input[1], depth + 1)
            if left is None or right is None or len(right) < 1:
                return None
            return tuple([*left[:-1], right[-1]])
        if op == "Concat":
            return self._eval_concat_shape(node, attrs, depth)
        if op == "Range":
            return self._eval_range_shape(node, depth)
        if op == "Expand":
            return self._eval_expand_shape(node, depth)
        if op in self._ELEMENTWISE_BINARY:
            return self._eval_broadcast_shape(node, depth)
        return None

    _ELEMENTWISE_BINARY = frozenset(
        {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Mod",
            "Pow",
            "Max",
            "Min",
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
        }
    )

    def _eval_range_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        start = self.eval_value(node.input[0], depth + 1)
        limit = self.eval_value(node.input[1], depth + 1)
        delta = self.eval_value(node.input[2], depth + 1)
        if not (start and limit and delta):
            return None
        start_v, limit_v, delta_v = start[0], limit[0], delta[0]
        if delta_v == 1 and start_v == 0:
            length = limit_v
        elif delta_v.is_Integer and start_v.is_Integer:
            length = sympy.ceiling((limit_v - start_v) / delta_v)
        else:
            return None
        return (sympy.Max(sympy.Integer(0), length),)

    def _eval_expand_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        source = self.eval_shape(node.input[0], depth + 1)
        target = self.eval_value(node.input[1], depth + 1)
        if source is None or target is None:
            return None
        return self._broadcast(source, tuple(target))

    def _eval_broadcast_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        result: tuple | None = ()
        for input_name in node.input:
            shape = self.eval_shape(input_name, depth + 1)
            if shape is None:
                return None
            result = self._broadcast(result, shape)
            if result is None:
                return None
        return result

    @staticmethod
    def _broadcast(left: tuple, right: tuple) -> tuple | None:
        result: list = []
        for i in range(max(len(left), len(right))):
            a = left[-1 - i] if i < len(left) else sympy.Integer(1)
            b = right[-1 - i] if i < len(right) else sympy.Integer(1)
            if a.is_Integer and int(a) == 1:
                result.append(b)
            elif b.is_Integer and int(b) == 1:
                result.append(a)
            elif a == b:
                result.append(a)
            else:
                return None
        return tuple(reversed(result))

    def _eval_reshape_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        target = self.eval_value(node.input[1], depth + 1)
        if target is None:
            return None
        source = self.eval_shape(node.input[0], depth + 1)
        dims = list(target)
        for index, dim in enumerate(dims):
            if dim == 0 and source is not None and index < len(source):
                dims[index] = source[index]
        if any(dim == -1 for dim in dims):
            if source is None:
                return None
            total = sympy.Integer(1)
            for dim in source:
                total = total * dim
            known = sympy.Integer(1)
            for dim in dims:
                if dim != -1:
                    known = known * dim
            for index, dim in enumerate(dims):
                if dim == -1:
                    dims[index] = sympy.simplify(total / known)
        return tuple(dims)

    def _eval_transpose_shape(self, node, attrs, depth) -> tuple | None:
        shape = self.eval_shape(node.input[0], depth + 1)
        if shape is None:
            return None
        perm = attrs.get("perm")
        if perm is None:
            perm = list(range(len(shape)))[::-1]
        return tuple(shape[p] for p in perm)

    def _eval_unsqueeze_shape(self, node, attrs, depth) -> tuple | None:
        shape = self.eval_shape(node.input[0], depth + 1)
        axes = (
            self.eval_value(node.input[1], depth + 1)
            if len(node.input) > 1 and node.input[1]
            else attrs.get("axes")
        )
        if shape is None or axes is None:
            return None
        axes = [int(a) for a in axes]
        rank = len(shape) + len(axes)
        normalized = sorted((a + rank if a < 0 else a) for a in axes)
        dims = list(shape)
        for axis in normalized:
            dims.insert(axis, sympy.Integer(1))
        return tuple(dims)

    def _eval_squeeze_shape(self, node, attrs, depth) -> tuple | None:
        shape = self.eval_shape(node.input[0], depth + 1)
        if shape is None:
            return None
        axes = (
            self.eval_value(node.input[1], depth + 1)
            if len(node.input) > 1 and node.input[1]
            else attrs.get("axes")
        )
        if axes is None:
            return tuple(dim for dim in shape if dim != 1)
        rank = len(shape)
        normalized = {int(a) + rank if int(a) < 0 else int(a) for a in axes}
        return tuple(dim for index, dim in enumerate(shape) if index not in normalized)

    def _eval_slice_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        shape = self.eval_shape(node.input[0], depth + 1)
        if shape is None:
            return None
        starts = self.eval_value(node.input[1], depth + 1)
        ends = self.eval_value(node.input[2], depth + 1)
        if starts is None or ends is None:
            return None
        axes = (
            self.eval_value(node.input[3], depth + 1)
            if len(node.input) > 3 and node.input[3]
            else None
        )
        steps = (
            self.eval_value(node.input[4], depth + 1)
            if len(node.input) > 4 and node.input[4]
            else None
        )
        rank = len(shape)
        axis_list = (
            [int(a) for a in axes] if axes is not None else list(range(len(starts)))
        )
        step_list = [int(s) for s in steps] if steps is not None else [1] * len(starts)
        dims = list(shape)
        for index, axis in enumerate(axis_list):
            if step_list[index] != 1:
                return None
            axis = axis + rank if axis < 0 else axis
            start = starts[index]
            end = ends[index]
            dim = shape[axis]
            if start.is_Integer and end.is_Integer and dim.is_Integer:
                extent = int(dim)
                start_i = int(start) + extent if int(start) < 0 else int(start)
                end_i = int(end) + extent if int(end) < 0 else int(end)
                end_i = min(end_i, extent)
                start_i = max(0, min(start_i, extent))
                dims[axis] = sympy.Integer(max(0, end_i - start_i))
            else:
                # Symbolic crop: trust end - start (clamping is not modelled).
                dims[axis] = sympy.simplify(end - start)
        return tuple(dims)

    def _eval_pad_shape(self, node: onnx.NodeProto, depth: int) -> tuple | None:
        shape = self.eval_shape(node.input[0], depth + 1)
        pads = (
            self.eval_value(node.input[1], depth + 1)
            if len(node.input) > 1 and node.input[1]
            else None
        )
        if shape is None or pads is None:
            return None
        rank = len(shape)
        # Pad-18 adds an optional `axes` input; pads then covers only those axes.
        if len(node.input) > 3 and node.input[3]:
            axes = self.eval_value(node.input[3], depth + 1)
            if axes is None:
                return None
            axes = [int(a) + rank if int(a) < 0 else int(a) for a in axes]
        else:
            axes = list(range(rank))
        if len(pads) != 2 * len(axes):
            return None
        dims = list(shape)
        for index, axis in enumerate(axes):
            dims[axis] = dims[axis] + pads[index] + pads[index + len(axes)]
        return tuple(dims)

    def _eval_concat_shape(self, node, attrs, depth) -> tuple | None:
        shapes = [self.eval_shape(input_name, depth + 1) for input_name in node.input]
        if any(shape is None for shape in shapes):
            return None
        rank = len(shapes[0])
        axis = int(attrs.get("axis", 0))
        axis = axis + rank if axis < 0 else axis
        dims = list(shapes[0])
        total = sympy.Integer(0)
        for shape in shapes:
            total = total + shape[axis]
        dims[axis] = total
        return tuple(dims)


def infer_symbolic_shapes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    """Fill dynamic value_info dims that are determined by runtime integer inputs.

    Returns ``(model, changed)`` where ``changed`` indicates whether any concrete
    dimension was recovered and written back.
    """

    if sympy is None:
        return model, False
    graph = model.graph
    if not _has_dynamic_dims(graph):
        return model, False

    evaluator = _SymbolicEvaluator(graph)

    # Collect equations: every value with a declared concrete shape constrains
    # its symbolically inferred shape.
    equations = []
    constrained_names = {value_info.name for value_info in graph.output}
    constrained_names.update(evaluator._static_shapes)
    for name in constrained_names:
        declared = evaluator._static_shapes.get(name)
        if declared is None:
            continue
        inferred = evaluator.eval_shape(name)
        if inferred is None or len(inferred) != len(declared):
            continue
        for declared_dim, inferred_dim in zip(declared, inferred):
            if inferred_dim is None or inferred_dim.is_Integer:
                continue
            equations.append(sympy.Eq(inferred_dim, declared_dim))

    if not equations:
        return model, False
    try:
        solutions = sympy.solve(equations, dict=True)
    except Exception:  # pragma: no cover - solver failure is non-fatal
        return model, False
    if not solutions:
        return model, False
    solution = solutions[0]
    if not solution:
        return model, False

    # Substitute the solved symbols into every tensor's inferred shape and write
    # back the ones that become fully concrete and non-negative.
    resolved: dict[str, tuple[int, ...]] = {}
    names = {name for node in graph.node for name in node.output if name}
    names.update(value_info.name for value_info in graph.output)
    for name in names:
        if name in evaluator._static_shapes:
            continue
        inferred = evaluator.eval_shape(name)
        if inferred is None:
            continue
        concrete: list[int] = []
        ok = True
        for dim in inferred:
            if dim is None:
                ok = False
                break
            value = dim.subs(solution)
            if not value.is_Integer or int(value) < 0:
                ok = False
                break
            concrete.append(int(value))
        if ok:
            resolved[name] = tuple(concrete)

    if not resolved:
        return model, False
    _apply_resolved_shapes(graph, resolved)
    return model, True


def _apply_resolved_shapes(
    graph: onnx.GraphProto, resolved: dict[str, tuple[int, ...]]
) -> None:
    for value_info in graph.value_info:
        _maybe_set_dims(value_info, resolved.get(value_info.name))
    for value_info in graph.output:
        _maybe_set_dims(value_info, resolved.get(value_info.name))


def _maybe_set_dims(
    value_info: onnx.ValueInfoProto, shape: tuple[int, ...] | None
) -> None:
    if shape is None:
        return
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return
    dims = tensor_type.shape.dim
    if len(dims) != len(shape):
        return
    for dim, value in zip(dims, shape):
        if dim.HasField("dim_value"):
            # Never overwrite an existing concrete dimension.
            continue
        dim.ClearField("dim_param")
        dim.dim_value = value
