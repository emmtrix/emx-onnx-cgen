from __future__ import annotations

from ..ir.ops import DeformConvOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("DeformConv")
def lower_deform_conv(graph: Graph, node: Node) -> DeformConvOp:
    if len(node.inputs) not in {3, 4, 5} or len(node.outputs) != 1:
        raise UnsupportedOpError("DeformConv must have 3–5 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "DeformConv supports float16, float, and double inputs only"
        )
    input_name = node.inputs[0]
    weight_name = node.inputs[1]
    offset_name = node.inputs[2]
    bias_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    mask_name = node.inputs[4] if len(node.inputs) == 5 and node.inputs[4] else None

    input_shape = _value_shape(graph, input_name, node)
    weight_shape = _value_shape(graph, weight_name, node)
    offset_shape = _value_shape(graph, offset_name, node)

    if len(input_shape) != 4:
        raise UnsupportedOpError("DeformConv only supports 2D (NCHW) inputs")
    if len(weight_shape) != 4:
        raise UnsupportedOpError("DeformConv weight must have rank 4 (oC, C/g, kH, kW)")

    batch, in_channels = input_shape[0], input_shape[1]
    in_h, in_w = input_shape[2], input_shape[3]
    out_channels = weight_shape[0]
    kernel_h, kernel_w = weight_shape[2], weight_shape[3]

    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise UnsupportedOpError("DeformConv expects group >= 1")
    if in_channels % group != 0 or out_channels % group != 0:
        raise ShapeInferenceError(
            "DeformConv group must evenly divide in/out channels, "
            f"got group={group}, in_channels={in_channels}, out_channels={out_channels}"
        )
    if weight_shape[1] != in_channels // group:
        raise ShapeInferenceError(
            "DeformConv weight in-channel dim must equal in_channels/group, "
            f"got {weight_shape[1]} vs {in_channels // group}"
        )

    offset_group = int(node.attrs.get("offset_group", 1))
    if offset_group <= 0:
        raise UnsupportedOpError("DeformConv expects offset_group >= 1")
    if in_channels % offset_group != 0:
        raise ShapeInferenceError(
            "DeformConv offset_group must evenly divide in_channels, "
            f"got offset_group={offset_group}, in_channels={in_channels}"
        )

    kernel_shape_attr = node.attrs.get("kernel_shape")
    if kernel_shape_attr is not None:
        ks_h, ks_w = int(kernel_shape_attr[0]), int(kernel_shape_attr[1])
        if ks_h != kernel_h or ks_w != kernel_w:
            raise ShapeInferenceError(
                f"DeformConv kernel_shape {(ks_h, ks_w)} must match weight shape "
                f"{(kernel_h, kernel_w)}"
            )

    strides = tuple(int(v) for v in node.attrs.get("strides", (1, 1)))
    if len(strides) != 2:
        raise UnsupportedOpError("DeformConv strides must have 2 elements")
    stride_h, stride_w = strides

    dilations = tuple(int(v) for v in node.attrs.get("dilations", (1, 1)))
    if len(dilations) != 2:
        raise UnsupportedOpError("DeformConv dilations must have 2 elements")
    dilation_h, dilation_w = dilations

    pads = tuple(int(v) for v in node.attrs.get("pads", (0, 0, 0, 0)))
    if len(pads) != 4:
        raise UnsupportedOpError("DeformConv pads must have 4 elements")
    pad_top, pad_left = pads[0], pads[1]

    eff_kernel_h = (kernel_h - 1) * dilation_h + 1
    eff_kernel_w = (kernel_w - 1) * dilation_w + 1
    out_h = (in_h + pads[0] + pads[2] - eff_kernel_h) // stride_h + 1
    out_w = (in_w + pads[1] + pads[3] - eff_kernel_w) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ShapeInferenceError("DeformConv output spatial dimensions must be > 0")

    expected_offset_channels = offset_group * kernel_h * kernel_w * 2
    if len(offset_shape) != 4:
        raise ShapeInferenceError(
            f"DeformConv offset must have rank 4, got {len(offset_shape)}"
        )
    if offset_shape[1] != expected_offset_channels:
        raise ShapeInferenceError(
            f"DeformConv offset channel dimension must be "
            f"offset_group*kH*kW*2={expected_offset_channels}, got {offset_shape[1]}"
        )
    if offset_shape[2] != out_h or offset_shape[3] != out_w:
        raise ShapeInferenceError(
            f"DeformConv offset spatial dims must match output {(out_h, out_w)}, "
            f"got {(offset_shape[2], offset_shape[3])}"
        )

    if bias_name is not None:
        bias_shape = _value_shape(graph, bias_name, node)
        if bias_shape != (out_channels,):
            raise ShapeInferenceError(
                f"DeformConv bias shape must be ({out_channels},), got {bias_shape}"
            )

    if mask_name is not None:
        mask_shape = _value_shape(graph, mask_name, node)
        expected_mask_channels = offset_group * kernel_h * kernel_w
        if len(mask_shape) != 4:
            raise ShapeInferenceError(
                f"DeformConv mask must have rank 4, got {len(mask_shape)}"
            )
        if mask_shape[1] != expected_mask_channels:
            raise ShapeInferenceError(
                f"DeformConv mask channel dimension must be "
                f"offset_group*kH*kW={expected_mask_channels}, got {mask_shape[1]}"
            )
        if mask_shape[2] != out_h or mask_shape[3] != out_w:
            raise ShapeInferenceError(
                f"DeformConv mask spatial dims must match output {(out_h, out_w)}, "
                f"got {(mask_shape[2], mask_shape[3])}"
            )

    expected_output_shape = (batch, out_channels, out_h, out_w)
    try:
        output_shape = _value_shape(graph, node.outputs[0], node)
    except ShapeInferenceError:
        output_shape = None
    if output_shape is not None and output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"DeformConv output shape must be {expected_output_shape}, "
            f"got {output_shape}"
        )
    if output_shape is None:
        raise ShapeInferenceError(
            f"Missing shape for value '{node.outputs[0]}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        )

    return DeformConvOp(
        input0=input_name,
        weights=weight_name,
        offset=offset_name,
        bias=bias_name,
        mask=mask_name,
        output=node.outputs[0],
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_top=pad_top,
        pad_left=pad_left,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        group=group,
        offset_group=offset_group,
        dtype=op_dtype,
    )
