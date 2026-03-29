class CompilerError(RuntimeError):
    """Base error for compiler failures."""


class UnsupportedOpError(CompilerError):
    """Raised when an ONNX operator is not supported."""


def format_onnx_operator_name(op_type: str, domain: str | None = None) -> str:
    op_domain = domain or "ai.onnx"
    return f"{op_domain}.{op_type}"


def unsupported_op_message(op_type: str, domain: str | None = None) -> str:
    return f"Unsupported op {format_onnx_operator_name(op_type, domain)}"


class ShapeInferenceError(CompilerError):
    """Raised when tensor shapes cannot be resolved."""


class CodegenError(CompilerError):
    """Raised when C code generation fails."""
