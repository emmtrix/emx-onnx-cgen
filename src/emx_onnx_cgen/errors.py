class CompilerError(RuntimeError):
    """Base error for compiler failures."""


class UnsupportedOpError(CompilerError):
    """Raised when an ONNX operator is not supported."""


class ShapeInferenceError(CompilerError):
    """Raised when tensor shapes cannot be resolved."""


class CodegenError(CompilerError):
    """Raised when C code generation fails."""
