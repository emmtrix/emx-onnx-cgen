"""ONNX backend adapter for emx-onnx-cgen."""

from .backend import (
    EmxOnnxCgenBackend,
    backend_name,
    prepare,
    run_model,
    supports_device,
)

__all__ = [
    "EmxOnnxCgenBackend",
    "backend_name",
    "prepare",
    "run_model",
    "supports_device",
]
