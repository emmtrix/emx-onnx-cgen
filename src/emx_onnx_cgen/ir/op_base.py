from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .context import GraphContext


class Emitter(Protocol):
    def render_op(self, op: "OpBase", ctx: "EmitContext") -> str:
        ...


@dataclass(frozen=True)
class EmitContext:
    op_index: int


class OpBase(ABC):
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __getattr__(self, name: str) -> str:
        if name == "kind":
            return self.__class__.__name__
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def validate(self, ctx: GraphContext) -> None:
        return None

    def infer_types(self, ctx: GraphContext) -> None:
        return None

    def infer_shapes(self, ctx: GraphContext) -> None:
        return None

    @abstractmethod
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        raise NotImplementedError


class RenderableOpBase(OpBase):
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.render_op(self, ctx)


class ElementwiseOpBase(RenderableOpBase):
    pass


class ReduceOpBase(RenderableOpBase):
    pass


class BroadcastingOpBase(RenderableOpBase):
    pass


class MatMulLikeOpBase(RenderableOpBase):
    pass


class GemmLikeOpBase(RenderableOpBase):
    pass


class ConvLikeOpBase(RenderableOpBase):
    pass
