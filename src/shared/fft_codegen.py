from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

FFTStageKind = Literal["radix2", "radix4"]
FFTVariant = Literal["dft", "radix2", "radix4"]


@dataclass(frozen=True)
class FFTStage:
    kind: FFTStageKind
    m: int
    stage_span: int


@dataclass(frozen=True)
class FFTCodegenPlan:
    fft_length: int
    variant: FFTVariant
    stages: tuple[FFTStage, ...]
    input_permutation: tuple[int, ...]


def build_fft_codegen_plan(fft_length: int) -> FFTCodegenPlan:
    if fft_length <= 0:
        raise ValueError("fft_length must be > 0")
    stages = _fft_stockham_stage_plan(fft_length)
    if not stages:
        return FFTCodegenPlan(
            fft_length=fft_length,
            variant="dft",
            stages=(),
            input_permutation=tuple(range(fft_length)),
        )
    stage_kinds = {stage.kind for stage in stages}
    if stage_kinds == {"radix2"}:
        variant: FFTVariant = "radix2"
    elif stage_kinds == {"radix4"}:
        variant = "radix4"
    else:
        raise ValueError("mixed FFT stage plans are not supported")
    return FFTCodegenPlan(
        fft_length=fft_length,
        variant=variant,
        stages=stages,
        input_permutation=fft_input_permutation(fft_length, stages=stages),
    )


def fft_twiddle_coefficients(fft_length: int) -> tuple[tuple[float, float], ...]:
    if fft_length <= 0:
        raise ValueError("fft_length must be > 0")
    twiddles: list[tuple[float, float]] = []
    for index in range(fft_length):
        angle = -2.0 * math.pi * (index / fft_length)
        twiddles.append((math.cos(angle), math.sin(angle)))
    return tuple(twiddles)


def fft_input_permutation(
    fft_length: int,
    *,
    stages: tuple[FFTStage, ...],
) -> tuple[int, ...]:
    if fft_length <= 0:
        raise ValueError("fft_length must be > 0")
    if not stages:
        return tuple(range(fft_length))
    kinds = {stage.kind for stage in stages}
    if kinds == {"radix2"}:
        bits = fft_length.bit_length() - 1

        def _bit_reverse(index: int) -> int:
            reversed_index = 0
            value = index
            for _ in range(bits):
                reversed_index = (reversed_index << 1) | (value & 1)
                value >>= 1
            return reversed_index

        return tuple(_bit_reverse(index) for index in range(fft_length))
    if kinds == {"radix4"}:
        digits = len(stages)

        def _base4_reverse(index: int) -> int:
            reversed_index = 0
            value = index
            for _ in range(digits):
                reversed_index = (reversed_index << 2) | (value & 0x3)
                value >>= 2
            return reversed_index

        return tuple(_base4_reverse(index) for index in range(fft_length))
    raise ValueError("mixed FFT stage plans are not supported")


def _fft_stockham_stage_plan(fft_length: int) -> tuple[FFTStage, ...]:
    if fft_length <= 1:
        return ()
    if fft_length & (fft_length - 1):
        return ()
    stages: list[FFTStage] = []
    base4_value = fft_length
    while base4_value % 4 == 0:
        base4_value //= 4
    if base4_value == 1:
        m = 1
        while m < fft_length:
            stage_span = fft_length // (4 * m)
            stages.append(FFTStage(kind="radix4", m=m, stage_span=stage_span))
            m *= 4
        return tuple(stages)
    m = 1
    while m < fft_length:
        stage_span = fft_length // (2 * m)
        stages.append(FFTStage(kind="radix2", m=m, stage_span=stage_span))
        m *= 2
    return tuple(stages)


__all__ = [
    "FFTCodegenPlan",
    "FFTStage",
    "FFTStageKind",
    "FFTVariant",
    "build_fft_codegen_plan",
    "fft_input_permutation",
    "fft_twiddle_coefficients",
]
