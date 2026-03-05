from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Set

from shared.fft_codegen import FFTCodegenPlan, build_fft_codegen_plan, fft_twiddle_coefficients
from shared.scalar_types import ScalarType

LiteralFormatter = Callable[[ScalarType, float | int | bool], str]


class FFTKernelError(RuntimeError):
    pass


@dataclass(frozen=True)
class FFTKernelKey:
    dtype: ScalarType
    fft_length: int


@dataclass(frozen=True)
class _GeneratedFFTKernel:
    lines: List[str]
    includes: Set[str]


def _default_literal_formatter(dtype: ScalarType, value: float | int | bool) -> str:
    if not dtype.is_float:
        raise FFTKernelError(f"FFT kernels require a float dtype, got {dtype.onnx_name}")
    numeric = float(value)
    if math.isnan(numeric):
        return "NAN"
    if math.isinf(numeric):
        return "INFINITY" if numeric > 0.0 else "-INFINITY"
    if dtype == ScalarType.F32:
        return f"{numeric:.9g}f"
    return f"{numeric:.17g}"


class FFTKernelRegistry:
    def __init__(self, *, literal_formatter: LiteralFormatter | None = None) -> None:
        self._literal_formatter = literal_formatter or _default_literal_formatter
        self._requested: List[FFTKernelKey] = []
        self._requested_set: Set[FFTKernelKey] = set()
        self._key_to_name: Dict[FFTKernelKey, str] = {}
        self._generated: Dict[FFTKernelKey, _GeneratedFFTKernel] = {}

    def request(self, *, dtype: object, fft_length: int) -> str:
        scalar_dtype = ScalarType.from_torch_dtype(dtype)
        if not scalar_dtype.is_float:
            raise FFTKernelError(
                f"FFT kernels require a float dtype, got {scalar_dtype.onnx_name}"
            )
        if fft_length <= 0:
            raise FFTKernelError("fft_length must be > 0")
        key = FFTKernelKey(dtype=scalar_dtype, fft_length=int(fft_length))
        name = self._key_to_name.get(key)
        if name is None:
            plan = build_fft_codegen_plan(key.fft_length)
            name = (
                f"emx_fft_kernel_{key.dtype.suffix}_{key.fft_length}_{plan.variant}"
            )
            self._key_to_name[key] = name
        if key not in self._requested_set:
            self._requested.append(key)
            self._requested_set.add(key)
        return name

    def include_lines(self) -> List[str]:
        includes: Set[str] = set()
        for key in self._requested:
            self._ensure_generated(key)
            includes.update(self._generated[key].includes)
        return sorted(includes)

    def render(self) -> List[str]:
        if not self._requested:
            return []
        lines: List[str] = []
        for key in self._requested:
            self._ensure_generated(key)
            lines.extend(self._generated[key].lines)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _ensure_generated(self, key: FFTKernelKey) -> None:
        if key in self._generated:
            return
        plan = build_fft_codegen_plan(key.fft_length)
        self._generated[key] = self._generate_kernel(key, plan=plan)

    def _generate_kernel(
        self,
        key: FFTKernelKey,
        *,
        plan: FFTCodegenPlan,
    ) -> _GeneratedFFTKernel:
        kernel_name = self._key_to_name[key]
        fft_length = key.fft_length
        c_type = key.dtype.c_type
        twiddles = fft_twiddle_coefficients(fft_length)
        twiddle_re = ", ".join(
            self._literal_formatter(key.dtype, value) for value, _ in twiddles
        )
        twiddle_im = ", ".join(
            self._literal_formatter(key.dtype, value) for _, value in twiddles
        )
        lines = [
            f"static void {kernel_name}(",
            f"    const {c_type} input_re[{fft_length}],",
            f"    const {c_type} input_im[{fft_length}],",
            f"    {c_type} output_re[{fft_length}],",
            f"    {c_type} output_im[{fft_length}]",
            ") {",
            f"    static const {c_type} twiddle_re[{fft_length}] = {{ {twiddle_re} }};",
            f"    static const {c_type} twiddle_im[{fft_length}] = {{ {twiddle_im} }};",
        ]
        if plan.variant != "dft":
            permutation_values = ", ".join(str(value) for value in plan.input_permutation)
            lines.append(
                f"    static const idx_t input_perm[{fft_length}] = {{ {permutation_values} }};"
            )
            lines.extend(
                [
                    f"    for (idx_t index = 0; index < {fft_length}; ++index) {{",
                    "        const idx_t permuted = input_perm[index];",
                    "        output_re[permuted] = input_re[index];",
                    "        output_im[permuted] = input_im[index];",
                    "    }",
                ]
            )
            for stage in plan.stages:
                if stage.kind == "radix2":
                    lines.extend(
                        [
                            f"    for (idx_t block = 0; block < {fft_length}; block += {2 * stage.m}) {{",
                            f"        for (idx_t j = 0; j < {stage.m}; ++j) {{",
                            f"            const idx_t tw_index = (idx_t)((j * {stage.stage_span}) % {fft_length});",
                            "            const "
                            f"{c_type} w_re = twiddle_re[tw_index];",
                            "            const "
                            f"{c_type} w_im = twiddle_im[tw_index];",
                            "            const idx_t i0 = block + j;",
                            f"            const idx_t i1 = i0 + {stage.m};",
                            f"            const {c_type} b_re = output_re[i1];",
                            f"            const {c_type} b_im = output_im[i1];",
                            f"            const {c_type} t_re = (b_re * w_re) - (b_im * w_im);",
                            f"            const {c_type} t_im = (b_re * w_im) + (b_im * w_re);",
                            f"            const {c_type} a_re = output_re[i0];",
                            f"            const {c_type} a_im = output_im[i0];",
                            "            output_re[i0] = a_re + t_re;",
                            "            output_im[i0] = a_im + t_im;",
                            "            output_re[i1] = a_re - t_re;",
                            "            output_im[i1] = a_im - t_im;",
                            "        }",
                            "    }",
                        ]
                    )
                    continue
                if stage.kind != "radix4":
                    raise FFTKernelError(f"unsupported FFT stage kind: {stage.kind}")
                lines.extend(
                    [
                        f"    for (idx_t block = 0; block < {fft_length}; block += {4 * stage.m}) {{",
                        f"        for (idx_t j = 0; j < {stage.m}; ++j) {{",
                        f"            const idx_t tw1 = (idx_t)((j * {stage.stage_span}) % {fft_length});",
                        f"            const idx_t tw2 = (idx_t)(((2 * j) * {stage.stage_span}) % {fft_length});",
                        f"            const idx_t tw3 = (idx_t)(((3 * j) * {stage.stage_span}) % {fft_length});",
                        "            const idx_t i0 = block + j;",
                        f"            const idx_t i1 = i0 + {stage.m};",
                        f"            const idx_t i2 = i1 + {stage.m};",
                        f"            const idx_t i3 = i2 + {stage.m};",
                        f"            const {c_type} x0_re = output_re[i0];",
                        f"            const {c_type} x0_im = output_im[i0];",
                        f"            const {c_type} x1_src_re = output_re[i1];",
                        f"            const {c_type} x1_src_im = output_im[i1];",
                        f"            const {c_type} x2_src_re = output_re[i2];",
                        f"            const {c_type} x2_src_im = output_im[i2];",
                        f"            const {c_type} x3_src_re = output_re[i3];",
                        f"            const {c_type} x3_src_im = output_im[i3];",
                        f"            const {c_type} x1_re = (x1_src_re * twiddle_re[tw1]) - (x1_src_im * twiddle_im[tw1]);",
                        f"            const {c_type} x1_im = (x1_src_re * twiddle_im[tw1]) + (x1_src_im * twiddle_re[tw1]);",
                        f"            const {c_type} x2_re = (x2_src_re * twiddle_re[tw2]) - (x2_src_im * twiddle_im[tw2]);",
                        f"            const {c_type} x2_im = (x2_src_re * twiddle_im[tw2]) + (x2_src_im * twiddle_re[tw2]);",
                        f"            const {c_type} x3_re = (x3_src_re * twiddle_re[tw3]) - (x3_src_im * twiddle_im[tw3]);",
                        f"            const {c_type} x3_im = (x3_src_re * twiddle_im[tw3]) + (x3_src_im * twiddle_re[tw3]);",
                        f"            const {c_type} s02_re = x0_re + x2_re;",
                        f"            const {c_type} s02_im = x0_im + x2_im;",
                        f"            const {c_type} d02_re = x0_re - x2_re;",
                        f"            const {c_type} d02_im = x0_im - x2_im;",
                        f"            const {c_type} s13_re = x1_re + x3_re;",
                        f"            const {c_type} s13_im = x1_im + x3_im;",
                        f"            const {c_type} d13_re = x1_re - x3_re;",
                        f"            const {c_type} d13_im = x1_im - x3_im;",
                        f"            const {c_type} minus_i_d13_re = d13_im;",
                        f"            const {c_type} minus_i_d13_im = -d13_re;",
                        "            output_re[i0] = s02_re + s13_re;",
                        "            output_im[i0] = s02_im + s13_im;",
                        "            output_re[i2] = s02_re - s13_re;",
                        "            output_im[i2] = s02_im - s13_im;",
                        "            output_re[i1] = d02_re + minus_i_d13_re;",
                        "            output_im[i1] = d02_im + minus_i_d13_im;",
                        "            output_re[i3] = d02_re - minus_i_d13_re;",
                        "            output_im[i3] = d02_im - minus_i_d13_im;",
                        "        }",
                        "    }",
                    ]
                )
        else:
            lines.extend(
                [
                    f"    for (idx_t k = 0; k < {fft_length}; ++k) {{",
                    f"        {c_type} acc_re = {self._literal_formatter(key.dtype, 0.0)};",
                    f"        {c_type} acc_im = {self._literal_formatter(key.dtype, 0.0)};",
                    f"        for (idx_t n = 0; n < {fft_length}; ++n) {{",
                    f"            const idx_t tw_index = (idx_t)((k * n) % {fft_length});",
                    f"            const {c_type} w_re = twiddle_re[tw_index];",
                    f"            const {c_type} w_im = twiddle_im[tw_index];",
                    f"            const {c_type} in_re = input_re[n];",
                    f"            const {c_type} in_im = input_im[n];",
                    "            acc_re += (in_re * w_re) - (in_im * w_im);",
                    "            acc_im += (in_re * w_im) + (in_im * w_re);",
                    "        }",
                    "        output_re[k] = acc_re;",
                    "        output_im[k] = acc_im;",
                    "    }",
                ]
            )
        lines.append("}")
        return _GeneratedFFTKernel(lines=lines, includes=set())


__all__ = [
    "FFTKernelError",
    "FFTKernelRegistry",
]
