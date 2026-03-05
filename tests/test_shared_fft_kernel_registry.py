from __future__ import annotations

from shared.fft_kernel_registry import FFTKernelRegistry
from shared.scalar_types import ScalarType


def test_fft_kernel_registry_deduplicates_requests() -> None:
    registry = FFTKernelRegistry()
    name_first = registry.request(dtype=ScalarType.F32, fft_length=16)
    name_second = registry.request(dtype=ScalarType.F32, fft_length=16)
    assert name_first == name_second
    rendered = "\n".join(registry.render())
    assert rendered.count(f"static void {name_first}(") == 1


def test_fft_kernel_registry_emits_radix2_kernel_for_length_8() -> None:
    registry = FFTKernelRegistry()
    name = registry.request(dtype=ScalarType.F32, fft_length=8)
    rendered = "\n".join(registry.render())
    assert f"static void {name}(" in rendered
    assert "input_perm[8]" in rendered
    assert "block < 8; block += 2" in rendered
    assert "block < 8; block += 4" in rendered
    assert "block < 8; block += 8" in rendered


def test_fft_kernel_registry_emits_dft_fallback_for_non_power_of_two() -> None:
    registry = FFTKernelRegistry()
    name = registry.request(dtype=ScalarType.F64, fft_length=12)
    rendered = "\n".join(registry.render())
    assert f"static void {name}(" in rendered
    assert "for (idx_t k = 0; k < 12; ++k)" in rendered
    assert "input_perm[12]" not in rendered
