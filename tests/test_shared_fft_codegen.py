from __future__ import annotations

import pytest

from shared.fft_codegen import build_fft_codegen_plan, fft_twiddle_coefficients


def test_fft_codegen_plan_prefers_radix4_for_power_of_four() -> None:
    plan = build_fft_codegen_plan(16)
    assert plan.variant == "radix4"
    assert tuple((stage.kind, stage.m, stage.stage_span) for stage in plan.stages) == (
        ("radix4", 1, 4),
        ("radix4", 4, 1),
    )
    assert plan.input_permutation == (
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    )


def test_fft_codegen_plan_uses_radix2_for_non_power_of_four_power_of_two() -> None:
    plan = build_fft_codegen_plan(8)
    assert plan.variant == "radix2"
    assert tuple((stage.kind, stage.m, stage.stage_span) for stage in plan.stages) == (
        ("radix2", 1, 4),
        ("radix2", 2, 2),
        ("radix2", 4, 1),
    )
    assert plan.input_permutation == (0, 4, 2, 6, 1, 5, 3, 7)


def test_fft_codegen_plan_falls_back_to_dft_for_non_power_of_two() -> None:
    plan = build_fft_codegen_plan(12)
    assert plan.variant == "dft"
    assert plan.stages == ()
    assert plan.input_permutation == tuple(range(12))


def test_fft_twiddle_coefficients_are_expected_for_quarter_turn() -> None:
    twiddles = fft_twiddle_coefficients(8)
    assert twiddles[0] == pytest.approx((1.0, 0.0))
    assert twiddles[2] == pytest.approx((0.0, -1.0))
    assert twiddles[4] == pytest.approx((-1.0, 0.0))
    assert twiddles[6] == pytest.approx((0.0, 1.0))
    for real, imag in twiddles:
        assert (real * real) + (imag * imag) == pytest.approx(1.0, abs=1e-12)


def test_fft_codegen_plan_rejects_non_positive_lengths() -> None:
    with pytest.raises(ValueError, match="fft_length must be > 0"):
        build_fft_codegen_plan(0)
    with pytest.raises(ValueError, match="fft_length must be > 0"):
        fft_twiddle_coefficients(-1)
