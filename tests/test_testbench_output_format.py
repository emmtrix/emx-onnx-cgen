from __future__ import annotations

import pytest

from emx_onnx_cgen.testbench_output_format import parse_testbench_output_format


def test_parse_testbench_output_format_txt_emmtrix_defaults_to_1000() -> None:
    parsed = parse_testbench_output_format("txt-emmtrix")

    assert parsed.kind == "txt-emmtrix"
    assert parsed.emmtrix_ulp == 1000.0
    assert parsed.emmtrix_ulp_tag == "1000"


def test_parse_testbench_output_format_txt_emmtrix_with_float() -> None:
    parsed = parse_testbench_output_format("txt-emmtrix:0.125")

    assert parsed.kind == "txt-emmtrix"
    assert parsed.emmtrix_ulp == 0.125
    assert parsed.emmtrix_ulp_tag == "0.125"


def test_parse_testbench_output_format_rejects_invalid_txt_emmtrix_float() -> None:
    with pytest.raises(ValueError, match="must be a float"):
        parse_testbench_output_format("txt-emmtrix:foo")
