from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedTestbenchOutputFormat:
    kind: str
    emmtrix_ulp: float | None = None

    @property
    def emmtrix_ulp_tag(self) -> str:
        if self.kind != "txt-emmtrix":
            raise ValueError("ULP tag is only available for txt-emmtrix format")
        assert self.emmtrix_ulp is not None
        return format(self.emmtrix_ulp, "g")


def parse_testbench_output_format(value: str) -> ParsedTestbenchOutputFormat:
    raw = value.strip()
    if raw in {"json", "txt"}:
        return ParsedTestbenchOutputFormat(kind=raw)
    if raw == "txt-emmtrix":
        return ParsedTestbenchOutputFormat(kind="txt-emmtrix", emmtrix_ulp=1000.0)
    prefix = "txt-emmtrix:"
    if raw.startswith(prefix):
        ulp_value = raw[len(prefix) :].strip()
        if not ulp_value:
            raise ValueError(
                "Invalid testbench output format 'txt-emmtrix:'; missing ULP value"
            )
        try:
            ulp = float(ulp_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid testbench output format {value!r}; "
                "txt-emmtrix ULP must be a float"
            ) from exc
        return ParsedTestbenchOutputFormat(kind="txt-emmtrix", emmtrix_ulp=ulp)
    raise ValueError(
        f"Unsupported testbench output format {value!r}; expected 'json', 'txt', "
        "'txt-emmtrix', or 'txt-emmtrix:<float>'"
    )
