# Pytest Speed Report Script

This repository includes a small helper that runs `pytest` with duration reporting
and writes a Markdown report with the slowest tests plus a run summary.

## Usage

```bash
python tools/pytest_speed_report.py --output reports/pytest_speed_report.md --top 25 -- -n auto -q
```

### Arguments

- `--output`: Path to write the report file (default: `pytest_speed_report.md`).
- `--top`: Number of slowest tests to include (default: `20`).
- `--`: Separator for passing additional `pytest` arguments.

## Example Report Output

```markdown
# Pytest Speed Report

Generated: 2025-01-01T00:00:00+00:00

## Run Summary
- Command: `python -m pytest --durations=0 --durations-min=0 -n auto -q`
- Exit code: 0
- Summary: 253 passed, 2 skipped
- Total duration: 66.260s

## Slowest Durations

| Rank | Duration (s) | Phase | Test |
| ---: | ---: | --- | --- |
| 1 | 4.321 | call | `tests/test_ops.py::test_operator_c_testbench_matches_onnxruntime[Pow]` |

Captured 42 duration entries (showing top 20).
```
