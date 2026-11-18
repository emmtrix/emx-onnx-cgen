# ONNX backend test data

This directory holds the official ONNX backend test models that
`tests/test_official_onnx_files.py` verifies. Their expectation files are
`tests/expected_errors/onnx-backend-test-data__*.json`. Only this README and
`generate.py` are checked in. Everything else is generated, gitignored, and
must not be edited by hand.

| Directory | Source |
| --- | --- |
| `node/` | Generated from `onnx.backend.test.case.node` (onnx 1.23 no longer ships these files, see onnx/onnx#7959) |
| `light/`, `pytorch-converted/`, `pytorch-operator/`, `real/`, `simple/` | Copied from `onnx/backend/test/data` of the installed onnx wheel |
| `.onnx-version` | Stamp of the onnx version and generator revision the data was built with |

## Building

pytest builds the data automatically on startup (`tests/conftest.py`). To
build it without pytest, e.g. before reproducing a single `verify` command
line from an expectation file, run:

```bash
python onnx-backend-test-data/generate.py
```

- The installed onnx version must match the `onnx==` pin in
  `requirements-ci.txt`. Otherwise nothing is generated, and pytest warns and
  skips the official tests.
- The data is rebuilt whenever `.onnx-version` does not match the pin and the
  generator revision, i.e. after an onnx upgrade or a generator change. It
  takes about 10 s.
- The `node/` reference outputs are computed by numpy. `generate.py` computes
  them in a child process restricted to numpy's baseline SIMD kernels and a
  fixed single-threaded OpenBLAS core type, so that every x86 host writes the
  same data. Otherwise the recorded ULP values differ between CI hosts.
- To force a rebuild, delete `.onnx-version`.
