## Summary
- fix `Loop` sequence-map lowering to resolve mapped body names to external graph values (prevents KeyError in `test_sequence_map_extract_shapes`)
- improve verify flow for adjusted test-data inputs: when sequence/tensor test inputs are normalized or reshaped for testbench format, compare against runtime reference outputs instead of stale pb outputs
- support sequence inputs in runtime feed and concretize sequence element shapes from provided shape-inference inputs
- update expected errors and generated support docs for `test_sequence_map_extract_shapes` and `_expanded` to passing status

## Validation
- `pytest -n auto -q tests/test_official_onnx_files.py -k "sequence_map_extract_shapes" --maxfail=10` (2 passed, 6.19s)
- `UPDATE_REFS=1 pytest -n auto -q tests/test_official_onnx_files_docs.py --maxfail=10` (1 passed, 9.20s)
- `pytest -n auto -q tests/test_official_onnx_files_docs.py --maxfail=10` (1 passed, 5.38s)

## Notes
- no submodule contents were modified
