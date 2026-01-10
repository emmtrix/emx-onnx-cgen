# emx-onnx2c

## CLI

Compile an ONNX model into a C source file:

```bash
python -m onnx2c compile path/to/model.onnx build/model.c
```

Verify an ONNX model end-to-end against ONNX Runtime:

```bash
python -m onnx2c verify path/to/model.onnx
```

### CLI Parameters

The CLI currently exposes a single `compile` command with the following parameters:

- `model` (positional): Path to the ONNX model file to compile.
- `output` (positional): Output path for the generated C source file.
- `--template-dir`: Directory containing the C templates. Defaults to `templates`.
- `--model-name`: Overrides the generated model name. Defaults to the output file stem.
- `--emit-testbench`: Emits a JSON-producing `main()` testbench for validation.
