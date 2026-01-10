# emx-onnx2c

## CLI

Compile an ONNX model into a C source file:

```bash
python -m onnx2c compile path/to/model.onnx build/model.c
```

Emit a JSON-producing testbench for end-to-end validation:

```bash
python -m onnx2c compile path/to/model.onnx build/model.c --emit-testbench
```
