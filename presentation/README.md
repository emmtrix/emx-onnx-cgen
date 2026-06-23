# ONNX Community Day presentation

Working notes for a 20 minute ONNX Community Day talk about
`emx-onnx-cgen`.

## Submitted title

**Lessons learned from building an AOT ONNX-to-C compiler**

## Submitted abstract

The submitted abstract is stored verbatim in
[`source-input.md`](source-input.md#submitted-abstract). The slide deck should
stay aligned with that scope: AOT ONNX-to-C compilation, 100% test coverage as
an architectural driver, boundedness issues, incomplete shape/type inference,
container handling, and numerical accuracy ambiguity.

## Talk goal

Present lessons learned from developing `emx-onnx-cgen`, an ahead-of-time
compiler that translates ONNX models into portable C code for embedded and
resource-constrained systems.

The talk should not be a pure project pitch. It should explain why the project
was started, what assumptions worked, what did not work, and what the ONNX
community can learn from an AOT compiler perspective.

## Audience

- ONNX Community Day attendees
- ONNX contributors, backend/runtime developers, model exporters, test authors
- Technical audience familiar with ONNX concepts

## Main message

ONNX is a strong interchange format, but deterministic ahead-of-time
compilation to static, portable C exposes requirements that ONNX does not
always constrain: bounded shapes, bounded containers, precise type information,
and clear numerical accuracy contracts.

## Repository sources

Relevant project files used for the talk material:

- `README.md`
- `SUPPORT_OPS.md`
- `ONNX_SUPPORT.md`
- `ONNX_ERRORS.md`
- `docs/development.md`
- `docs/output-format.md`
- `AGENTS.md`

## Working files

- `presentation/story.md`: long-form story document; primary working document
  before slide creation
- `presentation/source-input.md`: raw user-provided input and project facts
- `presentation/talk-brief.md`: condensed talk framing
- `presentation/outline.md`: proposed 20 minute structure
- `presentation/lessons-learned.md`: detailed technical lesson notes
