"""Print the ONNX operator specification from the installed onnx package.

Replaces ``docs/Operators.md`` (current version) and ``docs/Changelog.md``
(all versions) of the onnx repository, which the onnx wheel does not ship.

Examples::

    python tools/onnx_op_spec.py Slice
    python tools/onnx_op_spec.py Slice --version 13
    python tools/onnx_op_spec.py Slice --all-versions
    python tools/onnx_op_spec.py LabelEncoder --domain ai.onnx.ml
"""

from __future__ import annotations

import argparse
import sys

import onnx
from onnx import defs, helper

_OPTION_SUFFIX = {
    defs.OpSchema.FormalParameterOption.Single: "",
    defs.OpSchema.FormalParameterOption.Optional: " (optional)",
    defs.OpSchema.FormalParameterOption.Variadic: " (variadic)",
}


def _domain_label(domain: str) -> str:
    return domain or "ai.onnx"


def _format_parameters(title: str, parameters: list) -> list[str]:
    lines = [f"## {title}", ""]
    if not parameters:
        return [*lines, "(none)", ""]
    for parameter in parameters:
        suffix = _OPTION_SUFFIX.get(parameter.option, "")
        lines.append(f"- {parameter.name}{suffix}: {parameter.type_str}")
        if parameter.description:
            lines.append(f"  {parameter.description.strip()}")
    return [*lines, ""]


def _format_attributes(schema: defs.OpSchema) -> list[str]:
    lines = ["## Attributes", ""]
    if not schema.attributes:
        return [*lines, "(none)", ""]
    for name, attribute in sorted(schema.attributes.items()):
        kind = attribute.type.name
        if attribute.required:
            detail = "required"
        elif attribute.default_value.type:
            value = helper.get_attribute_value(attribute.default_value)
            detail = f"default {value!r}"
        else:
            detail = "optional"
        lines.append(f"- {name} ({kind}, {detail})")
        if attribute.description:
            lines.append(f"  {attribute.description.strip()}")
    return [*lines, ""]


def _format_type_constraints(schema: defs.OpSchema) -> list[str]:
    lines = ["## Type constraints", ""]
    if not schema.type_constraints:
        return [*lines, "(none)", ""]
    for constraint in schema.type_constraints:
        allowed = ", ".join(constraint.allowed_type_strs)
        lines.append(f"- {constraint.type_param_str}: {allowed}")
        if constraint.description:
            lines.append(f"  {constraint.description.strip()}")
    return [*lines, ""]


def format_schema(schema: defs.OpSchema) -> str:
    header = (
        f"# {schema.name}-{schema.since_version} "
        f"(domain {_domain_label(schema.domain)})"
    )
    notes = []
    if schema.deprecated:
        notes.append("deprecated")
    if schema.has_function or schema.has_context_dependent_function:
        notes.append("defined as a function")
    lines = [header, ""]
    if notes:
        lines += [f"Notes: {', '.join(notes)}", ""]
    lines += [(schema.doc or "").strip(), ""]
    lines += _format_parameters("Inputs", list(schema.inputs))
    lines += _format_parameters("Outputs", list(schema.outputs))
    lines += _format_attributes(schema)
    lines += _format_type_constraints(schema)
    return "\n".join(lines)


def _schema_versions(op_type: str, domain: str) -> list[defs.OpSchema]:
    return sorted(
        (
            schema
            for schema in defs.get_all_schemas_with_history()
            if schema.name == op_type and schema.domain == domain
        ),
        key=lambda schema: schema.since_version,
        reverse=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("op_type", help="Operator name, e.g. Slice")
    parser.add_argument(
        "--domain",
        default="",
        help="Operator domain, e.g. ai.onnx.ml (default: ai.onnx)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--version",
        type=int,
        help="Print the schema in effect at this opset version",
    )
    selection.add_argument(
        "--all-versions",
        action="store_true",
        help="Print every version of the operator, newest first",
    )
    args = parser.parse_args(argv)
    domain = "" if args.domain == "ai.onnx" else args.domain

    versions = _schema_versions(args.op_type, domain)
    if not versions:
        print(
            f"No schema for {args.op_type} in domain {_domain_label(domain)} "
            f"(onnx {onnx.__version__}).",
            file=sys.stderr,
        )
        return 1
    if args.all_versions:
        selected = versions
    elif args.version is not None:
        selected = [v for v in versions if v.since_version <= args.version][:1]
        if not selected:
            print(
                f"{args.op_type} does not exist at opset {args.version}; "
                f"it was introduced in opset {versions[-1].since_version}.",
                file=sys.stderr,
            )
            return 1
    else:
        selected = versions[:1]
    print(f"onnx {onnx.__version__}\n")
    print("\n".join(format_schema(schema) for schema in selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
