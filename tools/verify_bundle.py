#!/usr/bin/env python3
"""Check an unpacked release bundle against its SBOM.

The bundle SBOMs list every file of the bundle as a ``file`` component with
its SHA-256 (symlinks with their target). This script checks a directory
against them. It reports files that are missing, changed or not listed, and
exits with 1 if there are any.

Examples::

    # One platform bundle, unpacked to emx-onnx-cgen/
    python verify_bundle.py --sbom emx-onnx-cgen-linux-amd64.cdx.json --root emx-onnx-cgen
    # The combined archive, unpacked to emx-onnx-cgen/ (its SBOM is inside)
    python verify_bundle.py --sbom emx-onnx-cgen/emx-onnx-cgen.cdx.json --root emx-onnx-cgen
    # Only the Linux part of the combined archive, copied elsewhere
    python verify_bundle.py --sbom emx-onnx-cgen.cdx.json --root product/emx-onnx-cgen \\
        --prefix linux-amd64

Only the standard library is used, so that it runs in any build environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterator

BUNDLE_FILE_PREFIX = "bundle-file:"


def bundle_files(bom: dict) -> Iterator[dict]:
    """Every bundle file component, wherever it is nested."""
    pending = [bom["metadata"].get("component", {}), *bom.get("components", ())]
    while pending:
        component = pending.pop()
        if component.get("type") == "file" and component.get("bom-ref", "").startswith(
            BUNDLE_FILE_PREFIX
        ):
            yield component
        pending.extend(component.get("components", ()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symlink_target(component: dict) -> str | None:
    return next(
        (
            prop["value"]
            for prop in component.get("properties", ())
            if prop["name"].endswith(":symlink-target")
        ),
        None,
    )


def verify(
    bom: dict, root: Path, prefix: str = "", ignore: set[str] = frozenset()
) -> list[str]:
    """Problems found in ``root``; empty if it matches the SBOM exactly."""
    expected: dict[str, dict] = {}
    for component in bundle_files(bom):
        name = component["name"]
        if prefix:
            if not name.startswith(prefix + "/"):
                continue
            name = name[len(prefix) + 1 :]
        expected[name] = component
    if not expected:
        return [f"The SBOM lists no files{f' below {prefix}/' if prefix else ''}."]

    problems = []
    for name, component in sorted(expected.items()):
        path = root / name
        target = _symlink_target(component)
        if target is not None:
            if not path.is_symlink():
                problems.append(f"{name}: expected a symlink to {target}")
            elif os.readlink(path).replace(os.sep, "/") != target:
                problems.append(
                    f"{name}: links to {os.readlink(path)}, expected {target}"
                )
            continue
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        hashes = {item["alg"]: item["content"] for item in component.get("hashes", ())}
        if _sha256(path) != hashes.get("SHA-256"):
            problems.append(f"{name}: SHA-256 differs from the SBOM")
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        name = path.relative_to(root).as_posix()
        if name not in expected and name not in ignore:
            problems.append(f"{name}: not listed in the SBOM")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--sbom", type=Path, required=True, help="CycloneDX SBOM")
    parser.add_argument("--root", type=Path, required=True, help="Unpacked bundle")
    parser.add_argument(
        "--prefix",
        default="",
        help="Check only the files below this directory of the SBOM, e.g. linux-amd64",
    )
    args = parser.parse_args(argv)
    bom = json.loads(args.sbom.read_text(encoding="utf-8"))
    ignore = set()
    sbom_path = args.sbom.resolve()
    if sbom_path.is_relative_to(args.root.resolve()):
        # The SBOM that the combined archive carries does not list itself.
        ignore.add(sbom_path.relative_to(args.root.resolve()).as_posix())
    problems = verify(bom, args.root, args.prefix, ignore)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s) in {args.root}", file=sys.stderr)
        return 1
    print(f"{args.root}: all files match the SBOM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
