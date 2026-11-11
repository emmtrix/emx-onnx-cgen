#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# The build environment installs all of requirements-ci.txt, and PyInstaller
# follows imports that only optional code paths reach. These modules are not
# needed at runtime, so keep them (and what only they pull in) out of the bundle:
# - pytest/_pytest/py: sympy.testing and mpmath.tests import them inside
#   try/except ImportError. This also drops pluggy, iniconfig and colorama.
# - pkg_resources (setuptools): only _pytest.monkeypatch imports it.
# - yaml: only numpy.show_config() uses it, and falls back without it.
# Pillow stays: onnx.reference needs it for ImageDecoder in
# `verify --runtime onnx-reference`.
EXCLUDED_MODULES = ("pytest", "_pytest", "py", "pkg_resources", "yaml")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    entrypoint_path = repo_root / "tools" / "pyinstaller_entrypoint.py"

    sys.path.insert(0, str(repo_root / "src"))
    from emx_onnx_cgen import lowering  # noqa: E402

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onedir",
        "--name",
        "emx-onnx-cgen",
        "--paths",
        "src",
        "--collect-data",
        "emx_onnx_cgen",
        str(entrypoint_path),
    ]

    for module_name in lowering._LOWERING_MODULES:
        command.extend(["--hidden-import", f"emx_onnx_cgen.lowering.{module_name}"])
    for module_name in EXCLUDED_MODULES:
        command.extend(["--exclude-module", module_name])

    command.extend(sys.argv[1:])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(command, check=True, cwd=repo_root, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
