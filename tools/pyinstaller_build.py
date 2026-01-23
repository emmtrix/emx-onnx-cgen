#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
        str(entrypoint_path),
    ]

    for module_name in lowering._LOWERING_MODULES:
        command.extend(["--hidden-import", f"emx_onnx_cgen.lowering.{module_name}"])

    command.extend(sys.argv[1:])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(command, check=True, cwd=repo_root, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
