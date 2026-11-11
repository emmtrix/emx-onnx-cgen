from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "combine_release_bundles", PROJECT_ROOT / "tools" / "combine_release_bundles.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


combine = _load_tool()


def _linux_bundle(path: Path) -> Path:
    with tarfile.open(path, "w:gz") as tar:

        def add(name: str, *, data: bytes = b"", mode: int = 0o644, link: str = ""):
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, link
            elif name.endswith("/"):
                info.type = tarfile.DIRTYPE
            info.mode, info.size = mode, len(data)
            tar.addfile(info, io.BytesIO(data) if data else None)

        add("emx-onnx-cgen/", mode=0o755)
        add("emx-onnx-cgen/emx-onnx-cgen", data=b"\x7fELF", mode=0o755)
        add("emx-onnx-cgen/_internal/numpy.libs/libopenblas.so", data=b"blas")
        add("emx-onnx-cgen/_internal/libopenblas.so", link="numpy.libs/libopenblas.so")
    return path


def _windows_bundle(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("emx-onnx-cgen.exe", b"MZ")
        bundle.writestr("_internal/base_library.zip", b"stdlib")
    return path


def _combined(tmp_path: Path, name: str) -> Path:
    output = tmp_path / name
    assert (
        combine.main(
            [
                "--linux",
                str(_linux_bundle(tmp_path / "linux.tar.gz")),
                "--windows",
                str(_windows_bundle(tmp_path / "windows.zip")),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    return output


def test_combined_archive_keeps_each_platform_in_its_directory(tmp_path: Path) -> None:
    with tarfile.open(_combined(tmp_path, "all.tar.gz")) as tar:
        members = {info.name: info for info in tar.getmembers()}

    assert list(members) == sorted(members, key=lambda name: name.split("/"))
    assert {
        "emx-onnx-cgen/linux-amd64/emx-onnx-cgen",
        "emx-onnx-cgen/linux-amd64/_internal/numpy.libs/libopenblas.so",
        "emx-onnx-cgen/windows-amd64/emx-onnx-cgen.exe",
        "emx-onnx-cgen/windows-amd64/_internal",
        "emx-onnx-cgen/windows-amd64/_internal/base_library.zip",
    } <= set(members)
    link = members["emx-onnx-cgen/linux-amd64/_internal/libopenblas.so"]
    assert link.issym() and link.linkname == "numpy.libs/libopenblas.so"
    assert members["emx-onnx-cgen/linux-amd64/emx-onnx-cgen"].mode == 0o755
    assert members["emx-onnx-cgen/windows-amd64/emx-onnx-cgen.exe"].mode == 0o755
    assert members["emx-onnx-cgen/windows-amd64/_internal"].isdir()
    assert {info.uid for info in members.values()} == {0}


def test_combined_archive_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first = _combined(tmp_path, "first.tar.gz").read_bytes()
    second = _combined(tmp_path, "second.tar.gz").read_bytes()

    assert first == second
    with tarfile.open(tmp_path / "first.tar.gz") as tar:
        assert {info.mtime for info in tar.getmembers()} == {1700000000}


def test_combined_archive_rejects_paths_outside_the_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape.txt", b"x")

    with pytest.raises(combine.CombineError, match="unsafe member path"):
        combine._zip_members(bundle)


def test_combined_archive_carries_the_sbom(tmp_path: Path) -> None:
    sbom = tmp_path / "inner.cdx.json"
    sbom.write_text('{"bomFormat": "CycloneDX"}', encoding="utf-8")
    output = tmp_path / "all.tar.gz"
    combine.main(
        [
            "--linux",
            str(_linux_bundle(tmp_path / "linux.tar.gz")),
            "--windows",
            str(_windows_bundle(tmp_path / "windows.zip")),
            "--sbom",
            str(sbom),
            "--output",
            str(output),
        ]
    )

    with tarfile.open(output) as tar:
        member = tar.extractfile("emx-onnx-cgen/emx-onnx-cgen.cdx.json")
        assert member.read() == sbom.read_bytes()
