from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_sbom", PROJECT_ROOT / "tools" / "generate_sbom.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sbom = _load_generator()


def _distribution_file(name: str, relative_path: str) -> Path:
    distribution = importlib.metadata.distribution(name)
    return Path(distribution.locate_file(relative_path)).resolve()


def _write_toc(workpath: Path, entries: list[tuple[str, str, str]]) -> None:
    workpath.mkdir(parents=True, exist_ok=True)
    # Same shape as PyInstaller's COLLECT/PYZ TOC files: a literal with the
    # entry list nested inside.
    (workpath / "COLLECT-00.toc").write_text(
        repr((str(workpath / "dist"), [*entries, ("x", "", "OPTION")])),
        encoding="utf-8",
    )


@pytest.fixture
def bundle(tmp_path: Path):
    workpath = tmp_path / "build" / "emx-onnx-cgen"
    workpath.mkdir(parents=True)
    (workpath / "base_library.zip").write_bytes(b"stdlib")
    (workpath / "emx-onnx-cgen.exe").write_bytes(b"bootloader")
    unknown = tmp_path / "elsewhere" / "libmystery.so"
    unknown.parent.mkdir()
    unknown.write_bytes(b"mystery")
    package_dir = PROJECT_ROOT / "src" / "emx_onnx_cgen"
    entries = [
        ("emx_onnx_cgen.cli", str(package_dir / "cli.py"), "PYMODULE"),
        (
            "emx_onnx_cgen/third_party/stb_image.h",
            str(package_dir / "third_party" / "stb_image.h"),
            "DATA",
        ),
        (
            "jinja2",
            str(_distribution_file("jinja2", "jinja2/__init__.py")),
            "PYMODULE",
        ),
        ("os", os.__file__, "PYMODULE"),
        ("base_library.zip", str(workpath / "base_library.zip"), "DATA"),
        ("emx-onnx-cgen.exe", str(workpath / "emx-onnx-cgen.exe"), "EXECUTABLE"),
        ("libmystery.so", str(unknown), "BINARY"),
        ("google", "-", "PYMODULE"),
    ]
    _write_toc(workpath, entries)
    return workpath


def _bundle_bom(workpath: Path):
    return sbom.bundle_bom(
        workpath=workpath,
        version="1.2.3",
        platform="test-platform",
        rpm_resolver_factory=lambda registry: None,
        # Stands in for PyInstaller, which the test environment may not have.
        pyinstaller_origin=_distribution_file("markupsafe", "markupsafe/__init__.py"),
    )


def test_bundle_bom_attributes_every_bundled_file(bundle: Path) -> None:
    bom, unresolved = _bundle_bom(bundle)

    components = {component["bom-ref"]: component for component in bom["components"]}
    names = {component["name"] for component in components.values()}
    assert {"stb_image", "Jinja2", "MarkupSafe", "CPython"} <= names
    assert [item.destination for item in unresolved] == ["libmystery.so"]
    mystery = components["file:libmystery.so"]
    assert mystery["type"] == "file"
    assert {item["alg"] for item in mystery["hashes"]} == {"SHA-256", "SHA-512"}

    project = bom["metadata"]["component"]
    assert project["name"] == "emx-onnx-cgen"
    assert project["version"] == "1.2.3"
    assert project["purl"] == "pkg:pypi/emx-onnx-cgen@1.2.3"
    assert project["licenses"] == [{"expression": "Apache-2.0"}]
    project_files = {
        prop["name"]: prop["value"] for prop in project.get("properties", ())
    }
    assert project_files["emx-onnx-cgen:bundled-files"] == "1"

    stb = next(c for c in components.values() if c["name"] == "stb_image")
    assert stb["version"] == "2.28"
    assert stb["licenses"] == [{"expression": "MIT OR Unlicense"}]

    cpython = next(c for c in components.values() if c["name"] == "CPython")
    assert cpython["version"] == sys.version.split()[0]
    assert cpython["cpe"].startswith("cpe:2.3:a:python:python:")


def test_bundle_bom_dependency_graph_reaches_every_component(bundle: Path) -> None:
    bom, _ = _bundle_bom(bundle)

    dependencies = {entry["ref"]: entry["dependsOn"] for entry in bom["dependencies"]}
    project_ref = bom["metadata"]["component"]["bom-ref"]
    refs = {component["bom-ref"] for component in bom["components"]}
    assert set(dependencies) == refs | {project_ref}

    jinja2 = next(ref for ref in refs if ref.startswith("pkg:pypi/jinja2@"))
    markupsafe = next(ref for ref in refs if ref.startswith("pkg:pypi/markupsafe@"))
    if sys.version_info >= (3, 11):
        # Declared in pyproject.toml, which needs tomllib to be read.
        assert jinja2 in dependencies[project_ref]
    assert markupsafe in dependencies[jinja2]

    reachable, pending = set(), [project_ref]
    while pending:
        ref = pending.pop()
        if ref not in reachable:
            reachable.add(ref)
            pending.extend(dependencies[ref])
    assert reachable == set(dependencies)


def test_bundle_bom_is_deterministic(bundle: Path, monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first, _ = _bundle_bom(bundle)
    second, _ = _bundle_bom(bundle)

    assert json.dumps(first) == json.dumps(second)
    assert first["metadata"]["timestamp"] == "2023-11-14T22:13:20Z"


def test_bundle_bom_rejects_a_foreign_interpreter(tmp_path: Path) -> None:
    workpath = tmp_path / "build"
    _write_toc(workpath, [("os", str(tmp_path / "other" / "os.py"), "PYMODULE")])

    with pytest.raises(sbom.SbomError, match="interpreter that ran PyInstaller"):
        _bundle_bom(workpath)


def test_read_bundled_files_requires_toc_files(tmp_path: Path) -> None:
    with pytest.raises(sbom.SbomError, match="No PyInstaller"):
        sbom.read_bundled_files(tmp_path)


def test_rpm_resolver_reports_the_owning_package() -> None:
    registry = sbom.Registry()
    query_output = (
        "openssl-libs\t1\t1.1.1k\t12.el8_9\tx86_64\tOpenSSL and ASL 2.0"
        "\t/usr/lib64/libssl.so.1.1.1k\n"
        "zlib\t0\t1.2.11\t25.el8\tx86_64\tzlib and Boost\t/usr/lib64/libz.so.1\n"
    )
    resolver = sbom.RpmResolver(
        registry, query_output, {"ID": "almalinux", "VERSION_ID": "8.10"}
    )

    component = resolver.resolve(
        sbom.BundledFile("libssl.so.1.1", Path("/usr/lib64/libssl.so.1.1.1k"), "BINARY")
    )

    assert component.purl == (
        "pkg:rpm/almalinux/openssl-libs@1.1.1k-12.el8_9"
        "?arch=x86_64&distro=almalinux-8.10&epoch=1"
    )
    assert component.version == "1.1.1k-12.el8_9"
    assert (
        resolver.resolve(sbom.BundledFile("other.so", Path("/opt/other.so"), "BINARY"))
        is None
    )


def test_distribution_licenses_prefer_declared_expression() -> None:
    from email.message import Message

    def metadata(**fields):
        message = Message()
        for key, values in fields.items():
            for value in values if isinstance(values, list) else [values]:
                message[key.replace("_", "-")] = value
        return message

    assert sbom._distribution_licenses(
        metadata(License_Expression="MIT", License="whatever")
    ) == ({"expression": "MIT"},)
    assert sbom._distribution_licenses(
        metadata(
            License="Copyright (c) 2005\nAll rights reserved.",
            Classifier=["License :: OSI Approved :: BSD License"],
        )
    ) == ({"license": {"name": "BSD License"}},)


@pytest.mark.skipif(
    importlib.util.find_spec("emx_onnx_cgen") is None
    or not any(
        dist.metadata["Name"] == "emx-onnx-cgen"
        for dist in importlib.metadata.distributions()
    ),
    reason="emx-onnx-cgen is not installed",
)
def test_wheel_bom_lists_the_runtime_closure() -> None:
    bom = sbom.wheel_bom()

    names = {component["name"].lower() for component in bom["components"]}
    assert {"stb_image", "numpy", "onnx", "jinja2", "markupsafe"} <= names
    assert "onnxruntime" not in names  # only an optional (verify) dependency
