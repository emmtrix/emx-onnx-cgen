"""Generate CycloneDX SBOMs for the emx-onnx-cgen release assets.

Two kinds of release assets are described:

``bundle``
    A PyInstaller onedir bundle (``emx-onnx-cgen-<platform>.tar.gz`` / ``.zip``).
    Its content is read from PyInstaller's own build records (the ``*.toc``
    files in the work path), and every bundled file is attributed to the
    component it comes from: emx-onnx-cgen itself, a vendored source (stb), an
    installed Python distribution (via its ``RECORD``), the Microsoft C/C++
    runtime, a system package (via ``rpm`` on Linux), or the CPython
    installation. Run it with the interpreter that ran PyInstaller, so that the
    installed distributions and the CPython version are the bundled ones.

``wheel``
    The emx-onnx-cgen wheel. Run it with the interpreter of a fresh virtual
    environment into which the wheel was installed; the runtime dependencies are
    reported in the versions pip resolved there.

``combine``
    The archive with all platform bundles (``tools/combine_release_bundles.py``),
    merged from the platform bundle SBOMs. Each component appears once, with the
    platforms it was found in.

Examples::

    python tools/generate_sbom.py bundle --workpath build/emx-onnx-cgen \\
        --release-tag v1.4.2 --platform windows-amd64 \\
        --artifact emx-onnx-cgen-windows-amd64.zip \\
        --output emx-onnx-cgen-windows-amd64.cdx.json
    python tools/generate_sbom.py wheel \\
        --artifact dist/emx_onnx_cgen-1.4.2-py3-none-any.whl \\
        --output emx-onnx-cgen-1.4.2-wheel.cdx.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator, Protocol

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: only the dependency graph gets coarser.
    tomllib = None

# A sibling script in tools/, which is sys.path[0] when this script runs.
import fetch_release_python  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "emx-onnx-cgen"
PROJECT_PACKAGE = "emx_onnx_cgen"
PROJECT_LICENSE = "Apache-2.0"
PROJECT_REPOSITORY = "https://github.com/emmtrix/emx-onnx-cgen"
PROJECT_WEBSITE = "https://www.emmtrix.com/wiki/emmtrix_ONNX-to-C_Code_Generator"
# Everything PyInstaller collects from here is emx-onnx-cgen itself.
PROJECT_SOURCE_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "tools" / "pyinstaller_entrypoint.py",
)
SUPPLIER = {"name": "emmtrix Technologies GmbH", "url": ["https://www.emmtrix.com"]}
SPEC_VERSION = "1.6"
PROPERTY_PREFIX = "emx-onnx-cgen"

# PyInstaller TOC typecodes of entries that end up as content in the bundle.
# PYZ/PKG containers and OPTION entries only describe how content is packed.
_BUNDLED_TYPECODES = frozenset(
    {"PYMODULE", "PYSOURCE", "EXTENSION", "BINARY", "DATA", "DEPENDENCY", "EXECUTABLE"}
)


class SbomError(RuntimeError):
    """The SBOM cannot be generated from the given inputs."""


@dataclass
class Component:
    bom_ref: str
    type: str
    name: str
    version: str | None = None
    purl: str | None = None
    cpe: str | None = None
    supplier: dict | None = None
    licenses: tuple[dict, ...] = ()
    description: str | None = None
    website: str | None = None
    hashes: tuple[dict, ...] = ()
    external_references: tuple[dict, ...] = ()
    properties: tuple[tuple[str, str], ...] = ()
    depends_on: set[str] = field(default_factory=set)
    # The component's files in the bundle, as nested "file" components.
    files: tuple[dict, ...] = ()

    def to_json(self) -> dict:
        data: dict = {"type": self.type, "bom-ref": self.bom_ref}
        if self.supplier:
            data["supplier"] = self.supplier
        data["name"] = self.name
        if self.version:
            data["version"] = self.version
        if self.description:
            data["description"] = self.description
        if self.hashes:
            data["hashes"] = list(self.hashes)
        if self.licenses:
            data["licenses"] = list(self.licenses)
        if self.cpe:
            data["cpe"] = self.cpe
        if self.purl:
            data["purl"] = self.purl
        references = list(self.external_references)
        if self.website:
            references.insert(0, {"type": "website", "url": self.website})
        if references:
            data["externalReferences"] = references
        if self.properties:
            data["properties"] = [
                {"name": name, "value": value} for name, value in self.properties
            ]
        if self.files:
            data["components"] = list(self.files)
        return data


@dataclass(frozen=True)
class BundledFile:
    """One file PyInstaller put into the bundle, with the file it was made from."""

    destination: str
    source: Path
    typecode: str


# --- helpers --------------------------------------------------------------


def _file_hashes(path: Path) -> tuple[dict, ...]:
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha256.update(chunk)
            sha512.update(chunk)
    return (
        {"alg": "SHA-256", "content": sha256.hexdigest()},
        {"alg": "SHA-512", "content": sha512.hexdigest()},
    )


def _path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _is_relative_to(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _distribution_licenses(metadata) -> tuple[dict, ...]:
    """Licenses as declared in the core metadata, without guessing SPDX ids."""
    expression = metadata.get("License-Expression")
    if expression:
        return ({"expression": expression.strip()},)
    license_field = (metadata.get("License") or "").strip()
    # Some projects put the whole license text into this field; only a short
    # one-line value names the license.
    if (
        license_field
        and "\n" not in license_field
        and len(license_field) <= 120
        and license_field.upper() != "UNKNOWN"
    ):
        return ({"license": {"name": license_field}},)
    names = [
        classifier.rsplit(" :: ", 1)[-1]
        for classifier in metadata.get_all("Classifier") or ()
        if classifier.startswith("License :: ")
    ]
    return tuple({"license": {"name": name}} for name in dict.fromkeys(names))


def _distribution_website(metadata) -> str | None:
    homepage = metadata.get("Home-page")
    if homepage and homepage.upper() != "UNKNOWN":
        return homepage.strip()
    urls = {}
    for entry in metadata.get_all("Project-URL") or ():
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for label in ("homepage", "home", "source", "repository", "source code"):
        if label in urls:
            return urls[label]
    return next(iter(urls.values()), None)


def _distribution_ref(distribution: importlib.metadata.Distribution) -> str:
    name = canonicalize_name(distribution.metadata["Name"])
    return f"pkg:pypi/{name}@{distribution.version}"


def _distribution_component(distribution: importlib.metadata.Distribution) -> Component:
    metadata = distribution.metadata
    ref = _distribution_ref(distribution)
    return Component(
        bom_ref=ref,
        type="library",
        name=metadata["Name"],
        version=distribution.version,
        purl=ref,
        licenses=_distribution_licenses(metadata),
        description=(metadata.get("Summary") or "").strip() or None,
        website=_distribution_website(metadata),
    )


def _requirement_names(requirements: Iterable[str]) -> set[str]:
    """Names of the requirements that apply without extras on this platform."""
    names = set()
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            names.add(canonicalize_name(requirement.name))
    return names


def _pe_file_version(path: Path) -> str | None:
    """File version of a Windows PE file (DLL/EXE), if readable."""
    if sys.platform != "win32" or path.suffix.lower() not in (".dll", ".pyd", ".exe"):
        return None
    import ctypes
    from ctypes import wintypes

    version_dll = ctypes.WinDLL("version")
    size = version_dll.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buffer = ctypes.create_string_buffer(size)
    if not version_dll.GetFileVersionInfoW(str(path), 0, size, buffer):
        return None
    info = ctypes.c_void_p()
    length = wintypes.UINT()
    if not version_dll.VerQueryValueW(
        buffer, "\\", ctypes.byref(info), ctypes.byref(length)
    ):
        return None
    # VS_FIXEDFILEINFO starts with signature, struct version, file version MS/LS.
    fields = ctypes.cast(info, ctypes.POINTER(wintypes.DWORD * 4)).contents
    major_minor, build_revision = fields[2], fields[3]
    return ".".join(
        str(part)
        for part in (
            major_minor >> 16,
            major_minor & 0xFFFF,
            build_revision >> 16,
            build_revision & 0xFFFF,
        )
    )


# --- reading the bundle ---------------------------------------------------


def _toc_entries(data: object) -> Iterator[tuple[str, str, str]]:
    if isinstance(data, (list, tuple)):
        if (
            len(data) == 3
            and all(isinstance(item, str) for item in data)
            and data[2].isupper()
        ):
            yield (data[0], data[1], data[2])
            return
        for item in data:
            yield from _toc_entries(item)


def read_bundled_files(workpath: Path) -> list[BundledFile]:
    """Every file PyInstaller bundled, read from the TOC files of ``workpath``.

    The TOC files are Python literals of ``(destination, source, typecode)``
    entries; they are searched structurally so that the surrounding layout,
    which differs between PyInstaller versions, does not matter.
    """
    toc_paths = sorted(workpath.glob("*.toc"))
    if not toc_paths:
        raise SbomError(f"No PyInstaller *.toc files in {workpath}.")
    files: dict[tuple[str, str], BundledFile] = {}
    for toc_path in toc_paths:
        data = ast.literal_eval(toc_path.read_text(encoding="utf-8"))
        for destination, source, typecode in _toc_entries(data):
            # Namespace packages have no file ("-").
            if typecode not in _BUNDLED_TYPECODES or source in ("", "-"):
                continue
            key = (destination, _path_key(source))
            files.setdefault(key, BundledFile(destination, Path(source), typecode))
    if not files:
        raise SbomError(f"The TOC files in {workpath} list no bundled files.")
    return sorted(files.values(), key=lambda item: (item.destination, str(item.source)))


def _check_interpreter(bundled_files: list[BundledFile]) -> None:
    """The bundle must have been built by the interpreter running this script."""
    for bundled in bundled_files:
        if bundled.typecode == "PYMODULE" and bundled.destination == "os":
            if _path_key(bundled.source) != _path_key(os.__file__):
                raise SbomError(
                    f"The bundle was built with the standard library in "
                    f"{bundled.source.parent}, but this script runs on "
                    f"{Path(os.__file__).parent}. Run it with the interpreter that "
                    "ran PyInstaller."
                )
            return


# --- attributing files to components --------------------------------------


class Registry:
    """Components by bom-ref; the first registration of a bom-ref wins."""

    def __init__(self) -> None:
        self.components: dict[str, Component] = {}

    def add(self, component: Component) -> Component:
        return self.components.setdefault(component.bom_ref, component)


class Resolver(Protocol):
    def resolve(self, bundled: BundledFile) -> Component | None: ...


@dataclass(frozen=True)
class VendoredSource:
    """Third-party source code copied into the emx_onnx_cgen package."""

    relative_path: str
    name: str
    version_pattern: str
    license_expression: str
    website: str

    def component(self, package_dir: Path) -> Component:
        path = package_dir / self.relative_path
        if not path.is_file():
            raise SbomError(f"Vendored source {self.name} not found at {path}.")
        match = re.search(self.version_pattern, path.read_text(encoding="utf-8"))
        if match is None:
            raise SbomError(f"Cannot read the {self.name} version from {path}.")
        version = match.group(1)
        return Component(
            bom_ref=f"pkg:generic/{self.name}@{version}",
            type="library",
            name=self.name,
            version=version,
            purl=f"pkg:generic/{self.name}@{version}",
            licenses=({"expression": self.license_expression},),
            website=self.website,
            hashes=_file_hashes(path),
            properties=(
                (
                    f"{PROPERTY_PREFIX}:vendored-path",
                    f"{PROJECT_PACKAGE}/{self.relative_path}",
                ),
            ),
        )


VENDORED_SOURCES = (
    VendoredSource(
        relative_path="third_party/stb_image.h",
        name="stb_image",
        version_pattern=r"stb_image - v(\d+(?:\.\d+)+)",
        license_expression="MIT OR Unlicense",
        website="https://github.com/nothings/stb",
    ),
)


class ProjectResolver:
    """emx-onnx-cgen's own files and the sources vendored into it."""

    def __init__(
        self,
        registry: Registry,
        project: Component,
        package_dir: Path,
        source_roots: Iterable[Path],
    ) -> None:
        self._registry = registry
        self._project = project
        self._source_roots = tuple(_path_key(root) for root in source_roots)
        self._vendored = {
            _path_key(package_dir / source.relative_path): source.component(package_dir)
            for source in VENDORED_SOURCES
        }

    def resolve(self, bundled: BundledFile) -> Component | None:
        key = _path_key(bundled.source)
        if key in self._vendored:
            vendored = self._registry.add(self._vendored[key])
            self._project.depends_on.add(vendored.bom_ref)
            return vendored
        if any(_is_relative_to(key, root) for root in self._source_roots):
            return self._project
        return None


class DistributionResolver:
    """Files installed by a Python distribution, according to its RECORD."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self._owners: dict[str, importlib.metadata.Distribution] = {}
        self.bundled: dict[str, importlib.metadata.Distribution] = {}
        for distribution in importlib.metadata.distributions():
            for file in distribution.files or ():
                self._owners.setdefault(
                    _path_key(distribution.locate_file(file)), distribution
                )

    def resolve(self, bundled: BundledFile) -> Component | None:
        distribution = self._owners.get(_path_key(bundled.source))
        if distribution is None:
            return None
        self.bundled[canonicalize_name(distribution.metadata["Name"])] = distribution
        return self._registry.add(_distribution_component(distribution))


class BuildOutputResolver:
    """Files PyInstaller generated in the work path from other sources.

    The bootloader and PyInstaller's loader modules come from PyInstaller;
    ``base_library.zip`` and the other precompiled modules come from the
    standard library. They are attributed to the owner of their origin.
    """

    def __init__(
        self,
        workpath: Path,
        pyinstaller_origin: Path,
        resolve_origin: Callable[[BundledFile], Component],
    ) -> None:
        self._workpath = _path_key(workpath)
        self._pyinstaller_origin = pyinstaller_origin
        self._stdlib_origin = Path(os.__file__).resolve()
        self._resolve_origin = resolve_origin

    def resolve(self, bundled: BundledFile) -> Component | None:
        if not _is_relative_to(_path_key(bundled.source), self._workpath):
            return None
        from_pyinstaller = bundled.typecode == "EXECUTABLE" or (
            bundled.source.name.startswith("pyimod")
        )
        origin = self._pyinstaller_origin if from_pyinstaller else self._stdlib_origin
        return self._resolve_origin(
            BundledFile(bundled.destination, origin, "PYMODULE")
        )


@dataclass(frozen=True)
class _RuntimeGroup:
    name: str
    purl_name: str
    file_pattern: re.Pattern[str]


class MicrosoftRuntimeResolver:
    """The Microsoft C/C++ runtime DLLs PyInstaller collects on Windows."""

    _GROUPS = (
        _RuntimeGroup(
            "Microsoft Universal C Runtime",
            "ucrt",
            re.compile(r"^(api-ms-win-.+|ucrtbase)\.dll$", re.IGNORECASE),
        ),
        _RuntimeGroup(
            "Microsoft Visual C++ Redistributable",
            "vcredist",
            re.compile(
                r"^(vcruntime140(_\d+)?|msvcp140(_\w+)?|concrt140|vcomp140)\.dll$",
                re.IGNORECASE,
            ),
        ),
    )
    _SUPPLIER = {"name": "Microsoft Corporation", "url": ["https://www.microsoft.com"]}

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def resolve(self, bundled: BundledFile) -> Component | None:
        for group in self._GROUPS:
            if not group.file_pattern.match(bundled.source.name):
                continue
            version = _pe_file_version(bundled.source)
            purl = f"pkg:generic/microsoft/{group.purl_name}"
            if version:
                purl += f"@{version}"
            return self._registry.add(
                Component(
                    bom_ref=purl,
                    type="library",
                    name=group.name,
                    version=version,
                    purl=purl,
                    supplier=self._SUPPLIER,
                    licenses=(
                        {"license": {"name": "Microsoft Software License Terms"}},
                    ),
                )
            )
        return None


class RpmResolver:
    """Files owned by an rpm package (the manylinux build container)."""

    QUERY_FORMAT = (
        "[%{=NAME}\\t%{=EPOCHNUM}\\t%{=VERSION}\\t%{=RELEASE}\\t%{=ARCH}"
        "\\t%{=LICENSE}\\t%{FILENAMES}\\n]"
    )

    def __init__(
        self, registry: Registry, rpm_query_output: str, os_release: dict[str, str]
    ) -> None:
        self._registry = registry
        self._distro_id = os_release.get("ID", "linux")
        distro_version = os_release.get("VERSION_ID")
        self._distro = f"{self._distro_id}-{distro_version}" if distro_version else None
        self._owners: dict[str, tuple[str, ...]] = {}
        for line in rpm_query_output.splitlines():
            fields = tuple(line.split("\t"))
            if len(fields) == 7 and fields[6]:
                self._owners.setdefault(os.path.normcase(fields[6]), fields[:6])

    @classmethod
    def from_system(cls, registry: Registry) -> RpmResolver | None:
        rpm = shutil.which("rpm")
        if rpm is None:
            return None
        output = subprocess.run(
            [rpm, "-qa", "--queryformat", cls.QUERY_FORMAT],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        os_release: dict[str, str] = {}
        os_release_path = Path("/etc/os-release")
        if os_release_path.is_file():
            for line in os_release_path.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                os_release[key.strip()] = value.strip().strip('"')
        return cls(registry, output, os_release)

    def resolve(self, bundled: BundledFile) -> Component | None:
        # rpm records the path as installed; the TOC may name it via a symlink.
        package = self._owners.get(os.path.normcase(str(bundled.source)))
        if package is None:
            package = self._owners.get(_path_key(bundled.source))
        if package is None:
            return None
        name, epoch, version, release, arch, license_name = package
        qualifiers = [f"arch={arch}"]
        if self._distro:
            qualifiers.append(f"distro={self._distro}")
        if epoch not in ("", "0"):
            qualifiers.append(f"epoch={epoch}")
        full_version = f"{version}-{release}"
        purl = f"pkg:rpm/{self._distro_id}/{name}@{full_version}?{'&'.join(qualifiers)}"
        return self._registry.add(
            Component(
                bom_ref=purl,
                type="library",
                name=name,
                version=full_version,
                purl=purl,
                licenses=({"license": {"name": license_name}},) if license_name else (),
            )
        )


@dataclass(frozen=True)
class PythonBuild:
    """The pinned CPython build of the release bundles (tools/release_python.json).

    The release workflows build both bundles with this python-build-standalone
    release. It records the archive and the versions of the libraries compiled
    into it that CPython cannot report at runtime.
    """

    distribution: str
    release: str
    python_version: str
    target: str
    archive_url: str
    archive_sha256: str
    libraries: dict[str, str]

    @classmethod
    def load(cls, path: Path, platform: str) -> PythonBuild:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        try:
            url = fetch_release_python.archive_url(config, platform)
        except fetch_release_python.FetchError as error:
            raise SbomError(f"{path}: {error}") from None
        entry = config["platforms"][platform]
        return cls(
            distribution=config["distribution"],
            release=config["release"],
            python_version=config["python_version"],
            target=entry["target"],
            archive_url=url,
            archive_sha256=entry["sha256"],
            libraries=dict(entry.get("libraries", {})),
        )

    def check_interpreter(self) -> None:
        running = sys.version.split()[0]
        if running != self.python_version:
            raise SbomError(
                f"The bundle was built with CPython {running}, but the Python build "
                f"pins {self.python_version}. Build with tools/fetch_release_python.py."
            )


def _openssl_version() -> str | None:
    import ssl

    return ssl.OPENSSL_VERSION.split()[1]


def _sqlite_version() -> str | None:
    import sqlite3

    return sqlite3.sqlite_version


def _expat_version() -> str | None:
    import pyexpat

    return pyexpat.EXPAT_VERSION.removeprefix("expat_")


def _zlib_version() -> str | None:
    import zlib

    # A zlib-ng build reports a zlib-compatible version such as "1.3.1.zlib-ng".
    return None if hasattr(zlib, "ZLIBNG_VERSION") else zlib.ZLIB_RUNTIME_VERSION


def _zlib_ng_version() -> str | None:
    import zlib

    return getattr(zlib, "ZLIBNG_VERSION", None)


def _mpdecimal_version() -> str | None:
    import _decimal

    return _decimal.__libmpdec_version__


def _zstd_version() -> str | None:
    import _zstd

    return _zstd.zstd_version


@dataclass(frozen=True)
class _EmbeddedLibrary:
    """A library compiled into CPython or into one of its extension modules.

    The version comes from the running interpreter when CPython exposes it,
    otherwise from the libraries table of the pinned Python build.
    """

    modules: tuple[str, ...]
    name: str
    cpe_product: str | None
    license_expression: Callable[[str], str]
    runtime_version: Callable[[], str | None] | None = None

    @property
    def key(self) -> str:
        return self.name.lower()


_EMBEDDED_LIBRARIES = (
    _EmbeddedLibrary(
        ("_ssl", "_hashlib"),
        "OpenSSL",
        "openssl:openssl",
        lambda version: "Apache-2.0" if int(version.split(".")[0]) >= 3 else "OpenSSL",
        _openssl_version,
    ),
    _EmbeddedLibrary(
        ("_sqlite3",), "SQLite", "sqlite:sqlite", lambda _: "blessing", _sqlite_version
    ),
    _EmbeddedLibrary(
        ("pyexpat",),
        "Expat",
        "libexpat_project:libexpat",
        lambda _: "MIT",
        _expat_version,
    ),
    _EmbeddedLibrary(("zlib",), "zlib", "zlib:zlib", lambda _: "Zlib", _zlib_version),
    _EmbeddedLibrary(
        ("zlib",), "zlib-ng", "zlib-ng:zlib-ng", lambda _: "Zlib", _zlib_ng_version
    ),
    _EmbeddedLibrary(
        ("_decimal",),
        "mpdecimal",
        "bytereef:mpdecimal",
        lambda _: "BSD-2-Clause",
        _mpdecimal_version,
    ),
    _EmbeddedLibrary(
        ("_zstd",),
        "zstd",
        "facebook:zstandard",
        lambda _: "BSD-3-Clause OR GPL-2.0-only",
        _zstd_version,
    ),
    _EmbeddedLibrary(("_bz2",), "bzip2", "bzip:bzip2", lambda _: "bzip2-1.0.6"),
    _EmbeddedLibrary(("_lzma",), "xz", "tukaani:xz", lambda _: "0BSD"),
    _EmbeddedLibrary(("_ctypes",), "libffi", "libffi_project:libffi", lambda _: "MIT"),
    _EmbeddedLibrary(("_uuid",), "uuid", None, lambda _: "BSD-3-Clause"),
    _EmbeddedLibrary(
        ("_curses", "_curses_panel", "readline"),
        "ncurses",
        "gnu:ncurses",
        lambda _: "X11",
    ),
    _EmbeddedLibrary(
        ("readline",), "libedit", "libedit_project:libedit", lambda _: "BSD-3-Clause"
    ),
)


class CPythonResolver:
    """Files of the CPython installation the bundle was built with."""

    def __init__(
        self,
        registry: Registry,
        *,
        report_embedded_libraries: bool,
        python_build: PythonBuild | None = None,
    ) -> None:
        self._registry = registry
        self._report_embedded_libraries = report_embedded_libraries
        self._python_build = python_build
        self._prefix = _path_key(sys.base_prefix)
        self._modules: set[str] = set()
        version = sys.version.split()[0]
        supplier = {
            "name": "Python Software Foundation",
            "url": ["https://www.python.org"],
        }
        references: tuple[dict, ...] = ()
        properties: tuple[tuple[str, str], ...] = ()
        if python_build is not None:
            supplier = {
                "name": python_build.distribution,
                "url": ["https://github.com/astral-sh/python-build-standalone"],
            }
            references = (
                {
                    "type": "distribution",
                    "url": python_build.archive_url,
                    "hashes": [
                        {"alg": "SHA-256", "content": python_build.archive_sha256}
                    ],
                },
            )
            properties = (
                (
                    f"{PROPERTY_PREFIX}:python-build",
                    f"{python_build.distribution} {python_build.release} "
                    f"{python_build.target}",
                ),
            )
        self.component = Component(
            bom_ref=f"pkg:generic/python@{version}",
            type="application",
            name="CPython",
            version=version,
            purl=f"pkg:generic/python@{version}",
            cpe=f"cpe:2.3:a:python:python:{version}:*:*:*:*:*:*:*",
            supplier=supplier,
            licenses=({"expression": "PSF-2.0"},),
            website="https://www.python.org",
            external_references=references,
            properties=properties,
        )

    def resolve(self, bundled: BundledFile) -> Component | None:
        if not _is_relative_to(_path_key(bundled.source), self._prefix):
            return None
        self._modules.add(Path(bundled.destination).name.split(".", 1)[0])
        return self._registry.add(self.component)

    def _library_version(self, library: _EmbeddedLibrary) -> tuple[str, str] | None:
        """The library's version and where it was taken from, if known."""
        if library.runtime_version is not None:
            try:
                version = library.runtime_version()
            except ImportError:
                return None
            return (version, "runtime") if version else None
        if (
            self._python_build is not None
            and library.key in self._python_build.libraries
        ):
            return self._python_build.libraries[
                library.key
            ], "tools/release_python.json"
        return None

    def add_embedded_libraries(self) -> None:
        """Report the libraries compiled into the bundled CPython modules."""
        if (
            not self._report_embedded_libraries
            or self.component.bom_ref not in self._registry.components
        ):
            return
        modules = self._modules | set(sys.builtin_module_names)
        for library in _EMBEDDED_LIBRARIES:
            if modules.isdisjoint(library.modules):
                continue
            found = self._library_version(library)
            if found is None:
                continue
            version, source = found
            embedded = self._registry.add(
                Component(
                    bom_ref=f"pkg:generic/{library.key}@{version}",
                    type="library",
                    name=library.name,
                    version=version,
                    purl=f"pkg:generic/{library.key}@{version}",
                    cpe=(
                        f"cpe:2.3:a:{library.cpe_product}:{version}:*:*:*:*:*:*:*"
                        if library.cpe_product
                        else None
                    ),
                    licenses=({"expression": library.license_expression(version)},),
                    properties=(
                        (f"{PROPERTY_PREFIX}:embedded-in", "CPython"),
                        (f"{PROPERTY_PREFIX}:version-source", source),
                    ),
                )
            )
            self.component.depends_on.add(embedded.bom_ref)


class UnknownFileResolver:
    """Last resort: report the file itself so nothing bundled goes unlisted."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self.unresolved: list[BundledFile] = []

    def resolve(self, bundled: BundledFile) -> Component:
        self.unresolved.append(bundled)
        name = bundled.destination.replace(os.sep, "/")
        return self._registry.add(
            Component(
                bom_ref=f"file:{name}",
                type="file",
                name=name,
                version=_pe_file_version(bundled.source),
                hashes=_file_hashes(bundled.source),
            )
        )


class Attributor:
    """Attributes a bundled file to the first resolver that knows its origin."""

    def __init__(self, fallback: UnknownFileResolver) -> None:
        self.resolvers: list[Resolver] = []
        self._fallback = fallback

    def resolve(self, bundled: BundledFile) -> Component:
        for resolver in self.resolvers:
            component = resolver.resolve(bundled)
            if component is not None:
                return component
        return self._fallback.resolve(bundled)


# --- dependency graph -----------------------------------------------------


def _declared_project_dependencies() -> set[str]:
    """Runtime and optional dependency names declared in ``pyproject.toml``."""
    if tomllib is None:
        return set()
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]
    requirements = list(project.get("dependencies", ()))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return {canonicalize_name(Requirement(text).name) for text in requirements}


def _link_python_dependencies(
    registry: Registry,
    distributions: dict[str, importlib.metadata.Distribution],
    project: Component,
    project_requirements: set[str],
) -> None:
    refs = {name: _distribution_ref(dist) for name, dist in distributions.items()}
    for name, distribution in distributions.items():
        component = registry.components[refs[name]]
        for required in _requirement_names(distribution.requires or ()):
            if required in refs and required != name:
                component.depends_on.add(refs[required])
    project.depends_on.update(
        refs[name] for name in project_requirements if name in refs
    )


def _mark_reachable(registry: Registry, ref: str, reachable: set[str]) -> None:
    pending = [ref]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(sorted(registry.components[current].depends_on))


def _attach_unreachable(registry: Registry, project: Component) -> None:
    """Make every component reachable from the project in the dependency graph.

    Bundled components that no declared dependency explains (the interpreter,
    system libraries, packages pulled in by an import) are attached directly
    to the project, which is what bundled them.
    """
    reachable: set[str] = set()
    _mark_reachable(registry, project.bom_ref, reachable)
    for ref in sorted(registry.components):
        if ref not in reachable:
            project.depends_on.add(ref)
            _mark_reachable(registry, ref, reachable)


# --- assembling the BOM ---------------------------------------------------


def _project_component(version: str) -> Component:
    return Component(
        bom_ref=f"pkg:pypi/{PROJECT_NAME}@{version}",
        type="application",
        name=PROJECT_NAME,
        version=version,
        purl=f"pkg:pypi/{PROJECT_NAME}@{version}",
        supplier=SUPPLIER,
        licenses=({"expression": PROJECT_LICENSE},),
        description="emmtrix ONNX-to-C Code Generator",
    )


def _project_external_references(
    artifact: Path | None, download_url: str | None
) -> list[dict]:
    references: list[dict] = [
        {"type": "website", "url": PROJECT_WEBSITE},
        {"type": "vcs", "url": PROJECT_REPOSITORY},
    ]
    if artifact is not None:
        references.append(
            {
                "type": "distribution",
                "url": download_url or artifact.name,
                "hashes": list(_file_hashes(artifact)),
            }
        )
    return references


def _timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = (
        datetime.fromtimestamp(int(epoch), timezone.utc)
        if epoch
        else datetime.now(timezone.utc)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_bom(
    *,
    registry: Registry,
    project: Component,
    artifact: Path | None,
    download_url: str | None,
    properties: Iterable[tuple[str, str]],
) -> dict:
    project_json = project.to_json()
    project_json["externalReferences"] = _project_external_references(
        artifact, download_url
    )
    identity = f"{project.name}@{project.version}"
    if artifact is not None:
        identity += f":{artifact.name}:{_file_hashes(artifact)[0]['content']}"
    return {
        "$schema": f"http://cyclonedx.org/schema/bom-{SPEC_VERSION}.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}",
        "version": 1,
        "metadata": {
            "timestamp": _timestamp(),
            "lifecycles": [{"phase": "build"}],
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": f"{PROJECT_NAME} tools/generate_sbom.py",
                        "version": project.version,
                    }
                ]
            },
            "supplier": SUPPLIER,
            "component": project_json,
            "properties": [
                {"name": name, "value": value} for name, value in properties
            ],
        },
        "components": [
            component.to_json()
            for ref, component in sorted(registry.components.items())
            if ref != project.bom_ref
        ],
        "dependencies": [
            {"ref": ref, "dependsOn": sorted(component.depends_on)}
            for ref, component in sorted(registry.components.items())
        ],
    }


def _pyinstaller_loader_origin() -> Path:
    import PyInstaller.loader

    return Path(PyInstaller.loader.__file__).resolve().with_name("pyimod01_archive.py")


# bom-ref prefix of the file components that list the bundle's files.
BUNDLE_FILE_PREFIX = "bundle-file:"
# TOC entries that PyInstaller writes as files of their own; the others go into
# the PYZ archive inside the executable.
_FILE_TYPECODES = frozenset({"EXTENSION", "BINARY", "DATA", "DEPENDENCY", "EXECUTABLE"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_file(relative: str, path: Path) -> dict:
    """The file component of one file in the bundle directory."""
    entry: dict = {
        "type": "file",
        "bom-ref": f"{BUNDLE_FILE_PREFIX}{relative}",
        "name": relative,
    }
    if path.is_symlink():
        entry["properties"] = [
            {
                "name": f"{PROPERTY_PREFIX}:symlink-target",
                "value": os.readlink(path).replace(os.sep, "/"),
            }
        ]
    else:
        entry["hashes"] = [{"alg": "SHA-256", "content": _sha256(path)}]
    return entry


def _destination_of(relative: str, owners: dict[str, str]) -> str | None:
    """The TOC destination of a file in the bundle directory.

    The executable sits at the top of the bundle, everything else in its
    contents directory (``_internal/``), relative to which TOC destinations are.
    """
    if relative in owners:
        return relative
    inner = relative.split("/", 1)[-1]
    return inner if inner in owners else None


def _attach_bundle_files(
    dist: Path,
    owners: dict[str, str],
    registry: Registry,
    unknown: UnknownFileResolver,
) -> None:
    """List every file of the bundle directory, with its hash, under its owner."""
    if not dist.is_dir():
        raise SbomError(f"Bundle directory {dist} does not exist.")
    files: dict[str, list[dict]] = {}
    for path in sorted(dist.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(dist).as_posix()
        destination = _destination_of(relative, owners)
        if destination is None and path.is_symlink():
            # A link to another bundled file belongs to the target's component.
            target = (PurePosixPath(relative).parent / os.readlink(path)).as_posix()
            destination = _destination_of(
                os.path.normpath(target).replace(os.sep, "/"), owners
            )
        if destination is None:
            owner = unknown.resolve(BundledFile(relative, path, "DATA")).bom_ref
        else:
            owner = owners[destination]
        files.setdefault(owner, []).append(_bundle_file(relative, path))
    for owner, entries in files.items():
        registry.components[owner].files = tuple(entries)


def bundle_bom(
    *,
    workpath: Path,
    version: str,
    platform: str,
    artifact: Path | None = None,
    download_url: str | None = None,
    source_roots: Iterable[Path] = PROJECT_SOURCE_ROOTS,
    rpm_resolver_factory: Callable[[Registry], RpmResolver | None] = (
        RpmResolver.from_system
    ),
    pyinstaller_origin: Path | None = None,
    python_build: PythonBuild | None = None,
    dist: Path | None = None,
) -> tuple[dict, list[BundledFile]]:
    """The SBOM of a PyInstaller bundle and the files without a known origin.

    With ``dist`` (the bundle directory PyInstaller wrote), every file in it is
    listed with its SHA-256 under the component it belongs to, so that an
    unpacked bundle can be checked against the SBOM.
    """
    bundled_files = read_bundled_files(workpath)
    _check_interpreter(bundled_files)
    if python_build is not None:
        python_build.check_interpreter()
    registry = Registry()
    project = registry.add(_project_component(version))
    distributions = DistributionResolver(registry)
    rpm = rpm_resolver_factory(registry)
    # A distribution-packaged interpreter links the system libraries, which rpm
    # reports; python.org and python-build-standalone builds carry their own copies.
    cpython = CPythonResolver(
        registry,
        report_embedded_libraries=rpm is None or python_build is not None,
        python_build=python_build,
    )
    unknown = UnknownFileResolver(registry)
    attributor = Attributor(unknown)
    attributor.resolvers += [
        ProjectResolver(
            registry, project, REPO_ROOT / "src" / PROJECT_PACKAGE, source_roots
        ),
        distributions,
        BuildOutputResolver(
            workpath,
            pyinstaller_origin or _pyinstaller_loader_origin(),
            attributor.resolve,
        ),
        MicrosoftRuntimeResolver(registry),
        *([rpm] if rpm is not None else []),
        cpython,
    ]

    file_counts: dict[str, int] = {}
    owners: dict[str, str] = {}
    for bundled in bundled_files:
        ref = attributor.resolve(bundled).bom_ref
        file_counts[ref] = file_counts.get(ref, 0) + 1
        if bundled.typecode in _FILE_TYPECODES:
            # The executable is emx-onnx-cgen itself: PyInstaller's bootloader
            # with the pure-Python modules of every component packed inside.
            owners[bundled.destination.replace(os.sep, "/")] = (
                project.bom_ref if bundled.typecode == "EXECUTABLE" else ref
            )
    if dist is not None:
        _attach_bundle_files(dist, owners, registry, unknown)
    cpython.add_embedded_libraries()
    for ref, count in file_counts.items():
        registry.components[ref].properties += (
            (f"{PROPERTY_PREFIX}:bundled-files", str(count)),
        )

    _link_python_dependencies(
        registry, distributions.bundled, project, _declared_project_dependencies()
    )
    _attach_unreachable(registry, project)
    bom = build_bom(
        registry=registry,
        project=project,
        artifact=artifact,
        download_url=download_url,
        properties=(
            (f"{PROPERTY_PREFIX}:artifact-kind", "pyinstaller-bundle"),
            (f"{PROPERTY_PREFIX}:platform", platform),
        ),
    )
    return bom, unknown.unresolved


def wheel_bom(*, artifact: Path | None = None, download_url: str | None = None) -> dict:
    """The SBOM of the installed emx-onnx-cgen wheel and its runtime closure."""
    try:
        root = importlib.metadata.distribution(PROJECT_NAME)
    except importlib.metadata.PackageNotFoundError as error:
        raise SbomError(
            f"{PROJECT_NAME} is not installed; install the wheel into a fresh "
            "virtual environment and run this script with its interpreter."
        ) from error
    spec = importlib.util.find_spec(PROJECT_PACKAGE)
    if spec is None or spec.origin is None:
        raise SbomError(f"Cannot locate the installed {PROJECT_PACKAGE} package.")
    package_dir = Path(spec.origin).parent

    registry = Registry()
    project = registry.add(_project_component(root.version))
    for source in VENDORED_SOURCES:
        project.depends_on.add(registry.add(source.component(package_dir)).bom_ref)

    project_requirements = _requirement_names(root.requires or ())
    distributions: dict[str, importlib.metadata.Distribution] = {}
    pending = sorted(project_requirements)
    while pending:
        name = pending.pop()
        if name in distributions:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise SbomError(
                f"Runtime dependency {name} of {PROJECT_NAME} is not installed."
            ) from error
        distributions[name] = distribution
        registry.add(_distribution_component(distribution))
        pending.extend(sorted(_requirement_names(distribution.requires or ())))
    _link_python_dependencies(registry, distributions, project, project_requirements)
    return build_bom(
        registry=registry,
        project=project,
        artifact=artifact,
        download_url=download_url
        or f"https://pypi.org/project/{PROJECT_NAME}/{root.version}/",
        properties=(
            (f"{PROPERTY_PREFIX}:artifact-kind", "wheel"),
            (
                f"{PROPERTY_PREFIX}:dependency-resolution",
                "Runtime dependencies in the versions pip resolved when the "
                "release was built; a later installation may resolve newer ones.",
            ),
        ),
    )


# Properties whose value depends on the platform bundle a component was found in.
_PLATFORM_PROPERTIES = frozenset(
    {f"{PROPERTY_PREFIX}:bundled-files", f"{PROPERTY_PREFIX}:python-build"}
)


def _component_from_json(data: dict) -> Component:
    references = list(data.get("externalReferences", ()))
    website = None
    if references and references[0]["type"] == "website":
        website = references.pop(0)["url"]
    return Component(
        bom_ref=data["bom-ref"],
        type=data["type"],
        name=data["name"],
        version=data.get("version"),
        purl=data.get("purl"),
        cpe=data.get("cpe"),
        supplier=data.get("supplier"),
        licenses=tuple(data.get("licenses", ())),
        description=data.get("description"),
        website=website,
        hashes=tuple(data.get("hashes", ())),
        external_references=tuple(references),
        properties=tuple(
            (prop["name"], prop["value"]) for prop in data.get("properties", ())
        ),
        files=tuple(data.get("components", ())),
    )


def _platform_file(entry: dict, platform: str) -> dict:
    """A bundle file component, with its path relative to the combined archive."""
    entry = dict(entry)
    entry["name"] = f"{platform}/{entry['name']}"
    entry["bom-ref"] = f"{BUNDLE_FILE_PREFIX}{entry['name']}"
    return entry


def _unique(items: Iterable) -> tuple:
    seen: dict[str, object] = {}
    for item in items:
        seen.setdefault(json.dumps(item, sort_keys=True), item)
    return tuple(seen.values())


def _platform_properties(
    properties: Iterable[tuple[str, str]], platform: str
) -> tuple[tuple[str, str], ...]:
    """Tag platform-dependent properties and record the platform itself."""
    return (
        *(
            (f"{name}:{platform}" if name in _PLATFORM_PROPERTIES else name, value)
            for name, value in properties
        ),
        (f"{PROPERTY_PREFIX}:platform", platform),
    )


def _merge_component(merged: Component, other: Component) -> None:
    """Fold the same component, found in another platform bundle, into ``merged``."""

    def identity(component: Component) -> dict:
        data = component.to_json()
        for merged_field in ("properties", "externalReferences", "components"):
            data.pop(merged_field, None)
        return data

    if identity(merged) != identity(other):
        raise SbomError(
            f"Component {merged.bom_ref} differs between the platform SBOMs; "
            "they must come from the same release build."
        )
    merged.properties = _unique(merged.properties + other.properties)
    merged.external_references = _unique(
        merged.external_references + other.external_references
    )
    merged.depends_on |= other.depends_on
    merged.files += other.files


def _bom_property(bom: dict, name: str) -> str | None:
    return next(
        (
            prop["value"]
            for prop in bom["metadata"].get("properties", ())
            if prop["name"] == f"{PROPERTY_PREFIX}:{name}"
        ),
        None,
    )


def combined_bom(
    *,
    inputs: Iterable[dict],
    version: str | None = None,
    artifact: Path | None = None,
    download_url: str | None = None,
) -> dict:
    """The SBOM of an archive that holds several platform bundles.

    Each component appears once. It carries an ``emx-onnx-cgen:platform``
    property for every bundle it was found in, and platform-dependent
    properties get the platform as a suffix.
    """
    bundles: dict[str, dict] = {}
    for bom in inputs:
        platform = _bom_property(bom, "platform")
        if _bom_property(bom, "artifact-kind") != "pyinstaller-bundle" or not platform:
            raise SbomError("Only platform bundle SBOMs can be combined.")
        if platform in bundles:
            raise SbomError(f"Two SBOMs for platform {platform}.")
        bundles[platform] = bom
    if not bundles:
        raise SbomError("No SBOMs to combine.")
    versions = {bom["metadata"]["component"]["version"] for bom in bundles.values()}
    if len(versions) != 1 or (version is not None and versions != {version}):
        expected = f" (expected {version})" if version else ""
        raise SbomError(
            f"The platform SBOMs describe versions {sorted(versions)}{expected}."
        )

    registry = Registry()
    project = registry.add(_project_component(versions.pop()))
    for platform, bom in sorted(bundles.items()):
        dependencies = {
            entry["ref"]: set(entry.get("dependsOn", ()))
            for entry in bom["dependencies"]
        }
        for data in (bom["metadata"]["component"], *bom["components"]):
            component = _component_from_json(data)
            component.properties = _platform_properties(component.properties, platform)
            component.depends_on = dependencies.get(component.bom_ref, set())
            component.files = tuple(
                _platform_file(entry, platform) for entry in component.files
            )
            if component.bom_ref == project.bom_ref:
                project.properties = _unique(project.properties + component.properties)
                project.depends_on |= component.depends_on
                project.files += component.files
            elif component.bom_ref in registry.components:
                _merge_component(registry.components[component.bom_ref], component)
            else:
                registry.add(component)
    return build_bom(
        registry=registry,
        project=project,
        artifact=artifact,
        download_url=download_url,
        properties=(
            (f"{PROPERTY_PREFIX}:artifact-kind", "multi-platform-bundle"),
            *(
                (f"{PROPERTY_PREFIX}:platform", platform)
                for platform in sorted(bundles)
            ),
        ),
    )


def _release_identity(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """The version and the asset's download URL from --version/--release-tag."""
    version, download_url = args.version, args.download_url
    if args.release_tag:
        version = args.release_tag.removeprefix("v")
        if download_url is None and args.artifact is not None:
            download_url = (
                f"{PROJECT_REPOSITORY}/releases/download/"
                f"{args.release_tag}/{args.artifact.name}"
            )
    return version, download_url


def _add_identity_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    identity = parser.add_mutually_exclusive_group(required=required)
    identity.add_argument("--version", help="emx-onnx-cgen version")
    identity.add_argument(
        "--release-tag",
        help="GitHub release tag (e.g. v1.4.2); sets the version and the asset's "
        "download URL",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="kind", required=True)
    bundle = subparsers.add_parser("bundle", help="SBOM of a PyInstaller bundle")
    bundle.add_argument(
        "--workpath",
        type=Path,
        required=True,
        help="PyInstaller work directory of the bundle, e.g. build/emx-onnx-cgen",
    )
    _add_identity_arguments(bundle, required=True)
    bundle.add_argument("--platform", required=True, help="e.g. linux-amd64")
    bundle.add_argument(
        "--dist",
        type=Path,
        help="Bundle directory PyInstaller wrote, e.g. dist/emx-onnx-cgen; lists "
        "every file in it with its SHA-256",
    )
    bundle.add_argument(
        "--python-build",
        type=Path,
        help="Pinned CPython build the bundle was made with "
        "(tools/release_python.json, as used by the release workflows)",
    )
    wheel = subparsers.add_parser("wheel", help="SBOM of the installed wheel")
    combine = subparsers.add_parser(
        "combine", help="SBOM of an archive holding several platform bundles"
    )
    combine.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="SBOM of one platform bundle (repeat for each platform)",
    )
    _add_identity_arguments(combine, required=False)
    for subparser in (bundle, wheel, combine):
        subparser.add_argument(
            "--artifact", type=Path, help="Release asset described (for its hashes)"
        )
        subparser.add_argument("--download-url", help="Download URL of the asset")
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.kind == "combine":
            version, download_url = _release_identity(args)
            bom = combined_bom(
                inputs=[
                    json.loads(path.read_text(encoding="utf-8")) for path in args.input
                ],
                version=version,
                artifact=args.artifact,
                download_url=download_url,
            )
        elif args.kind == "bundle":
            version, download_url = _release_identity(args)
            bom, unresolved = bundle_bom(
                workpath=args.workpath,
                version=version,
                platform=args.platform,
                artifact=args.artifact,
                download_url=download_url,
                python_build=(
                    PythonBuild.load(args.python_build, args.platform)
                    if args.python_build
                    else None
                ),
                dist=args.dist,
            )
            for bundled in unresolved:
                print(
                    f"warning: no known origin for {bundled.destination} "
                    f"({bundled.source}); listed as a file component",
                    file=sys.stderr,
                )
        else:
            bom = wheel_bom(artifact=args.artifact, download_url=args.download_url)
    except SbomError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(bom['components'])} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
