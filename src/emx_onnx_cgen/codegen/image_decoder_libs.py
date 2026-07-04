"""Image decoding library registry for the ImageDecoder operator.

The generated C code detects the image format at runtime (magic bytes) and
dispatches to a decoder implementation. Which implementation backs each
format is configured via ``--image-decoder-libs``: a comma-separated priority
list where the first library that supports a format wins.

The registry is data-driven so new libraries can be added without touching
codegen or CLI logic.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

import onnx

from ..errors import CodegenError

#: Formats required by the ONNX ImageDecoder operator specification.
IMAGE_DECODER_FORMATS: tuple[str, ...] = (
    "bmp",
    "jpeg",
    "jpeg2000",
    "png",
    "pnm",
    "tiff",
    "webp",
)


@dataclass(frozen=True)
class ImageDecoderLibrary:
    name: str
    formats: tuple[str, ...]
    #: Libraries passed to the linker as ``-l<name>`` when this decoder is used.
    link_libraries: tuple[str, ...] = ()
    #: pkg-config module used to resolve extra compile flags (include dirs).
    pkg_config_module: str | None = None
    #: Header-only vendored dependency emitted next to the generated C file.
    support_headers: tuple[str, ...] = ()


# Priority is user-defined via --image-decoder-libs; this dict only defines
# which libraries exist. stb is bundled (header-only, public domain) and is
# the default. Its JPEG decoder is not bit-exact with libjpeg; select
# libjpeg-turbo ahead of stb when exact JPEG output is required.
IMAGE_DECODER_LIBRARIES: dict[str, ImageDecoderLibrary] = {
    library.name: library
    for library in (
        ImageDecoderLibrary(
            name="stb",
            formats=("bmp", "jpeg", "png", "pnm"),
            support_headers=("stb_image.h",),
        ),
        ImageDecoderLibrary(
            name="libjpeg-turbo",
            formats=("jpeg",),
            link_libraries=("jpeg",),
        ),
        ImageDecoderLibrary(
            name="libwebp",
            formats=("webp",),
            link_libraries=("webp",),
        ),
        ImageDecoderLibrary(
            name="libtiff",
            formats=("tiff",),
            link_libraries=("tiff",),
        ),
        ImageDecoderLibrary(
            name="openjpeg",
            formats=("jpeg2000",),
            link_libraries=("openjp2",),
            pkg_config_module="libopenjp2",
        ),
    )
}

DEFAULT_IMAGE_DECODER_LIBS: tuple[str, ...] = ("stb",)

#: stb_image is compiled with only the decoders assigned to it enabled.
STB_FORMAT_DEFINES: dict[str, str] = {
    "bmp": "STBI_ONLY_BMP",
    "jpeg": "STBI_ONLY_JPEG",
    "png": "STBI_ONLY_PNG",
    "pnm": "STBI_ONLY_PNM",
}


def image_decoder_function_name(library_name: str) -> str:
    return f"emx_image_decode_{library_name.replace('-', '_')}"


def parse_image_decoder_libs(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated priority list of image decoder libraries."""
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("--image-decoder-libs requires at least one library name")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in IMAGE_DECODER_LIBRARIES:
            known = ", ".join(IMAGE_DECODER_LIBRARIES)
            raise ValueError(
                f"Unknown image decoder library {name!r} " f"(known libraries: {known})"
            )
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


@dataclass(frozen=True)
class ImageDecoderPlan:
    """Resolved format -> library assignment for one compilation."""

    libraries: tuple[str, ...]
    format_to_library: tuple[tuple[str, str], ...]

    def library_for(self, image_format: str) -> str | None:
        for fmt, library in self.format_to_library:
            if fmt == image_format:
                return library
        return None

    @property
    def used_libraries(self) -> tuple[ImageDecoderLibrary, ...]:
        used = {library for _, library in self.format_to_library}
        return tuple(
            IMAGE_DECODER_LIBRARIES[name] for name in self.libraries if name in used
        )

    @property
    def link_libraries(self) -> tuple[str, ...]:
        flags: list[str] = []
        for library in self.used_libraries:
            for flag in library.link_libraries:
                if flag not in flags:
                    flags.append(flag)
        return tuple(flags)

    @property
    def support_headers(self) -> tuple[str, ...]:
        headers: list[str] = []
        for library in self.used_libraries:
            for header in library.support_headers:
                if header not in headers:
                    headers.append(header)
        return tuple(headers)

    @property
    def pkg_config_modules(self) -> tuple[str, ...]:
        modules: list[str] = []
        for library in self.used_libraries:
            if (
                library.pkg_config_module is not None
                and library.pkg_config_module not in modules
            ):
                modules.append(library.pkg_config_module)
        return tuple(modules)


def resolve_image_decoder_plan(libs: Iterable[str]) -> ImageDecoderPlan:
    libraries = tuple(libs)
    for name in libraries:
        if name not in IMAGE_DECODER_LIBRARIES:
            raise CodegenError(f"Unknown image decoder library {name!r}")
    assignments: list[tuple[str, str]] = []
    for image_format in IMAGE_DECODER_FORMATS:
        for name in libraries:
            if image_format in IMAGE_DECODER_LIBRARIES[name].formats:
                assignments.append((image_format, name))
                break
    return ImageDecoderPlan(
        libraries=libraries,
        format_to_library=tuple(assignments),
    )


def model_uses_image_decoder(model: onnx.ModelProto) -> bool:
    def graph_has_image_decoder(graph: onnx.GraphProto) -> bool:
        for node in graph.node:
            if node.op_type == "ImageDecoder":
                return True
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    if graph_has_image_decoder(attribute.g):
                        return True
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    if any(graph_has_image_decoder(g) for g in attribute.graphs):
                        return True
        return False

    return graph_has_image_decoder(model.graph)


def support_header_text(header_name: str) -> str:
    resource = resources.files("emx_onnx_cgen").joinpath("third_party", header_name)
    return resource.read_text(encoding="utf-8")


def prepare_image_decoder_build(
    model: onnx.ModelProto,
    libs: Iterable[str],
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    """Set up an ImageDecoder-aware build next to the generated C file.

    Writes the bundled support headers (e.g. ``stb_image.h``) into
    ``output_dir`` and returns ``(extra_cflags, extra_link_flags)`` for the
    C compiler invocation. Both lists are empty when the model does not use
    the ImageDecoder operator.
    """
    if not model_uses_image_decoder(model):
        return [], []
    plan = resolve_image_decoder_plan(tuple(libs))
    for header in plan.support_headers:
        (output_dir / header).write_text(support_header_text(header), encoding="utf-8")
    cflags = [
        flag for module in plan.pkg_config_modules for flag in pkg_config_cflags(module)
    ]
    link_flags = [f"-l{name}" for name in plan.link_libraries]
    return cflags, link_flags


@lru_cache(maxsize=None)
def pkg_config_cflags(module: str) -> tuple[str, ...]:
    """Best-effort include flags for a pkg-config module (empty on failure)."""
    pkg_config = shutil.which("pkg-config")
    if pkg_config is None:
        return ()
    try:
        result = subprocess.run(
            [pkg_config, "--cflags", module],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ()
    return tuple(flag for flag in result.stdout.split() if flag)
