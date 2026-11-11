#!/usr/bin/env python3
"""Combine the per-platform release bundles into one archive.

The archive holds every platform bundle in its own directory::

    emx-onnx-cgen/
      linux-amd64/     emx-onnx-cgen, _internal/
      windows-amd64/   emx-onnx-cgen.exe, _internal/

The bundles cannot share one directory: both have an ``_internal/`` with
platform-specific files of the same name (``base_library.zip``, the numpy
libraries, ...).

The Linux bundle is copied member by member from its tar archive, so its
symlinks survive even when this script runs on Windows. The output is
deterministic: sorted members, fixed owner, and timestamps from
``SOURCE_DATE_EPOCH`` (default 1980-01-01).
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TOP_DIRECTORY = "emx-onnx-cgen"
# 1980-01-01, the earliest timestamp a zip archive can hold.
DEFAULT_EPOCH = 315532800


class CombineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Member:
    """One entry of the combined archive, relative to its platform directory."""

    path: PurePosixPath
    kind: bytes  # tarfile.REGTYPE, DIRTYPE or SYMTYPE
    mode: int
    data: bytes = b""
    linkname: str = ""


def _safe_path(name: str, archive: Path) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise CombineError(f"{archive.name}: unsafe member path {name!r}")
    return path


def _tar_members(archive: Path, strip: str) -> list[Member]:
    """Members of a tar bundle below its ``strip`` top directory."""
    members = []
    with tarfile.open(archive, "r:*") as tar:
        for info in tar.getmembers():
            path = _safe_path(info.name, archive)
            if path.parts[:1] != (strip,):
                raise CombineError(
                    f"{archive.name}: {info.name!r} is not below {strip}/"
                )
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts:
                continue
            if info.isdir():
                members.append(Member(relative, tarfile.DIRTYPE, info.mode))
            elif info.issym():
                members.append(
                    Member(relative, tarfile.SYMTYPE, info.mode, linkname=info.linkname)
                )
            elif info.isfile():
                data = tar.extractfile(info).read()
                members.append(Member(relative, tarfile.REGTYPE, info.mode, data))
            else:
                raise CombineError(f"{archive.name}: unsupported member {info.name!r}")
    return members


def _zip_members(archive: Path) -> list[Member]:
    """Members of a zip bundle, which has no top directory."""
    members = []
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = _safe_path(info.filename.rstrip("/"), archive)
            if info.is_dir():
                members.append(Member(relative, tarfile.DIRTYPE, 0o755))
            else:
                # Windows zips carry no Unix modes; keep the executable runnable
                # when the archive is unpacked on a Unix host.
                mode = 0o755 if relative.suffix.lower() == ".exe" else 0o644
                members.append(
                    Member(relative, tarfile.REGTYPE, mode, bundle.read(info))
                )
    return members


def _with_parent_directories(members: list[Member]) -> list[Member]:
    present = {member.path for member in members}
    directories = {
        parent
        for member in members
        for parent in member.path.parents
        if parent != PurePosixPath(".") and parent not in present
    }
    return members + [Member(path, tarfile.DIRTYPE, 0o755) for path in directories]


def combine(bundles: dict[str, list[Member]], output: Path, epoch: int) -> None:
    """Write the platform bundles below ``emx-onnx-cgen/<platform>/``."""
    directory = Member(PurePosixPath(), tarfile.DIRTYPE, 0o755)
    entries = [(PurePosixPath(TOP_DIRECTORY), directory)]
    for platform, members in bundles.items():
        root = PurePosixPath(TOP_DIRECTORY, platform)
        entries.append((root, directory))
        entries += [
            (root / member.path, member) for member in _with_parent_directories(members)
        ]
    entries.sort(key=lambda entry: entry[0].parts)
    # gzip stores a timestamp in its header too.
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for path, member in entries:
            info = tarfile.TarInfo(str(path))
            info.type = member.kind
            info.mode = member.mode & 0o7777
            info.mtime = epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.linkname = member.linkname
            info.size = len(member.data)
            tar.addfile(info, io.BytesIO(member.data) if member.data else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--linux", type=Path, required=True, help="linux-amd64 .tar.gz")
    parser.add_argument(
        "--windows", type=Path, required=True, help="windows-amd64 .zip"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_EPOCH))
    try:
        bundles = {
            "linux-amd64": _tar_members(args.linux, TOP_DIRECTORY),
            "windows-amd64": _zip_members(args.windows),
        }
        combine(bundles, args.output, epoch)
    except (CombineError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
