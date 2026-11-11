#!/usr/bin/env python3
"""Download the CPython build that the release bundles are made with.

Both release workflows build their PyInstaller bundle with the same
python-build-standalone release, pinned with its SHA-256 per platform in
``tools/release_python.json``. This script downloads the archive for one
platform, checks the hash and unpacks it. It prints the path of the Python
executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "release_python.json"


class FetchError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def archive_url(config: dict, platform: str) -> str:
    try:
        target = config["platforms"][platform]["target"]
    except KeyError:
        known = ", ".join(sorted(config["platforms"]))
        raise FetchError(f"Unknown platform {platform!r}; known: {known}") from None
    return config["url_template"].format(
        release=config["release"],
        python_version=config["python_version"],
        target=target,
        variant=config["variant"],
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def python_executable(install_dir: Path) -> Path:
    if (install_dir / "python.exe").is_file():
        return install_dir / "python.exe"
    return install_dir / "bin" / "python3"


def fetch(config: dict, platform: str, dest: Path) -> Path:
    """Download, verify and unpack the archive; return the Python executable."""
    url = archive_url(config, platform)
    expected = config["platforms"][platform]["sha256"]
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / url.rsplit("/", 1)[1]
    if not archive.is_file() or sha256_of(archive) != expected:
        print(f"Downloading {url}", file=sys.stderr)
        download(url, archive)
    actual = sha256_of(archive)
    if actual != expected:
        archive.unlink()
        raise FetchError(
            f"SHA-256 mismatch for {archive.name}: expected {expected}, got {actual}. "
            "Update tools/release_python.json only from the release's SHA256SUMS."
        )
    install_dir = dest / "python"
    if install_dir.exists():
        shutil.rmtree(install_dir)
    with tarfile.open(archive, "r:gz") as tar:
        # The "data" filter rejects absolute paths and links leaving the
        # destination; it exists from Python 3.10.12/3.11.4 on.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
    executable = python_executable(install_dir)
    if not executable.is_file():
        raise FetchError(f"{archive.name} does not contain {executable}")
    return executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--platform", required=True, help="Release platform, e.g. linux-amd64"
    )
    parser.add_argument(
        "--dest", required=True, type=Path, help="Directory to download and unpack into"
    )
    args = parser.parse_args(argv)
    try:
        executable = fetch(load_config(), args.platform, args.dest)
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
