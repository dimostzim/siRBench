"""Download and verify the immutable siRBench version 2 archive."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ARCHIVE_NAME = "siRBench-v2-2026-09-27-zenodo.zip"
PACKAGE_NAME = "siRBench-v2-2026-09-27"
ARCHIVE_SHA256 = "418f1105c304fad56b85ba9e2605f7caf95e08825c3b1c706af7766f074cc158"
MANIFEST_SHA256 = "28cfd24aecda8fc2bbbe5b2431e1430e1b3effb5cd7c1c229d22d5aeef3a2829"
ARCHIVE_URL = f"https://zenodo.org/records/23001225/files/{ARCHIVE_NAME}?download=1"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(package):
    if digest(package / "SHA256SUMS") != MANIFEST_SHA256:
        raise ValueError("Archive checksum manifest differs from published version 2")
    count = 0
    for line in (package / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (package / name).resolve()
        path.relative_to(package.resolve())
        if digest(path) != expected:
            raise ValueError(f"Archive file checksum differs: {name}")
        count += 1
    return count


def extract_archive(archive, destination):
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("ZIP checksum differs from the published version 2 archive")
    destination.mkdir(parents=True, exist_ok=True)
    package = destination / PACKAGE_NAME
    if package.exists():
        verify_files(package)
        return package
    with tempfile.TemporaryDirectory(dir=destination, prefix=".extract-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                (staging / item.filename).resolve().relative_to(staging.resolve() / PACKAGE_NAME)
            zipped.extractall(staging)
        verify_files(staging / PACKAGE_NAME)
        (staging / PACKAGE_NAME).rename(package)
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Use an already downloaded ZIP")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "archive")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.zip or args.destination / ARCHIVE_NAME
    if args.zip is None and not archive.exists():
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(partial) != ARCHIVE_SHA256:
            raise ValueError("Downloaded ZIP checksum differs; no archive was installed")
        partial.rename(archive)
    print(extract_archive(archive, args.destination).resolve())


if __name__ == "__main__":
    main()
