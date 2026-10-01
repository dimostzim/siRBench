import hashlib
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_archive
from reproduce_dataset import audited_replacements


def package_zip(tmp_path, name="records.csv"):
    archive = tmp_path / "example.zip"
    content = b"id,value\n1,0.5\n"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(f"{fetch_archive.PACKAGE_NAME}/{name}", content)
        zipped.writestr(f"{fetch_archive.PACKAGE_NAME}/SHA256SUMS",
                        hashlib.sha256(content).hexdigest() + f"  {name}\n")
    return archive


def test_archive_rejects_wrong_zip_before_extracting(tmp_path):
    archive = package_zip(tmp_path)
    with pytest.raises(ValueError, match="ZIP checksum"):
        fetch_archive.extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_archive_verifies_extraction_and_existing_files(tmp_path, monkeypatch):
    archive = package_zip(tmp_path)
    monkeypatch.setattr(fetch_archive, "ARCHIVE_SHA256", fetch_archive.digest(archive))
    with zipfile.ZipFile(archive) as zipped:
        checksum = hashlib.sha256(zipped.read(f"{fetch_archive.PACKAGE_NAME}/SHA256SUMS")).hexdigest()
    monkeypatch.setattr(fetch_archive, "MANIFEST_SHA256", checksum)
    package = fetch_archive.extract_archive(archive, tmp_path / "output")
    assert fetch_archive.extract_archive(archive, tmp_path / "output") == package
    (package / "records.csv").write_text("changed")
    with pytest.raises(ValueError, match="file checksum"):
        fetch_archive.extract_archive(archive, tmp_path / "output")
    (package / "SHA256SUMS").write_text("")
    with pytest.raises(ValueError, match="checksum manifest"):
        fetch_archive.extract_archive(archive, tmp_path / "output")


def test_archive_rejects_escape_path(tmp_path, monkeypatch):
    archive = package_zip(tmp_path, "../../outside.csv")
    monkeypatch.setattr(fetch_archive, "ARCHIVE_SHA256", fetch_archive.digest(archive))
    with pytest.raises(ValueError):
        fetch_archive.extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "outside.csv").exists()


def correction():
    return {"old_siRNA": "A" * 19, "siRNA": "U" * 19,
            "old_context": "T" * 57, "context": "A" * 57,
            "efficiency": "0.5", "upstream_line": "2"}


def test_correction_preserves_label_and_central_target():
    result = audited_replacements([correction()])[("A" * 19, "T" * 57)]
    assert result == {"siRNA": "U" * 19, "mRNA": "A" * 19,
                      "extended_mRNA": "A" * 57, "efficiency": 0.5, "upstream_line": "2"}


def test_invalid_or_duplicate_correction_is_rejected():
    row = correction()
    with pytest.raises(ValueError, match="Duplicate"):
        audited_replacements([row, row])
    row["context"] = "C" * 57
    with pytest.raises(ValueError, match="strand transformation"):
        audited_replacements([row])
