import os
from pathlib import Path
import subprocess

TOOL = Path(__file__).resolve().parents[2] / 'competitors/tools/ensirna'


def test_unverified_image_manifest_is_rejected_without_installing(tmp_path):
    manifest = tmp_path / 'unverified.json'
    manifest.write_text('{}\n')
    output = tmp_path / 'runtime'
    result = subprocess.run(['bash', str(TOOL / 'fetch_rosetta.sh')], capture_output=True, text=True,
        env={**os.environ, 'ROSETTA_IMAGE_MANIFEST': str(manifest), 'ROSETTA_OUT_DIR': str(output)})
    assert result.returncode != 0 and 'SHA256 mismatch' in result.stderr
    assert not output.exists()


def test_existing_unmatched_runtime_is_preserved(tmp_path):
    output = tmp_path / 'runtime'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('Existing user runtime must not be replaced.\n')
    result = subprocess.run(['bash', str(TOOL / 'fetch_rosetta.sh')], capture_output=True, text=True,
        env={**os.environ, 'ROSETTA_OUT_DIR': str(output)})
    assert result.returncode != 0 and 'SHA256 mismatch' in result.stderr
    assert marker.read_text() == 'Existing user runtime must not be replaced.\n'
