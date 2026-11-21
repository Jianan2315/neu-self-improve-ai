"""Idempotent workbook upload: compare bytes, never silently replace a benchmark."""
from pathlib import Path


def ensure_workbook(volume, local_path, remote_path):
    local_bytes = Path(local_path).read_bytes()
    try:
        remote_bytes = b"".join(volume.read_file(remote_path))
    except FileNotFoundError:
        with volume.batch_upload() as upload:
            upload.put_file(str(local_path), remote_path)
        return "uploaded"
    if remote_bytes != local_bytes:
        raise ValueError("The cloud workbook differs from the local file. Inspect it; no overwrite was performed.")
    return "reused"
