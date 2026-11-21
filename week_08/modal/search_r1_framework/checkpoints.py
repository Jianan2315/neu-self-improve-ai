"""Verify persisted single-GPU veRL checkpoints before explicit restoration."""
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def required_files(algorithm):
    if algorithm not in ("ppo", "grpo"):
        raise ValueError("Expected PPO or GRPO.")
    names = ["data.pt", "actor/lora_adapter/adapter_config.json",
             "actor/lora_adapter/adapter_model.safetensors"]
    for role in (["actor", "critic"] if algorithm == "ppo" else ["actor"]):
        names += [f"{role}/{stem}_world_size_1_rank_0.pt"
                  for stem in ("model", "optim", "extra_state")]
        names.append(f"{role}/fsdp_config.json")
    return names


def check_structure(folder, algorithm):
    for name in required_files(algorithm):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint file: {path}")
        if path.suffix == ".pt" and not zipfile.is_zipfile(path):
            raise ValueError(f"Incomplete PyTorch checkpoint archive: {path}")
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            if path.name == "fsdp_config.json" and value.get("world_size") != 1:
                raise ValueError("This deployment restores only single-GPU checkpoints.")


def seal_latest_checkpoint(output, algorithm):
    """Record hashes only after veRL finishes all workers and writes its marker.

    The caller commits the Volume after this receipt is written. A receipt is
    immutable; restoration rechecks every recorded file against its saved hash.
    """
    output = Path(output)
    marker = output / "checkpoints/latest_checkpointed_iteration.txt"
    if not marker.exists():
        return None
    raw = marker.read_text().strip()
    if not raw:  # The native marker itself may be in the middle of a write.
        return None
    step = int(raw)
    receipt_path = output / "checkpoint_receipts" / f"{step}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    folder = output / "checkpoints" / f"global_step_{step}"
    check_structure(folder, algorithm)
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            before = path.stat()
            checksum = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Checkpoint changed during verification: {path}")
            files[path.relative_to(folder).as_posix()] = {
                "bytes": after.st_size, "sha256": checksum}
    receipt = {"schema": 1, "step": step, "algorithm": algorithm,
               "manifest_sha256": digest(output / "manifest.json"), "files": files}
    write_json(receipt_path, receipt)
    print(f"CHECKPOINT_VERIFIED step={step} files={len(files)}; awaiting Volume commit", flush=True)
    return receipt


def select_checkpoint(output, algorithm, target_steps):
    """Fail closed, or explicitly report fallback to an older verified checkpoint."""
    output = Path(output)
    candidates = sorted((output / "checkpoint_receipts").glob("*.json"),
                        key=lambda p: int(p.stem), reverse=True)
    rejected = []
    for path in candidates:
        step = int(path.stem)
        folder = output / "checkpoints" / f"global_step_{step}"
        try:
            receipt = json.loads(path.read_text())
            if (receipt.get("schema") != 1 or receipt.get("algorithm") != algorithm
                    or receipt.get("step") != step or step < 1 or step > target_steps
                    or receipt.get("manifest_sha256") != digest(output / "manifest.json")):
                raise ValueError("Checkpoint receipt does not match this run.")
            check_structure(folder, algorithm)
            files = receipt.get("files", {})
            if not set(required_files(algorithm)).issubset(files):
                raise ValueError("Checkpoint receipt omits required restoration files.")
            for name, expected in files.items():
                item = (folder / name).resolve()
                if not item.is_relative_to(folder.resolve()):
                    raise ValueError("Invalid checkpoint receipt path.")
                if (not item.is_file() or item.stat().st_size != expected["bytes"]
                        or digest(item) != expected["sha256"]):
                    raise ValueError(f"Checkpoint file changed or was not fully persisted: {name}")
            return {"step": step, "path": str(folder.resolve()),
                    "target_steps": target_steps, "remaining_steps": target_steps - step,
                    "rejected_newer_checkpoints": rejected}
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected.append({"step": step, "error": str(error)})
    raise ValueError("No complete, verified checkpoint is available. Refusing to start from scratch. "
                     + json.dumps(rejected))


def validate_resume_request(output, run_info):
    output = Path(output)
    if not output.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {output}")
    if (output / "finished.json").exists():
        raise ValueError("This training run has already finished; do not resubmit it.")
    previous = json.loads((output / "manifest.json").read_text())
    for key in ("job", "settings", "preparation_name", "data_manifest", "smoke_dev_questions"):
        if previous.get(key) != run_info.get(key):
            raise ValueError(f"Resume configuration mismatch: {key}")
    for name in ("tools.yaml", "agents.yaml"):
        if not (output / "runtime_config" / name).is_file():
            raise ValueError(f"Missing original runtime configuration: {name}")
    return select_checkpoint(output, run_info["job"].removeprefix("train_"),
                             run_info["settings"]["steps"])


def activate_recovery(output, recovery):
    """Preserve rejected receipts before rerunning those step numbers."""
    from datetime import datetime, timezone
    output = Path(output)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for receipt in (output / "checkpoint_receipts").glob("*.json"):
        if int(receipt.stem) > recovery["step"]:
            archive = output / "rejected_checkpoint_receipts" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            receipt.replace(archive / receipt.name)
    marker = output / "checkpoints/latest_checkpointed_iteration.txt"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(str(recovery["step"]), encoding="utf-8")
    temporary.replace(marker)
    write_json(output / "resume_history" / f"{stamp}.json", recovery)
