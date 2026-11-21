"""Small validation subset for functional checks, never for benchmark reporting."""
import json
from pathlib import Path


def validate_completed_smoke(result, job, preparation, settings, count=4):
    """Reuse only a saved successful smoke run with the requested configuration."""
    if result == {"status": "not_written_yet"}:
        return False
    expected = {"job": job, "preparation_name": preparation,
                "settings": settings.to_dict(), "smoke_dev_questions": count}
    if not isinstance(result, dict) or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Existing smoke completion has different settings; inspect it before reusing the run name.")
    adapter = result.get("adapter_path")
    if not isinstance(adapter, str) or not adapter.endswith(f"global_step_{settings.steps}/actor/lora_adapter"):
        raise ValueError("Smoke completion is missing its final adapter path.")
    return True


def smoke_overrides(arguments, prepared, output, count):
    import pandas as pd
    prepared, output = Path(prepared), Path(output)
    frame = pd.read_parquet(prepared / "dev.parquet")
    if type(count) is not int or not 1 <= count <= len(frame):
        raise ValueError("Smoke validation count exceeds the prepared development set.")
    subset = frame.iloc[:count].copy()
    path = output / "smoke_dev.parquet"
    subset.to_parquet(path, index=False)
    (output / "smoke_validation.json").write_text(json.dumps({
        "purpose": "Functional check only; not a benchmark result",
        "source": str(prepared / "dev.parquet"), "questions": count,
        "question_ids": [item["question_id"] for item in subset["extra_info"]],
    }, indent=2), encoding="utf-8")
    replacements = {"data.val_files": str(path), "data.val_batch_size": count,
                    "trainer.val_before_train": False}
    return [f"{key}={json.dumps(replacements[key])}" if key in replacements else argument
            for argument in arguments for key in [argument.split("=", 1)[0]]]
