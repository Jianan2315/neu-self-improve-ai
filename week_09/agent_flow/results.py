"""AgentFlow manifest creation; shared result storage uses the baseline helpers."""
from baseline.config import MODEL_ID, JUDGE_PROMPT
from baseline.results import (read_json, write_json, load_benchmark, prepare_response,
                              response_records, load_records, save_judgment)
from .prompts import PROMPTS, TOOLS


def prepare_run(root, run_name, settings):
    folder = root / run_name
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError("run name already used") from None
    manifest = {"protocol": {"method": "agentflow", "model_id": MODEL_ID,
                             "dataset": "Bamboogle", "settings": settings,
                             "module_prompts": PROMPTS, "tools": TOOLS,
                             "judge_prompt": JUDGE_PROMPT}}
    write_json(folder / "manifest.json", manifest)
    return folder, manifest
