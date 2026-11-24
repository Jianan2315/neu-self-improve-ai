"""Local benchmark reading, experiment files and saved grading decisions."""
import json
import os
from pathlib import Path
from .config import JUDGE_PROMPT, SCHEMA


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_benchmark(path):
    return read_json(path)


def prepare_run(root, run_name, settings, system_prompt, model_id):
    """Create a new experiment directory and save its settings."""
    folder = Path(root) / run_name
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError("run name already used") from None
    manifest = {
        "protocol": {"method": "qwen7b_direct", "model_id": model_id,
                     "dataset": "Bamboogle", "settings": settings,
                     "system_prompt": system_prompt, "judge_prompt": JUDGE_PROMPT},
    }
    write_json(folder / "manifest.json", manifest)
    return folder, manifest


def prepare_response(saved, manifest, questions):
    """Add references and grading fields to answers before saving one result file."""
    saved["schema_version"] = SCHEMA
    settings = manifest["protocol"]["settings"]
    seeds = settings["seeds"]
    batches = saved["output"]["outputs"]
    for batch in batches:
        if [a["question"] for a in batch["responses"]] != [q["question"] for q in questions]:
            raise ValueError("Returned answers do not match submitted questions.")
        batch["trial"] = seeds.index(batch["seed"]) + 1
        for question, answer in zip(questions, batch["responses"]):
            answer.update(question)
            answer.setdefault("judge", None)
    return saved


def prepare_regrading(root, run_name, source_run, judge_model):
    """Copy saved answers into a new run with new grading instructions and no scores."""
    root = Path(root)
    manifest = read_json(root / source_run / "manifest.json")
    saved = read_json(root / source_run / "sequential_response.json")
    manifest["source_run"] = source_run
    manifest["protocol"]["judge_prompt"] = JUDGE_PROMPT
    manifest["protocol"]["settings"]["judge_model"] = judge_model
    for batch in saved["output"]["outputs"]:
        for answer in batch["responses"]:
            answer["judge"] = None
    folder = root / run_name
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError("run name already used") from None
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "sequential_response.json", saved)
    return folder, manifest


def load_records(folder):
    saved = read_json(Path(folder) / "sequential_response.json")
    return response_records(saved)


def response_records(saved):
    """Build in-memory table rows with trial and seed from their parent group."""
    return [dict(answer, trial=batch["trial"], seed=batch["seed"])
            for batch in saved["output"]["outputs"]
            for answer in batch["responses"]]


def save_judgment(folder, trial, question_id, verdict):
    """Persist one grading decision by atomically replacing the result JSON."""
    path = Path(folder) / "sequential_response.json"
    saved = read_json(path)
    matches = [answer for batch in saved["output"]["outputs"] if batch["trial"] == trial
               for answer in batch["responses"] if answer["question_id"] == question_id]
    if len(matches) != 1:
        raise ValueError("Answer to grade was not found.")
    matches[0]["judge"] = verdict
    write_json(path, saved)


# import csv
# from .evaluation import summary
# Optional file export (disabled).
# def export_results(folder, manifest):
#     rows = load_records(folder)
#     report = summary(rows, manifest)
#     write_json(Path(folder) / "summary.json", report)
#     with (Path(folder) / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as out:
#         fields = ["trial", "seed", "question_id", "question", "reference_answers",
#                   "response", "correct", "judge_analysis", "input_tokens", "output_tokens",
#                   "finish_reason"]
#         writer = csv.DictWriter(out, fieldnames=fields)
#         writer.writeheader()
#         for row in rows:
#             item = {k: row.get(k) for k in fields}
#             item["reference_answers"] = json.dumps(row["reference_answers"], ensure_ascii=False)
#             item["correct"] = row["judge"]["correct"] if row.get("judge") else ""
#             item["judge_analysis"] = row["judge"]["analysis"] if row.get("judge") else ""
#             writer.writerow(item)
#     return rows, report
