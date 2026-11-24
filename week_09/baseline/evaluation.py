"""GPT-4o answer grading and accuracy statistics."""
import json
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .config import JUDGE_PROMPT, SCHEMA


def judge_answer(question, reference_answers, response, model, api_key, prompt=JUDGE_PROMPT):
    """Grade with the run's saved instructions. HTTP/schema errors remain unscored."""
    if not api_key:
        raise ValueError("An OpenAI API key is required only for this judging step.")
    payload = {"model": model, "temperature": 0, "max_tokens": 512,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({
                                "question": question, "reference_answers": reference_answers,
                                "candidate_response": response}, ensure_ascii=False)}]}
    request = Request("https://api.openai.com/v1/chat/completions",
                      data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as result:
                body = json.load(result)
            break
        except HTTPError as error:
            if attempt == 2 or error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Judge HTTP {error.code}; prediction remains saved and unscored.") from None
            time.sleep(2 ** attempt)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError("Judge connection failed; prediction remains saved and unscored.") from None
            time.sleep(2 ** attempt)
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
        raise ValueError("Incomplete/refused judge response; do not assign a score.")
    verdict = json.loads(choice["message"]["content"])
    if type(verdict.get("correct")) is not bool or not isinstance(verdict.get("analysis"), str):
        raise ValueError("Invalid judge JSON types; do not assign a score.")
    return {"correct": verdict["correct"], "analysis": verdict["analysis"],
            "requested_model": model, "returned_model": body.get("model"),
            "response_id": body.get("id"), "usage": body.get("usage")}


def summary(rows, manifest):
    settings = manifest["protocol"]["settings"]
    count = settings["question_limit"]
    trials = []
    for trial, seed in enumerate(settings["seeds"], 1):
        generated = [r for r in rows if r["trial"] == trial]
        judged = [r for r in generated if r.get("judge") is not None]
        correct = sum(r["judge"]["correct"] for r in judged)
        complete = len(generated) == count and len(judged) == count
        trials.append({"trial": trial, "seed": seed, "expected": count,
                       "generated": len(generated), "judged": len(judged), "correct": correct,
                       "accuracy_percent": 100 * correct / count if complete else None,
                       "status": "complete" if complete else "incomplete"})
    complete = all(t["status"] == "complete" for t in trials)
    accuracies = [t["accuracy_percent"] for t in trials] if complete else []
    return {"schema_version": SCHEMA, "method": manifest["protocol"]["method"],
            "model_revision": settings["model_revision"],
            "trials": trials, "complete": complete,
            "benchmark_scope": "full_125" if count == 125 else "subset_not_full_benchmark",
            "mean_accuracy_percent": statistics.mean(accuracies) if complete else None,
            "sample_std_percentage_points": statistics.stdev(accuracies) if complete and len(accuracies) > 1 else None,
            "length_limited_responses": sum(r["finish_reason"] == "length" for r in rows)}
