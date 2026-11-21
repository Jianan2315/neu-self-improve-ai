"""Launch the existing veRL CLI and summarize its evaluation export."""
import json
from collections import deque
from pathlib import Path
import subprocess
import sys
import os
import signal
import threading
import time
from queue import Queue, Empty


class CheckpointTestInterrupted(RuntimeError):
    """An explicitly requested recovery test stopped a real training subprocess."""


def run_framework(arguments, directory, backend_failures=None, progress=None,
                  interrupt_after_checkpoint=None):
    directory = Path(directory)
    # Hydra composition is checked separately before allocating worker processes.
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *arguments]
    composed = subprocess.run([*command, "--cfg", "job", "--resolve"], check=True,
                              text=True, capture_output=True)
    (directory / "resolved_config.yaml").write_text(composed.stdout, encoding="utf-8")
    with (directory / "console.log").open("a", encoding="utf-8") as log:
        recent_lines = deque(maxlen=60)
        error_lines = deque(maxlen=8)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=(os.name == "posix"))
        lines = Queue()
        def read_lines():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        next_progress = time.monotonic()
        try:
            while True:
                if interrupt_after_checkpoint is not None:
                    marker = directory / "checkpoints/latest_checkpointed_iteration.txt"
                    raw = marker.read_text().strip() if marker.exists() else ""
                    if raw and int(raw) >= interrupt_after_checkpoint:
                        # Persist and verify the native save before simulating interruption.
                        progress()
                        raise CheckpointTestInterrupted(f"Interrupted after checkpoint {raw}")
                if progress and time.monotonic() >= next_progress:
                    progress()
                    next_progress = time.monotonic() + 60
                # Check even when veRL emits no output while waiting for a tool.
                if backend_failures:
                    raise RuntimeError(f"Search backend failed: {backend_failures[0][:2000]}. "
                                       f"See {directory / 'search_requests.jsonl'}; stopping veRL.")
                try:
                    line = lines.get(timeout=0.2)
                except Empty:
                    continue
                if line is None:
                    break
                # Bound the remote exception while keeping the complete log on disk.
                if len(line) <= 1500:
                    recent_lines.append(line)
                    if any(marker in line.lower() for marker in ("error:", "exception:", "invalid ")):
                        error_lines.append(line)
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
            if returncode:
                detail = "".join(error_lines) or "".join(recent_lines)[-4000:]
                raise RuntimeError(
                    f"veRL exited with code {returncode}. Full log: {directory / 'console.log'}\n"
                    f"{detail[:6000]}"
                )
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=10)
            reader.join(timeout=2)
            # Preserve trailing worker output after an early backend failure.
            while not lines.empty():
                pending = lines.get_nowait()
                if pending is not None:
                    log.write(pending)


def final_actor(directory, expected_steps):
    root = Path(directory) / "checkpoints"
    marker = root / "latest_checkpointed_iteration.txt"
    step = int(marker.read_text().strip())
    if step != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} completed steps; checkpoint marker reports {step}.")
    adapter = root / f"global_step_{step}" / "actor" / "lora_adapter"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(f"Missing trained adapter artifact: {adapter / name}")
    return str(adapter)


def summarize_evaluation(directory, method, expected_ids):
    paths = list((Path(directory) / "evaluation").glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError("An inference-only run must produce exactly one evaluation export.")
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line]
    ids = [r["question_id"] for r in rows]
    if len(rows) != len(expected_ids) or set(ids) != set(expected_ids) or len(ids) != len(set(ids)):
        raise ValueError("Evaluation omitted, duplicated, or substituted benchmark questions.")
    if any(float(r["score"]) not in (0.0, 1.0) for r in rows):
        raise ValueError("Expected binary normalized exact-match scores.")
    correct = sum(float(r["score"]) for r in rows)
    return {"method": method, "questions": len(rows), "correct": int(correct),
            "exact_match": correct / len(rows), "predictions_file": str(paths[0])}
