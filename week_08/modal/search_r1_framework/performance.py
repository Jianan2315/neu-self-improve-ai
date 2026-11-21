"""Short, explicit performance probes using native veRL metrics and nvidia-smi.

No training algorithm is replaced. Reports describe observed short workloads,
not a guaranteed capacity limit or an automatic choice of production settings.
"""
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time
from search_r1_framework.search_batcher import RETRIEVAL_EXECUTION

# Modal requires a finite function timeout. Use its maximum; no shorter work cap.
PERFORMANCE_HARD_TIMEOUT_SECONDS = 24 * 3600


def execution_completed(report):
    """Accept completed work with warnings, including persisted legacy reports."""
    expected = report.get("settings", {}).get("steps", 0)
    return (report.get("status") in ("completed", "completed_with_warnings", "interrupted_measurement")
            and report.get("error") is None
            and type(expected) is int and expected > 0
            and set(report.get("completed_steps", [])) == set(range(1, expected + 1))
            and report.get("measurement", {}).get("complete_measurement") is True)


def validate_reusable_report(report, algorithm, preparation, settings, probe):
    """Never overwrite unfinished work or reuse different experimental settings."""
    expected = probe.settings_for(settings).to_dict()
    if not execution_completed(report):
        raise ValueError("Existing run is unfinished or failed. Inspect it before choosing how to continue.")
    if (report.get("algorithm") != algorithm or report.get("preparation_name") != preparation
            or report.get("settings") != expected or report.get("probe") != probe.to_dict()
            or report.get("retrieval_execution") != RETRIEVAL_EXECUTION):
        raise ValueError("Existing report has different settings or retrieval code; do not reuse or overwrite it.")
    return report


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class PerformanceProbe:
    micro_batch: int = 2
    warmup_steps: int = 2
    measured_steps: int = 8
    vllm_memory_fraction: float = 0.3
    max_num_seqs: int = 32
    cpu_offload: bool = True
    gradient_checkpointing: bool = True
    long_sequence_check: bool = False
    fixed_question_indices: list[int] | None = None

    def settings_for(self, base):
        if type(self.warmup_steps) is not int or self.warmup_steps < 1:
            raise ValueError("Use at least one excluded warmup step.")
        if type(self.measured_steps) is not int or not 2 <= self.measured_steps <= 50:
            raise ValueError("Use 2 to 50 measured steps for a short probe.")
        if self.warmup_steps > 10:
            raise ValueError("Use no more than ten warmup steps.")
        records = base.questions_per_step * base.responses_per_question
        if self.fixed_question_indices is not None:
            if (not self.long_sequence_check or len(self.fixed_question_indices) != base.questions_per_step
                    or len(set(self.fixed_question_indices)) != len(self.fixed_question_indices)
                    or any(type(i) is not int or i < 0 or i >= base.train_questions for i in self.fixed_question_indices)):
                raise ValueError("Fixed questions require a unique valid row per prompt in a capacity probe.")
        if type(self.micro_batch) is not int or self.micro_batch < 1 or records % self.micro_batch:
            raise ValueError("Micro-batch must divide the unchanged response batch.")
        if type(self.max_num_seqs) is not int or not 1 <= self.max_num_seqs <= records:
            raise ValueError("Generation concurrency must be between one and the record count.")
        if type(self.cpu_offload) is not bool or type(self.gradient_checkpointing) is not bool:
            raise ValueError("Offload and checkpointing must be booleans.")
        if type(self.long_sequence_check) is not bool:
            raise ValueError("long_sequence_check must be a boolean.")
        return replace(base, steps=self.warmup_steps + self.measured_steps,
                       micro_batch_per_gpu=self.micro_batch,
                       vllm_memory_fraction=self.vllm_memory_fraction).validate()

    def to_dict(self):
        return asdict(self)


def performance_overrides(arguments, probe):
    """Keep model, data, objective and effective batch unchanged; disable test/save."""
    values = {k: json.loads(v) for k, v in (a.split("=", 1) for a in arguments)}
    values.update({
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.save_freq": -1,
        "trainer.resume_mode": "disable",
        "actor_rollout_ref.rollout.max_num_seqs": probe.max_num_seqs,
        "actor_rollout_ref.model.enable_gradient_checkpointing": probe.gradient_checkpointing,
    })
    # The critic also has its own activation-checkpointing setting.
    values["critic.model.enable_gradient_checkpointing"] = probe.gradient_checkpointing
    if probe.long_sequence_check:
        directory = Path(values["trainer.default_local_dir"]).parent
        values["data.train_files"] = str(directory / "capacity_train.parquet")
        values["data.val_files"] = str(directory / "capacity_val.parquet")
        values["data.shuffle"] = False
        # One full validation batch at the same length boundary; measured separately.
        values["trainer.test_freq"] = probe.warmup_steps + probe.measured_steps
    for prefix in ("actor_rollout_ref.actor.fsdp_config", "critic.model.fsdp_config"):
        values[f"{prefix}.param_offload"] = probe.cpu_offload
        values[f"{prefix}.optimizer_offload"] = probe.cpu_offload
    return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]


def parse_gpu_sample(output, timestamp):
    row = next(csv.reader(output.strip().splitlines()))
    if len(row) != 6:
        raise ValueError("Unexpected nvidia-smi fields.")
    result = {"time": timestamp, "gpu_name": row[0].strip()}
    for key, raw in zip(("memory_total_mib", "memory_used_mib", "gpu_utilization_pct",
                         "power_w", "power_limit_w"), row[1:]):
        try:
            value = float(raw.strip())
            result[key] = value if math.isfinite(value) else None
        except ValueError:
            result[key] = None
    return result


class GpuSampler:
    """Sample the single allocated GPU; never infer missing data."""
    def __init__(self, directory, interval_seconds=2, sample_cpu=False):
        self.path = Path(directory) / "gpu_samples.jsonl"
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = None
        self.interval_seconds = interval_seconds
        self.sample_cpu = sample_cpu
        self.previous_cpu = None

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        command = ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw,power.limit",
                   "--format=csv,noheader,nounits"]
        with self.path.open("w", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
                    sample = parse_gpu_sample(result.stdout, time.time())
                    if self.sample_cpu:
                        try:
                            root = Path('/sys/fs/cgroup')
                            stats = dict(line.split() for line in (root / 'cpu.stat').read_text().splitlines())
                            now = time.monotonic()
                            usage = int(stats['usage_usec']) / 1e6
                            if self.previous_cpu:
                                previous_time, previous_usage = self.previous_cpu
                                sample['training_container_cpu_cores_used'] = (usage - previous_usage) / (now - previous_time)
                            self.previous_cpu = (now, usage)
                            sample['training_container_ram_bytes'] = int((root / 'memory.current').read_text())
                        except (OSError, ValueError, KeyError) as error:
                            sample['cpu_sampling_error'] = str(error)
                    self.samples.append(sample)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
                    if len(self.errors) < 5:
                        self.errors.append(str(error))
                self.stop_event.wait(self.interval_seconds)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=12)


def read_native_metrics(directory):
    """Use TensorBoard's event reader instead of parsing human console formatting."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(Path(directory) / "tensorboard"), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows = {}
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            row = rows.setdefault(event.step, {"step": event.step, "metrics": {}})
            row["metrics"][tag] = float(event.value)
            if tag == "timing_s/step":
                row["wall_time"] = event.wall_time
    return [rows[key] for key in sorted(rows)]


def summarize_performance(rows, samples, probe, records_per_step):
    expected = set(range(probe.warmup_steps + 1, probe.warmup_steps + probe.measured_steps + 1))
    selected = [row for row in rows if row["step"] in expected]
    complete = (len(selected) == len(expected) and {row["step"] for row in selected} == expected
                and all(math.isfinite(row["metrics"].get("timing_s/step", float('nan')))
                        and row["metrics"]["timing_s/step"] > 0 for row in selected))
    summary = {"complete_measurement": complete, "observed_measured_steps": len(selected),
               "requested_measured_steps": probe.measured_steps, "records_per_update": records_per_step}
    if not complete:
        return summary
    durations = [row["metrics"]["timing_s/step"] for row in selected]
    tags = set().union(*(row["metrics"].keys() for row in selected))
    metric_means = {}
    for tag in sorted(tags):
        values = [row["metrics"][tag] for row in selected if tag in row["metrics"]
                  and math.isfinite(row["metrics"][tag])]
        if len(values) == len(selected):
            metric_means[tag] = statistics.mean(values)
    # Event logging happens after each step. These are approximate GPU sample windows.
    windows = [(row["wall_time"] - row["metrics"]["timing_s/step"], row["wall_time"])
               for row in selected if "wall_time" in row]
    measured_gpu = [s for s in samples if any(start <= s["time"] <= end for start, end in windows)]
    gpu = gpu_summary(measured_gpu)
    summary.update({"mean_step_seconds": statistics.mean(durations),
                    "median_step_seconds": statistics.median(durations),
                    "min_step_seconds": min(durations), "max_step_seconds": max(durations),
                    "records_per_second": records_per_step * len(selected) / sum(durations),
                    "native_metric_means": metric_means, "measured_window_gpu": gpu})
    return summary


def gpu_summary(samples):
    def values(key):
        return [s[key] for s in samples if isinstance(s.get(key), (int, float))]
    used, totals, utilization, power = (values(key) for key in
        ("memory_used_mib", "memory_total_mib", "gpu_utilization_pct", "power_w"))
    free = [s['memory_total_mib'] - s['memory_used_mib'] for s in samples
            if isinstance(s.get('memory_total_mib'), (int, float))
            and isinstance(s.get('memory_used_mib'), (int, float))]
    return {"samples": len(samples), "gpu_names": sorted({s['gpu_name'] for s in samples}),
            "sampled_peak_memory_mib": max(used) if used else None,
            "gpu_memory_total_mib": min(totals) if totals else None,
            "sampled_min_free_memory_mib": min(free) if free else None,
            "mean_gpu_utilization_pct": statistics.mean(utilization) if utilization else None,
            "mean_power_w": statistics.mean(power) if power else None}


def write_report(directory, probe, settings, algorithm, sampler, elapsed, error=None,
                 running=False, phases=None):
    directory = Path(directory)
    report = {"kind": "performance_only", "algorithm": algorithm, "probe": probe.to_dict(),
              "retrieval_execution": RETRIEVAL_EXECUTION,
              "settings": settings.to_dict(), "elapsed_job_body_seconds": elapsed,
              "status": "failed" if error else "completed",
              "error": str(error) if error else None,
              "phase_seconds": dict(phases or {}),
              "work_budget_seconds": None,
              "hard_timeout_seconds": PERFORMANCE_HARD_TIMEOUT_SECONDS,
              "gpu_sampling_errors": sampler.errors,
              "whole_probe_gpu": gpu_summary(sampler.samples),
              "notes": [("One test-only full-length validation batch; no checkpoint saving or benchmark evaluation."
                         if probe.long_sequence_check else "No validation, checkpoint saving, or benchmark evaluation in this probe."),
                        "Warmup steps are excluded from step statistics; setup is included in job-body time.",
                        "GPU sample windows are approximate and can miss memory peaks.",
                        "GPU utilization is kernel-active time, not a percentage of theoretical FLOPS.",
                        "This short workload does not prove a universal safe batch limit."]}
    try:
        rows = read_native_metrics(directory)
        atomic_json(directory / "performance_steps.json", rows)
        if probe.long_sequence_check:
            report["long_sequence_check"] = {
                "normal_steps": list(range(1, probe.warmup_steps + 1)),
                "stress_steps": list(range(probe.warmup_steps + 1, settings.steps + 1)),
                "required_prompt_tokens": settings.prompt_tokens,
                "required_response_tokens": settings.response_tokens,
                "note": "Stress timings use forced full-length generation, not ordinary QA throughput. Final validation uses 16 full-length records; no checkpoint is saved.",
            }
        report["completed_steps"] = sorted({row["step"] for row in rows
            if row["metrics"].get("timing_s/step", 0) > 0})
        report["partial_step_timings"] = [{"step": row["step"],
            "timings": {k: v for k, v in row["metrics"].items() if k.startswith("timing_s/")}}
            for row in rows if row["metrics"].get("timing_s/step", 0) > 0]
        report["measurement"] = summarize_performance(
            rows, sampler.samples, probe, settings.questions_per_step * settings.responses_per_question)
        if not error and not report["measurement"]["complete_measurement"]:
            report["status"] = "incomplete_metrics"
    except Exception as metric_error:
        report["metrics_error"] = str(metric_error)
        if not error:
            report["status"] = "incomplete_metrics"
    if error:
        log = directory / "console.log"
        tail = ""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                tail = stream.read().decode("utf-8", errors="replace")
        text = (str(error) + tail).lower()
        report["failure_kind"] = "gpu_out_of_memory" if any(marker in text for marker in
            ("cuda out of memory", "cuda error: out of memory", "torch.outofmemoryerror")) else "other_error"
        if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            report["failure_kind"] = "timeout"
    events = []
    all_events = []
    event_file = directory / "search_requests.jsonl"
    if event_file.exists():
        try:
            all_events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line]
            events = [event for event in all_events if event.get("kind") in ("extended_wait", "retriever_restarted")]
        except (ValueError, OSError) as event_error:
            report["search_event_log_error"] = str(event_error)
            if report["status"] == "completed":
                report["status"] = "unverified_search_events"
    report["search_recovery_events"] = events
    completed = [event["detail"] for event in all_events if event.get("kind") == "request_completed"]
    report["search_request_summary"] = {"completed_requests": len(completed),
        "mean_rpc_wait_seconds": statistics.mean([item["rpc_wait_seconds"] for item in completed]) if completed else None,
        "backend_seconds_sum": {key: sum(item.get("backend", {}).get(key, 0) for item in completed)
                                for key in ("encoding_seconds", "index_search_seconds", "passage_read_seconds")},
        "note": "Request totals may overlap; they are not additive with trainer wall time. Includes warmup."}
    restarted = any(event.get("kind") == "retriever_restarted" for event in events)
    if restarted and report["status"] == "completed":
        report["status"] = "completed_with_warnings"
    if running:
        report["status"] = "running"
    report["comparison_eligible"] = report["status"] == "completed"
    report["snapshot_unix_time"] = time.time()
    if events:
        report["notes"].append(
            "Extended waits are included in observed performance and do not invalidate a completed run. "
            "A confirmed retriever replacement excludes the probe from normal comparison, including warmup events.")
    atomic_json(directory / "performance.json", report)
    return report
