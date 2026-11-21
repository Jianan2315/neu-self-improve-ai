"""Modal deployment for the framework notebook. Deploying does not start training."""
from pathlib import Path
import json
import re
import modal

HERE = Path(__file__).resolve().parent
APP_NAME = "week08-search-r1-framework"
app = modal.App(APP_NAME)
cache = modal.Volume.from_name("week08-search-r1-cache", create_if_missing=True)
data = modal.Volume.from_name("week08-search-r1-data", create_if_missing=True)
results = modal.Volume.from_name("week08-search-r1-framework-results", create_if_missing=True)

# The old veRL image uses Python 3.10, rejected by Modal builder 2026.10.
# This official vLLM image provides Python 3.12, CUDA 12.8 and PyTorch 2.7.1.
IMAGE_TAG = "vllm/vllm-openai:v0.10.0"
RETRIEVER_STARTUP_TIMEOUT_SECONDS = 600
# Build hooks import this file before local project modules are mounted.
# Keep decorator configuration self-contained; import runtime modules in functions.
PERFORMANCE_HARD_TIMEOUT_SECONDS = 24 * 3600
FORMAL_GPU = "H200"


def install_and_check_training_extensions():
    """Install the matching prebuilt FlashAttention wheel without GPU allocation."""
    import importlib
    from importlib.metadata import version
    import platform
    import subprocess
    import sys
    import torch

    if sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise RuntimeError("The training image requires CPython 3.12 on Linux x86_64.")
    expected = {"torch": "2.7.1", "vllm": "0.10.0", "verl": "0.6.1",
                "transformers": "4.55.4", "tensordict": "0.8.3", "peft": "0.17.1",
                "hydra-core": "1.3.2", "omegaconf": "2.3.0", "pycairo": "1.26.1"}
    for package, wanted in expected.items():
        installed = version(package).split("+")[0]
        if installed != wanted:
            raise RuntimeError(f"{package}: expected {wanted}, found {installed}.")
    if not torch.version.cuda or not torch.version.cuda.startswith("12."):
        raise RuntimeError("The FlashAttention wheel requires a CUDA 12 PyTorch build.")
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    wheel = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
             f"flash_attn-2.8.3+cu12torch2.7cxx11abi{abi}-cp312-cp312-linux_x86_64.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", wheel], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    # Check native imports during the CPU image build, before a paid training job.
    for module in ("cairo", "flash_attn", "tensordict", "peft", "verl.protocol"):
        importlib.import_module(module)
    print("Training image dependency checks passed:", expected, "FlashAttention 2.8.3")


def check_retriever_dependencies():
    """Exercise retrieval dependencies during image build without model downloads."""
    from importlib.metadata import version
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import torch
    import faiss
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer, BertConfig

    expected = {"numpy": "1.26.4", "pyarrow": "19.0.1", "pandas": "2.2.3",
                "datasets": "3.5.0", "torch": "2.6.0", "transformers": "4.51.3",
                "huggingface_hub": "0.30.2", "faiss-cpu": "1.10.0"}
    for package, wanted in expected.items():
        if version(package).split("+")[0] != wanted:
            raise RuntimeError(f"Unexpected retriever dependency version: {package}")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    # Exercise native NumPy/Arrow/FAISS boundaries, not just package metadata.
    vectors = np.eye(2, dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    _, indices = index.search(vectors[:1], 1)
    assert indices.tolist() == [[0]]
    assert torch.from_numpy(vectors).numpy().shape == (2, 2)
    assert pa.Table.from_pandas(pd.DataFrame({"id": [0]})).num_rows == 1
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "tiny.jsonl"
        source.write_text('{"contents": "retriever dependency check"}\n', encoding="utf-8")
        corpus = load_dataset("json", data_files=str(source), split="train",
                              cache_dir=str(Path(temporary) / "cache"))
        assert corpus[0]["contents"] == "retriever dependency check"
    # Resolve and run the encoder architecture with tiny random weights locally.
    encoder = AutoModel.from_config(BertConfig(vocab_size=16, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
    with torch.inference_mode():
        assert encoder(input_ids=torch.tensor([[1, 2]])).last_hidden_state.shape == (1, 2, 8)
    assert callable(AutoTokenizer.from_pretrained)
    print("Retriever dependency and CPU operation checks passed:", expected, flush=True)


def check_framework_json_export():
    """Fix and exercise the pinned framework's NumPy-to-JSON export boundary."""
    import ast
    import json
    import os
    import tempfile
    from importlib.metadata import distribution, version
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np

    if version("verl") != "0.6.1":
        raise RuntimeError("Recheck the export compatibility fix before changing veRL.")
    path = Path(distribution("verl").locate_file("verl/trainer/ppo/ray_trainer.py"))
    source = path.read_text()
    old = "json.dumps(entry, ensure_ascii=False)"
    new = "json.dumps(entry, ensure_ascii=False, default=_assignment_json_default)"
    helper = '''\n\ndef _assignment_json_default(value):
    # Preserve numeric values and nested arrays at the JSON export boundary.
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(f"Unsupported JSON export value: {type(value).__name__}")
'''
    if source.count(old) == 1 and "def _assignment_json_default(" not in source:
        source = source.replace(old, new) + helper
        path.write_text(source)
    elif source.count(new) != 1 or source.count("def _assignment_json_default(") != 1:
        raise RuntimeError("Unexpected veRL export source; refusing an unchecked patch.")

    # Execute the installed method itself on CPU, without initializing Ray or CUDA.
    tree = ast.parse(source)
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_dump_generations")
    convert = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_assignment_json_default")
    namespace = {"json": json, "os": os, "np": np}
    module = ast.fix_missing_locations(ast.Module(body=[convert, method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    with tempfile.TemporaryDirectory() as temporary:
        namespace["_dump_generations"](
            SimpleNamespace(global_steps=0), ["question"], ["<answer>answer</answer>"],
            [{"target": np.array(["answer", "alias"])}], [np.float32(1)],
            {"model_tokens": [np.int64(17)], "has_answer": [np.int64(1)],
             "question_id": [np.str_("sample_1")], "flag": [np.bool_(True)]}, temporary)
        record = json.loads((Path(temporary) / "0.jsonl").read_text())
        assert record["model_tokens"] == 17 and type(record["model_tokens"]) is int
        assert record["gts"]["target"] == ["answer", "alias"]
        assert record["flag"] is True and record["score"] == 1
    try:
        namespace["_assignment_json_default"](object())
    except TypeError:
        pass
    else:
        raise AssertionError("Unsupported values must not be silently stringified.")
    print("Installed veRL NumPy JSON export check passed; no GPU or model loaded.", flush=True)


# Legacy workspace builders invoke `python -m pip` during registry setup,
# before any chained run_commands. Use the image's existing Python environment.
REGISTRY_SETUP_COMMANDS = [
    "RUN test -x /usr/bin/python3.12 && ln -sf /usr/bin/python3.12 /usr/local/bin/python",
    "RUN python --version && python -m pip --version",
]

image = (
    modal.Image.from_registry(IMAGE_TAG, setup_dockerfile_commands=REGISTRY_SETUP_COMMANDS).entrypoint([])
    # Ubuntu's inherited PyGObject metadata requires pycairo, even for a headless job.
    # Pycairo builds against Cairo headers and the existing Python 3.12 headers.
    .apt_install("libcairo2-dev", "pkg-config")
    .pip_install_from_requirements(str(HERE / "requirements-cloud.txt"))
    .run_function(install_and_check_training_extensions)
    .run_function(check_framework_json_export)
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/root", "VLLM_USE_V1": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE / "search_r1_framework", "/root/search_r1_framework")
)
retriever_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "datasets==3.5.0",
                 "huggingface_hub==0.30.2", "numpy==1.26.4", "faiss-cpu==1.10.0",
                 "pyarrow==19.0.1", "pandas==2.2.3")
    .run_function(check_retriever_dependencies)
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "search_r1_retriever.py", "/root/search_r1_retriever.py")
    .add_local_dir(HERE / "search_r1_framework", "/root/search_r1_framework")
)


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use a short name containing letters, digits, underscores or hyphens.")
    return value


@app.function(image=image, volumes={"/data": data, "/cache": cache},
              cpu=4, memory=32768, timeout=24 * 3600)
def prepare_data(preparation_name: str, settings: dict):
    from search_r1_framework.config import Settings
    from search_r1_framework.data import prepare
    data.reload()
    try:
        return prepare(f"/data/framework/{safe_name(preparation_name)}",
                       "/data/Bamboogle Prerelease.xlsx", Settings(**settings), reuse_existing=True)
    finally:
        data.commit()
        cache.commit()


@app.function(image=retriever_image, volumes={"/data": data, "/cache": cache},
              cpu=4, memory=32768, timeout=24 * 3600)
def prepare_retrieval():
    from search_r1_retriever import prepare_assets
    data.reload()
    try:
        return prepare_assets("/data/retrieval")
    finally:
        data.commit()
        cache.commit()


@app.cls(image=retriever_image, volumes={"/data": data, "/cache": cache},
         cpu=16, memory=131072, timeout=3600, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=32)
class Retriever:
    @modal.enter()
    async def load(self):
        import asyncio
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        from search_r1_retriever import DenseRetriever
        from search_r1_framework.shared_retrieval import SharedRetrievalQueue
        from search_r1_framework.retrieval_protocol import retrieve_batch
        data.reload()
        # Load and search on the same thread: FAISS OpenMP settings are thread-local.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval")
        self.engine = await asyncio.get_running_loop().run_in_executor(
            self.executor, DenseRetriever, "/data/retrieval")
        self.instance_id = uuid.uuid4().hex
        async def search(queries, topk):
            return await asyncio.get_running_loop().run_in_executor(
                self.executor, retrieve_batch, self.engine, queries, topk)
        self.queue = SharedRetrievalQueue(search, wait_seconds=0.1, target_queries=4,
                                          max_queries=32, log=lambda s: print(s, flush=True))

    @modal.method()
    async def retrieve(self, queries: list[str], topk: int = 3):
        response = await self.queue(queries, topk)
        response["retriever_instance_id"] = self.instance_id
        return response

    @modal.method()
    async def benchmark_query_cache(self, queries: list[str], topk: int = 3, batch_size: int = 8):
        import asyncio
        # Serialize diagnostic access with production access to the mutable engine.
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self._benchmark_query_cache, queries, topk, batch_size)

    @modal.exit()
    async def close(self):
        import asyncio
        await self.queue.close()
        await asyncio.to_thread(self.executor.shutdown, wait=True)

    def _benchmark_query_cache(self, queries: list[str], topk: int = 3, batch_size: int = 8):
        """CPU-only paired replay; no training, new index, or model download."""
        import time
        from search_r1_framework.retrieval_protocol import validate_request
        validate_request(queries, topk)
        if not 1 <= len(queries) <= 256 or not 1 <= batch_size <= 32:
            raise ValueError("Use at most 256 replay queries and batch size 1..32.")
        phases = []
        reference = None
        self.engine.clear_query_cache()
        for name, enabled in (("uncached", False), ("cache_cold", True), ("cache_warm", True)):
            started = time.monotonic()
            result, metrics = [], []
            for offset in range(0, len(queries), batch_size):
                result.extend(self.engine.retrieve_many(queries[offset:offset+batch_size], topk,
                                                       use_cache=enabled))
                metrics.append(dict(self.engine.last_batch_metrics))
            elapsed = time.monotonic() - started
            ids = [[x["id"] for x in group] for group in result]
            scores = [[x["score"] for x in group] for group in result]
            if reference is None:
                reference = (ids, scores)
            same_ids = ids == reference[0]
            difference = (max((abs(a-b) for xs, ys in zip(scores, reference[1])
                               for a,b in zip(xs,ys)), default=0) if same_ids else None)
            phases.append({"phase": name, "seconds": elapsed, "metrics": metrics,
                           "same_ranked_document_ids": same_ids, "max_score_difference": difference})
            print(f"CACHE_BENCHMARK phase={name} seconds={elapsed:.3f} same_ids={same_ids}", flush=True)
        # Do not seed future training with benchmark-only cached queries.
        self.engine.clear_query_cache()
        return {"query_count": len(queries), "unique_queries": len(set(queries)),
                "batch_size": batch_size, "phases": phases,
                "retriever_instance_id": self.instance_id}


def warmup_retriever(backend, timeout=RETRIEVER_STARTUP_TIMEOUT_SECONDS):
    """Bound the wait for a cold retriever, including startup and queue time."""
    call = backend.retrieve.spawn(["Wikipedia"], 1)
    print(f"Waiting up to {timeout}s for Retriever startup; call {call.object_id}", flush=True)
    try:
        response = call.get(timeout=timeout)
    except Exception as error:
        # A client-side get timeout alone does not cancel its remote request.
        try:
            call.cancel()
        except Exception as cancel_error:
            print(f"Could not cancel retriever call {call.object_id}: {cancel_error}", flush=True)
        if isinstance(error, (TimeoutError, modal.exception.TimeoutError)):
            raise TimeoutError(
                f"Retriever was not ready within {timeout}s. Training was not started. "
                f"Inspect Retriever startup logs; request {call.object_id} was cancelled "
                "unless cancellation failed as reported above."
            ) from error
        raise
    print("Retriever startup check passed; starting the training/evaluation runner.", flush=True)
    return response


def start_search_bridge(log_path=None):
    """Private loopback HTTP bridge from veRL's existing SearchTool to Modal RPC."""
    import threading
    import time
    from search_r1_framework.search_bridge import create_search_api
    from search_r1_framework.search_batcher import SearchBatcher
    from search_r1_framework.search_wait import wait_for_search_call, RetrieverInstanceTracker
    import uvicorn
    backend = Retriever()
    # Warm the large CPU retriever before the first timed search request.
    ready = warmup_retriever(backend)
    if not isinstance(ready.get("retriever_instance_id"), str):
        raise RuntimeError("Retriever is running old code; redeploy before starting this run.")
    tracker = RetrieverInstanceTracker(ready["retriever_instance_id"])
    failures = []
    async def search(queries, topk):
        call = await backend.retrieve.spawn.aio(queries, topk)
        def notify(kind, detail):
            api.state.record_search_event(kind, {"queries": queries, "topk": topk}, detail)
        response = await wait_for_search_call(call, TimeoutError, notify)
        tracker.observe(response, notify)
        return response
    # Coalescing delay belongs only at the shared entrance, across both trainers.
    batcher = SearchBatcher(search, wait_seconds=0, max_queries=32)
    api = create_search_api(batcher, failures, log_path=log_path)
    api.add_event_handler("shutdown", batcher.close)

    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=8000,
                           log_level="warning", timeout_graceful_shutdown=5))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() >= deadline:
            raise RuntimeError("Search bridge failed to start.")
        time.sleep(0.05)
    return server, thread, failures


@app.function(image=image, volumes={"/data": data, "/cache": cache, "/results": results},
              gpu=FORMAL_GPU, cpu=8, memory=65536, timeout=24 * 3600, max_containers=2)
def run_job(job: str, run_name: str, preparation_name: str, settings: dict,
            training_run: str | None = None, resume: bool = False, performance: dict | None = None,
            smoke_dev_questions: int | None = None, interrupt_after_checkpoint: int | None = None):
    import os
    import time
    from importlib.metadata import version
    from search_r1_framework.config import Settings, overrides, write_support_configs
    from search_r1_framework.runner import (run_framework, final_actor, summarize_evaluation,
                                           CheckpointTestInterrupted)
    from search_r1_framework.checkpoints import seal_latest_checkpoint, validate_resume_request
    from search_r1_framework.data import load_bamboogle, validate_prepared
    if job not in ("train_ppo", "train_grpo", "direct", "evaluate"):
        raise ValueError("Unknown job.")
    s = Settings(**settings).validate()
    if smoke_dev_questions is not None:
        if (type(smoke_dev_questions) is not int or not 1 <= smoke_dev_questions <= s.dev_questions
                or not job.startswith("train_") or performance is not None or s.steps > 2):
            raise ValueError("Smoke validation is only for training checks of at most two steps.")
    if interrupt_after_checkpoint is not None:
        if (type(interrupt_after_checkpoint) is not int or interrupt_after_checkpoint != 1
                or s.steps != 2 or s.save_every != 1 or smoke_dev_questions != 1
                or not job.startswith("train_") or resume or performance is not None):
            raise ValueError("Interruption is only allowed in the explicit two-step recovery test.")
    probe = None
    if performance is not None:
        from search_r1_framework.performance import (PerformanceProbe, performance_overrides, GpuSampler,
            write_report, atomic_json)
        if not job.startswith("train_") or resume or training_run:
            raise ValueError("Performance probes must be fresh, separate training runs.")
        probe = PerformanceProbe(**performance)
        s = probe.settings_for(s)
        settings = s.to_dict()
    job_started = time.monotonic()
    data.reload()
    cache.reload()
    results.reload()
    prepared = Path("/data/framework") / safe_name(preparation_name)
    manifest = validate_prepared(prepared, "/data/Bamboogle Prerelease.xlsx", s)
    output = Path("/results") / safe_name(run_name)
    if output.exists() and not resume:
        raise FileExistsError("Choose a new run name, or explicitly resume the same training run.")
    if resume and (not job.startswith("train_") or (output / "finished.json").exists()):
        raise ValueError("Resume only an unfinished training run.")
    algorithm = job.removeprefix("train_") if job.startswith("train_") else "grpo"
    actor = str(prepared / "initial_actor")
    source_finished = None
    if job == "evaluate":
        if not training_run:
            raise ValueError("Select a completed training run.")
        source_finished = json.loads((Path("/results") / safe_name(training_run) / "finished.json").read_text())
        if source_finished["job"] not in ("train_ppo", "train_grpo"):
            raise ValueError("The source run did not train an actor.")
        if source_finished["preparation_name"] != preparation_name:
            raise ValueError("Training and evaluation must use the same prepared dataset and model revision.")
        actor = source_finished["adapter_path"]
        for key in ("prompt_tokens", "response_tokens", "assistant_turns", "tool_response_chars"):
            if source_finished["settings"][key] != settings[key]:
                raise ValueError(f"Keep the trained search protocol unchanged: {key}.")
    if job == "direct":
        actor = None
    output.mkdir(parents=True, exist_ok=True)
    run_info = {"job": job, "settings": settings, "preparation_name": preparation_name,
                "training_run": training_run, "data_manifest": manifest,
                "image": IMAGE_TAG, "versions": {name: version(name) for name in
                                                  ("verl", "torch", "vllm", "transformers", "peft")}}
    run_info["smoke_dev_questions"] = smoke_dev_questions
    if probe:
        run_info["performance_probe"] = probe.to_dict()
    recovery = None
    if resume:
        recovery = validate_resume_request(output, run_info)
        from search_r1_framework.checkpoints import write_json, activate_recovery
        activate_recovery(output, recovery)
        write_json(output / "resume_selection.json", recovery)
        print("RESUME_SELECTION " + json.dumps(recovery), flush=True)
    else:
        (output / "manifest.json").write_text(json.dumps(run_info, indent=2))
    arguments = overrides(s, algorithm, prepared, output, manifest["model_path"], actor,
                          str(prepared / "initial_critic"), evaluation=not job.startswith("train_"),
                          direct=job == "direct", resume=resume,
                          resume_checkpoint=recovery["path"] if recovery else None)
    # Deploying new code must not reuse stale tools.yaml from a previous preparation.
    # Resume preserves the original run's tool definitions; fresh runs use current code.
    support = output / "runtime_config"
    if resume:
        for name in ("tools.yaml", "agents.yaml"):
            if not (support / name).is_file():
                raise ValueError("Cannot resume a legacy run without its runtime tool configuration.")
    else:
        write_support_configs(support)
    if smoke_dev_questions is not None:
        from search_r1_framework.smoke import smoke_overrides
        arguments = smoke_overrides(arguments, prepared, output, smoke_dev_questions)
    if probe:
        if probe.long_sequence_check:
            from search_r1_framework.capacity import prepare_capacity_inputs
            prepare_capacity_inputs(prepared, output, s, probe)
        arguments = performance_overrides(arguments, probe)
    os.environ["TENSORBOARD_DIR"] = str(output / "tensorboard")
    server = thread = None
    failures = []
    if probe:
        sampler = GpuSampler(output, interval_seconds=0.2, sample_cpu=probe.fixed_question_indices is not None) if probe.long_sequence_check else GpuSampler(output)
        sampler.start()
        error = None
        phases = {"preparation": time.monotonic() - job_started}
        training_started = None
        def snapshot():
            current_phases = dict(phases)
            if training_started is not None:
                current_phases["trainer_elapsed"] = time.monotonic() - training_started
            report = write_report(output, probe, s, algorithm, sampler,
                                  time.monotonic() - job_started, running=True, phases=current_phases)
            report["run_name"] = run_name
            atomic_json(output / "performance.json", report)
            completed = max(report.get("completed_steps", []), default=0)
            print(f"PERFORMANCE_PROGRESS run={run_name} completed_steps={completed}/{s.steps} "
                  f"elapsed_minutes={(time.monotonic() - job_started) / 60:.1f} "
                  "(latest flushed metrics; waiting does not stop this job)", flush=True)
            results.commit()
        try:
            snapshot()
            tick = time.monotonic()
            server, thread, failures = start_search_bridge(output / "search_requests.jsonl")
            phases["retriever_startup"] = time.monotonic() - tick
            training_started = time.monotonic()
            run_framework(arguments, output, backend_failures=failures,
                          progress=snapshot)
            if failures:
                raise RuntimeError(f"Search backend errors occurred: {failures[0]}")
        except Exception as caught:
            error = caught
        finally:
            if training_started is not None:
                phases["trainer_elapsed"] = time.monotonic() - training_started
            sampler.stop()
            if server:
                server.should_exit = True
                thread.join(timeout=20)
            try:
                report = write_report(output, probe, s, algorithm, sampler,
                                      time.monotonic() - job_started, error, phases=phases)
                report["run_name"] = run_name
                report["preparation_name"] = preparation_name
                report["model_revision"] = manifest["model_revision"]
                report["training_data_sha256"] = manifest["files_sha256"]["train.parquet"]
                atomic_json(output / "performance.json", report)
            finally:
                results.commit()
                cache.commit()
        if error:
            raise error
        from search_r1_framework.performance import execution_completed
        if not execution_completed(report):
            raise RuntimeError(f"Performance report status: {report['status']}. "
                               "Inspect performance.json; incomplete or interrupted measurements cannot be compared.")
        if not report["comparison_eligible"]:
            print("Completed with a measurement warning; report saved. Excluded from clean speed ranking.", flush=True)
        # No formal-training completion marker or inference adapter is produced.
        return report
    from search_r1_framework.performance import atomic_json, read_native_metrics
    def formal_progress(status="running", error=None):
        receipt = seal_latest_checkpoint(output, algorithm) if job.startswith("train_") else None
        rows, metrics_error = [], None
        if job.startswith("train_") and (output / "tensorboard").exists():
            try:
                rows = read_native_metrics(output)
            except Exception as metrics_exception:
                # Observability must not turn a temporarily unreadable event file into a training failure.
                metrics_error = str(metrics_exception)
        completed = max((row["step"] for row in rows if "timing_s/step" in row["metrics"]), default=0)
        progress = {"run_name": run_name, "job": job, "status": status,
                    "completed_training_steps": completed,
                    "target_training_steps": s.steps if job.startswith("train_") else None,
                    "elapsed_seconds": time.monotonic() - job_started,
                    "error": str(error) if error else None,
                    "verified_checkpoint_step": receipt["step"] if receipt else None,
                    "metrics_read_error": metrics_error,
                    "note": "Latest flushed training metrics; inference progress is in console.log."}
        atomic_json(output / "progress.json", progress)
        print(f"JOB_PROGRESS run={run_name} status={status} training_steps={completed} "
              f"elapsed_minutes={progress['elapsed_seconds']/60:.1f}", flush=True)
        results.commit()

    try:
        formal_progress()
        if job != "direct":
            server, thread, failures = start_search_bridge(output / "search_requests.jsonl")
        # If training finished its final save before being interrupted, complete
        # artifact publication without accidentally running another update.
        if not recovery or recovery["remaining_steps"] > 0:
            run_framework(arguments, output, backend_failures=failures, progress=formal_progress,
                          interrupt_after_checkpoint=interrupt_after_checkpoint)
        if failures:
            raise RuntimeError(f"Search backend errors occurred; inspect console.log. First: {failures[0]}")
        finished = {"job": job, "settings": settings, "preparation_name": preparation_name,
                    "smoke_dev_questions": smoke_dev_questions}
        if job.startswith("train_"):
            finished["adapter_path"] = final_actor(output, s.steps)
            # Persist verified recovery state before publishing completion.
            # A failed seal/commit must never leave a misleading finished.json.
            seal_latest_checkpoint(output, algorithm)
            results.commit()
        else:
            ids = [r["id"] for r in load_bamboogle("/data/Bamboogle Prerelease.xlsx")]
            method = "Direct Inference" if job == "direct" else f"Search-R1 adaptation ({source_finished['job'][6:].upper()} + LoRA)"
            summary = summarize_evaluation(output, method, ids)
            summary.update({"benchmark_sha256": manifest["benchmark_sha256"],
                            "model_revision": manifest["model_revision"], "training_run": training_run})
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            finished["summary"] = summary
        (output / "finished.json").write_text(json.dumps(finished, indent=2))
        formal_progress("completed")
        return finished
    except CheckpointTestInterrupted as error:
        formal_progress("interrupted_for_resume_test", error)
        return {"status": "interrupted_for_resume_test", "run_name": run_name,
                "checkpoint_step": 1, "target_steps": s.steps}
    except Exception as error:
        formal_progress("failed", error)
        raise
    finally:
        if server:
            server.should_exit = True
            thread.join(timeout=20)
        results.commit()
        cache.commit()


@app.function(image=image, volumes={"/data": data, "/results": results},
              cpu=2, memory=8192, timeout=3600)
def check_training_request(job: str, run_name: str, preparation_name: str, settings: dict,
                           resume: bool = False, smoke_dev_questions: int | None = None):
    """CPU preflight before GPU allocation; never create or repair missing state."""
    from search_r1_framework.config import Settings
    from search_r1_framework.data import validate_prepared
    from search_r1_framework.checkpoints import validate_resume_request
    if job not in ("train_ppo", "train_grpo"):
        raise ValueError("Only PPO and GRPO training can be checked here.")
    data.reload()
    results.reload()
    s = Settings(**settings).validate()
    prepared = Path("/data/framework") / safe_name(preparation_name)
    manifest = validate_prepared(prepared, "/data/Bamboogle Prerelease.xlsx", s)
    output = Path("/results") / safe_name(run_name)
    if not resume:
        if output.exists():
            raise FileExistsError("Run already exists. Select resume, skip, or a new run name explicitly.")
        return {"action": "new", "job": job, "run_name": run_name, "passed": True}
    recovery = validate_resume_request(output, {
        "job": job, "settings": settings, "preparation_name": preparation_name,
        "data_manifest": manifest, "smoke_dev_questions": smoke_dev_questions})
    return {"action": "resume", "job": job, "run_name": run_name, "passed": True, **recovery}


@app.function(image=image, volumes={"/data": data, "/cache": cache, "/results": results},
              gpu="H100", cpu=8, memory=65536, timeout=PERFORMANCE_HARD_TIMEOUT_SECONDS, max_containers=1)
def run_performance_test(algorithm: str, run_name: str, preparation_name: str,
                         settings: dict, probe: dict):
    """Run all requested steps; only Modal's maximum function timeout applies."""
    if algorithm not in ("ppo", "grpo"):
        raise ValueError("Select PPO or GRPO explicitly.")
    return run_job.local(f"train_{algorithm}", run_name, preparation_name, settings, performance=probe)


@app.function(image=image, volumes={"/data": data, "/cache": cache, "/results": results},
              gpu="H200", cpu=8, memory=65536, timeout=PERFORMANCE_HARD_TIMEOUT_SECONDS, max_containers=1)
def run_performance_test_h200(algorithm: str, run_name: str, preparation_name: str,
                              settings: dict, probe: dict):
    """Use the identical performance path on H200 without changing formal training."""
    if algorithm not in ("ppo", "grpo"):
        raise ValueError("Select PPO or GRPO explicitly.")
    return run_job.local(f"train_{algorithm}", run_name, preparation_name, settings, performance=probe)


@app.function(image=image, volumes={"/data": data, "/cache": cache}, cpu=2, memory=8192, timeout=600)
def check_capacity_setup(preparation_name: str, base_settings: dict | None = None,
                         micro_batches: list[int] | None = None, fixed_question_indices: list[int] | None = None):
    """Real installed-framework preflight without GPU allocation or model weights."""
    import inspect
    import subprocess
    import sys
    import tempfile
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    from search_r1_framework.config import Settings, overrides, write_support_configs
    from search_r1_framework.performance import PerformanceProbe, performance_overrides
    from search_r1_framework.capacity import prepare_capacity_inputs, full_length_sampling, extend_prompt
    from search_r1_framework.capacity_agent import CapacitySearchAgent
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase
    data.reload()
    prepared = Path('/data/framework') / safe_name(preparation_name)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    base = Settings(**base_settings).validate() if base_settings is not None else Settings()
    candidates = micro_batches if micro_batches is not None else [16]
    checked = []
    # Match the installed server's max_tokens ownership, not a guessed API.
    source = inspect.getsource(vLLMHttpServerBase.generate)
    assert 'SamplingParams(max_tokens=max_tokens, **sampling_params)' in source
    assert '_handle_pending_state' in ToolAgentLoop.__dict__
    SamplingParams(max_tokens=base.response_tokens, **full_length_sampling({'temperature': 1.0}, base.response_tokens))
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    assert len(extend_prompt([1, 2], base.prompt_tokens, tokenizer.encode(' context', add_special_tokens=False)[0])) == base.prompt_tokens
    for micro_batch in candidates:
        probe = PerformanceProbe(micro_batch=micro_batch, warmup_steps=2, measured_steps=2, long_sequence_check=True,
                                 fixed_question_indices=fixed_question_indices)
        settings = probe.settings_for(base)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_support_configs(output / 'runtime_config')
            prepare_capacity_inputs(prepared, output, settings, probe)
            for algorithm in ('grpo', 'ppo'):
                arguments = overrides(settings, algorithm, prepared, output, manifest['model_path'],
                                      str(prepared / 'initial_actor'), str(prepared / 'initial_critic'))
                arguments = performance_overrides(arguments, probe)
                subprocess.run([sys.executable, '-m', 'verl.trainer.main_ppo', *arguments, '--cfg', 'job', '--resolve'],
                               capture_output=True, text=True, check=True)
                checked.append({'algorithm': algorithm, 'micro_batch': micro_batch})
    return {'passed': True, 'candidates': checked, 'prompt_tokens': base.prompt_tokens, 'response_tokens': base.response_tokens,
            'agent': CapacitySearchAgent.__name__, 'validation_questions': len(fixed_question_indices) if fixed_question_indices is not None else 16}


@app.function(image=image, volumes={"/data": data, "/cache": cache}, cpu=2, memory=8192, timeout=600)
def check_length_sample_setup(preparation_name: str):
    import subprocess
    import sys
    import tempfile
    from search_r1_framework.length_sample import prepare_sample
    from search_r1_framework.length_sample_agent import LengthSampleAgent
    data.reload()
    prepared = Path('/data/framework') / safe_name(preparation_name)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        settings, selection, arguments = prepare_sample(prepared, Path(temporary), manifest)
        subprocess.run([sys.executable, '-m', 'verl.trainer.main_ppo', *arguments, '--cfg', 'job', '--resolve'],
                       capture_output=True, text=True, check=True)
    return {'passed':True, 'selection':selection, 'settings':settings.to_dict(), 'agent':LengthSampleAgent.__name__}


@app.function(image=image, volumes={"/data": data, "/cache": cache, "/results": results},
              gpu="H100", cpu=8, memory=65536, timeout=24*3600, max_containers=1)
def run_length_sample(run_name: str, preparation_name: str):
    import os
    import time
    from search_r1_framework.length_sample import prepare_sample, summarize_sample
    from search_r1_framework.runner import run_framework
    from search_r1_framework.performance import GpuSampler, gpu_summary, atomic_json
    from search_r1_framework.data import validate_prepared
    data.reload()
    cache.reload()
    results.reload()
    prepared = Path('/data/framework') / safe_name(preparation_name)
    output = Path('/results') / safe_name(run_name)
    output.mkdir(exist_ok=False)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    settings, selection, arguments = prepare_sample(prepared, output, manifest)
    validate_prepared(prepared, '/data/Bamboogle Prerelease.xlsx', settings)
    os.environ['TENSORBOARD_DIR'] = str(output / 'tensorboard')
    sampler = GpuSampler(output, interval_seconds=0.2)
    server = thread = None
    failures = []
    started = time.monotonic()
    report = {'kind':'natural_length_sample', 'status':'running', 'selection':selection,
              'settings':settings.to_dict(), 'model_revision':manifest['model_revision'],
              'training_data_sha256':manifest['files_sha256']['train.parquet']}
    def snapshot():
        report['elapsed_seconds'] = time.monotonic()-started
        report['whole_gpu'] = gpu_summary(sampler.samples)
        atomic_json(output/'length_report.json', report)
        results.commit()
        print(f'LENGTH_SAMPLE_PROGRESS elapsed_minutes={report["elapsed_seconds"]/60:.1f}', flush=True)
    sampler.start()
    try:
        server, thread, failures = start_search_bridge(output/'search_requests.jsonl')
        run_framework(arguments, output, backend_failures=failures, progress=snapshot)
        if failures:
            raise RuntimeError(str(failures[0]))
        report.update(summarize_sample(output, selection))
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        sampler.stop()
        if server:
            server.should_exit = True
            thread.join(timeout=20)
        report['gpu_sampling_errors'] = sampler.errors
        snapshot()
        cache.commit()
    return report


@app.function(image=modal.Image.debian_slim(), volumes={"/results": results}, timeout=60)
def read_run(run_name: str, filename: str = "console.log", last_lines: int = 20):
    from collections import deque
    if filename not in ("console.log", "summary.json", "manifest.json", "finished.json", "resolved_config.yaml",
                        "performance.json", "performance_steps.json", "search_requests.jsonl", "smoke_validation.json", "progress.json"):
        raise ValueError("Unsupported output file.")
    results.reload()
    path = Path("/results") / safe_name(run_name) / filename
    if not path.exists():
        return {"status": "not_written_yet"}
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open() as stream:
        return "".join(deque(stream, maxlen=max(1, min(last_lines, 500))))
