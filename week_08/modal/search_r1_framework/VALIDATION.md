# Framework implementation: verification and scope

Updated on 2026-11-18 after the user explicitly selected veRL for both PPO and GRPO.
Packaged English entry: `../assignment_08_verl.ipynb`.
Packaging and dependency instructions: `../README.md`.
Earlier notebooks remain preserved; no TRL trainer is used by this entry point.

## What was verified locally

- All new Python modules and all notebook code cells parse.
- Twenty-one CPU-only glue tests pass (`python -m unittest search_r1_framework.test_glue -v` from this `modal` folder), including seven performance-probe checks added on 2026-11-19.
- The two algorithms share generation, search, data, and actor configuration. PPO selects GAE with a critic; GRPO disables the critic.
- Each cloud run verifies the four prepared dataset files and benchmark workbook against the preparation manifest. Resume additionally checks the saved data manifest.
- The real assignment workbook contains 125 distinct questions. Both answer columns are retained; Excel row 49 has two accepted aliases.
- Default notebook cells execute with the IPython display frontend replaced by a no-op display stub. No Modal import, download, model initialization, deployment, or training occurs in that check.
- Source inspection used the pinned veRL v0.6.1 trainer, worker, configuration, agent-loop, and reward-scoring files. This is API/source evidence, not execution evidence.

## Library ownership

veRL owns PPO/GRPO advantage estimation, policy/value losses, gradient accumulation, optimizer updates, checkpointing, multi-turn generation and response masks. vLLM owns generation and scheduling. PEFT owns adapter creation/loading. The assignment code converts data, supplies configuration, connects the search service, selects generated answer text, and invokes the existing normalized exact-match function.

`agents.py` delegates the complete rollout to the existing agent before adding outcome scoring. `compat.py` exposes the critic adapter-path field that the v0.6.1 worker reads but its config dataclass does not declare. The PPO critic is initialized through PEFT's token-classification adapter API so the scalar `score` head is trainable and saved. No worker or optimizer implementation is monkey-patched.

## Not yet verified

The corrected training image was built on Modal on 2026-11-19; see the final verification section below. No actual GPU forward/backward pass, vLLM/LoRA synchronization, native search interaction, checkpoint round-trip, or full training/evaluation run has occurred in these checks. GPU memory use, runtime compatibility, throughput, and model quality remain to be checked in the explicit two-step cloud smoke runs. No benchmark results are claimed.

The notebook and cloud entry point deliberately separate deployment, data preparation, smoke runs, formal training, and final evaluation. Formal action switches default to false; test action switches now default to true as documented below.

## Experimental differences to report

- Native Qwen tool calls rather than the paper's literal search-tag protocol.
- PEFT policy training, and PEFT plus a small trainable value head for PPO.
- A sampled 2,000-question training set and 200-question development set, balanced across NQ and HotpotQA; Bamboogle is reserved for final evaluation.
- Framework PPO/GRPO defaults with explicit KL loss, constant learning rate, 2-question batches, 16 responses per question, and 2-record micro-batches.
- 1,024 prompt tokens and 4,096 response-stream tokens, where the latter includes inserted tool text; at most five assistant turns.
- Training logs contain framework metrics rather than every full rollout; inference-only evaluation exports every benchmark response.

This is a course adaptation, not a claim of matching the paper's complete experimental setup or published scores.

## English deployment copy

Prepared on 2026-11-19. Runtime Python files and the workbook were copied without content changes. Notebook prose and displayed messages were translated to English; all saved outputs were cleared. Training settings, app names, volume names, and cloud switches were preserved. No deployment or cloud execution was performed.

## Performance-probe extension (2026-11-19)

After the initial unchanged copy, the Modal entry point was extended with an explicit one-hour performance function, `performance.py` was added, and notebook section 9A was inserted without removing existing cells. The standard Settings defaults and formal run path are preserved. Probe runs disable validation/saving and do not create a formal completion marker. They use native veRL TensorBoard step metrics and nvidia-smi samples; no alternative optimizer or training loop was introduced.

Local checks verify effective-batch preservation, warmup exclusion, incomplete metrics, missing GPU values, OOM classification, and a simulated Modal-to-trainer call. These are mocks and CPU logic checks, not measured GPU performance. No capacity limit, speedup, safe maximum micro-batch, or billing total has been established. Native metric schema and configuration were inspected against the pinned v0.6.1 source. Abrupt container termination can prevent final report persistence.


## Separate test notebook (2026-11-19)

`../assignment_08_tests.ipynb` now contains the two-step GRPO/PPO functional smoke tests and performance probes formerly embedded in the formal notebook. `../assignment_08_verl.ipynb` contains formal training and final evaluation. Both have independent setup/helper cells and reuse the same deployment modules and cloud resources. Baseline settings and preparation names must match; notebook variables and selected probe settings are not automatically shared.

The original notebook and documentation were backed up before the split. Structural checks confirmed every retained formal code cell is unchanged and every original test code cell is preserved in the test notebook. Both notebooks passed independent default execution in separate Python processes from an isolated package copy, using a display stub and explicitly rejecting imports of Modal/model-runtime libraries. All cloud switches remained false (11 formal, 13 test); default formal settings remained 2,000 steps and micro-batch 2. English cell text, Python syntax, and local notebook links were checked. No deployment or cloud test was run. Runtime Python modules were not changed by this separation.


## Test notebook Run All defaults (2026-11-19)

At the user's request, both test notebooks now enable all 13 action switches. Earlier default-execution checks above describe the previous disabled-switch versions and no longer mean these notebooks are safe to execute without cloud work. The formal notebook is unchanged. Local AST comparison verified that only the 13 action assignments changed executable behavior; resume=False and all wait calls are retained. Exceptions are not caught to continue execution, and existing-resource collision errors are preserved. No notebook cells or cloud jobs were executed for this edit.


## Explicit sequential micro-batch sweep (2026-11-19)

Both test notebooks now run the user-requested configuration comparison: GRPO then PPO, each at micro-batches 2, 4, 8, 16, 32. Waiting is mandatory within the submission loop, replacing the separate optional performance-wait switch. Up to 10 performance jobs run, each retaining the existing 2 warmup + 8 measured updates. Existing deployment/runtime Python code and formal training notebook are unchanged. The notebook does not skip failed candidates, retry jobs, or choose formal settings.

Local simulated calls verified the complete 10-job order and strictly alternating submit/wait events. A simulated third-job exception and a third-job incomplete report both prevented a fourth submission. A complete simulated report set passed separate per-algorithm comparisons. English/Chinese executable ASTs match. These are control-flow checks only: no real cloud jobs or GPU performance measurements were executed.

## Modal image compatibility correction (2026-11-19)

Observed deployment failure: the previous veRL registry image used Python 3.10, rejected by Modal image builder 2026.10. Its tensordict 0.6.2 also violated veRL 0.6.1 requirements. No GPU training had begun in this failed deployment.

The training base is now the official vllm/vllm-openai:v0.10.0 image. Docker Hub image metadata was inspected and confirmed Python 3.12 / CUDA 12.8.1. vLLM 0.10.0 pins PyTorch 2.7.1. requirements-cloud.txt explicitly pins the training stack, tensordict 0.8.3, Hydra 1.3.2 and OmegaConf 2.3.0. veRL is installed with dependency resolution. An image-build function installs an official FlashAttention 2.8.3 CPython 3.12 / torch 2.7 / CUDA 12 wheel selected for the actual PyTorch C++ ABI, then runs pip check and imports critical modules. Both ABI wheel assets were verified in the upstream GitHub release. No GPU is requested for this build function.

Verification: Linux / Python 3.12 pip dry-run dependency resolution passed (no packages installed into the user's environment); its report is stored outside the deployment folder under ../diagnostics/cloud_dependency_plan.json. This fresh-environment resolver check is not an installed-image or CUDA binary compatibility test. Python syntax checks and simulated build-function checks passed for both ABI branches and rejection of Python 3.10. All 21 existing CPU glue tests passed. The new image has NOT been built on Modal and has NOT passed a real GPU smoke test.

Notebook code and saved outputs, training hyperparameters, data and cloud resource names were left unchanged. Backups of changed pre-existing files are under ../.revision_backups/modal_image_fix_20261119_134300/.

## Python executable alias correction (2026-11-19)

The next real deployment failed before dependency installation: Modal generated `python -m pip install -r /.requirements.txt`, but the official vLLM image only provided versioned Python commands. The saved notebook and cloud screenshot both show `/bin/sh: 1: python: not found` and exit status 127. This supersedes any assumption that the previous metadata checks covered the unversioned executable.

Before pip_install_from_requirements, the image now checks /usr/bin/python3.12 and creates /usr/local/bin/python as a symbolic link to it, then checks `python --version` and `python -m pip --version`. This reuses the existing interpreter and installed packages; it does not install another Python environment. Python syntax and build-step ordering were checked locally. No new Modal deployment or GPU execution was performed. Notebook outputs and requirements-cloud.txt were preserved. The earlier 21 CPU tests and dependency resolver checks do not validate this cloud build step.

## Actual CPU image verification and Cairo repair (2026-11-19)

This section supersedes the earlier dated statements that the corrected image had not been built. The latest failed build installed FlashAttention successfully, then stopped at `pip check`: `pygobject 3.42.1 requires pycairo, which is not installed`. The CPU audit traced PyGObject to the base image's Ubuntu package `python3-gi`, under `/usr/lib/python3/dist-packages`. A fresh dependency resolver dry-run did not cover this inherited system-package requirement.

The deployment image now installs `libcairo2-dev` and `pkg-config`, and requirements-cloud.txt pins `pycairo==1.26.1`. Build checks assert that version and import `cairo`. Strict `pip check` remains enabled. Training settings and notebook outputs were not changed.

With explicit user authorization, a temporary Modal CPU sandbox used the successful dependency-install layer from the failed deployment, applied the repair, installed the matching FlashAttention wheel, and passed `pip check`. Separate import checks passed for all 11 modules: cairo, torch, tensordict, flash_attn, peft, vllm, verl.protocol, verl.trainer.main_ppo, verl.workers.fsdp_workers, verl.experimental.agent_loop.tool_agent_loop, and verl.tools.search_tool. The log records `IMPORT_FAILURES []`, `SANDBOX_EXIT 0`, and `SANDBOX_TERMINATED`.

The corrected deployment script's training image was then built using Modal Image.build(). Its version checks, strict dependency check, and five build-time imports passed. The final success marker is `VERIFIED_TRAINING_IMAGE im-nLe2QOuKgl6zVjHPbj70hL`. This is an actual cloud image build, not a simulated check. It does not establish GPU execution or successful full-app deployment.

Evidence retained outside the deployment package:

- `../../diagnostics/cloud_image_audit_20261119.log`: sandbox diagnostics and 11 imports.
- `../../diagnostics/final_image_build_20261119.log`: final image build and dependency checks.
- `../../diagnostics/audit_cloud_image.py` and `../../diagnostics/build_training_image.py`: diagnostic entry points; these invoke real cloud resources and are not required for notebook deployment.

No GPU was allocated, model weights downloaded, or training/evaluation run by these checks. Pre-edit backups are under `../../.revision_backups/modal_cairo_fix_20261119_142212/`.

## Retriever startup repair and real CPU image check (2026-11-19)

The user's subsequent deployment succeeded and data preparation completed. The GRPO smoke call then waited for Retriever startup. Cloud logs show repeated `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` in `DenseRetriever.__init__`, while importing datasets. The separately configured retrieval image pinned NumPy but left PyArrow unpinned; the earlier successful training-image check did not cover this environment. Logs are retained in `../../diagnostics/retriever_failure_20261119.log`.

The retrieval image now pins PyArrow 19.0.1 and Pandas 2.2.3 alongside the existing NumPy 1.26.4. Its new CPU build function verifies all eight direct dependency versions, runs pip check, loads a one-row local JSON dataset, converts a Pandas table to Arrow, executes FAISS search and NumPy/Torch interchange, and runs a tiny randomly initialized BERT encoder. It does not download pretrained weights or access the full corpus/index.

Actual Modal build passed: `VERIFIED_RETRIEVER_IMAGE im-V6pdqr7U5AO3UZQTJlxGA7`, with `No broken requirements found` and `Retriever dependency and CPU operation checks passed`. Evidence: `../../diagnostics/retriever_image_build_20261119.log`; reproducible CPU-only build entry: `../../diagnostics/build_retriever_image.py`. The complete application was not redeployed by this repair, and no GPU task was started. Full-index loading and an end-to-end search/training run remain unverified.

Retriever warmup now uses spawn/get with a 600-second wait limit, including startup and queue time. Timeout cancels that request and raises before the framework runner starts; other errors propagate. Cancellation failures are printed rather than hidden. The setting is an operational deadline, not a measured cold-start duration. Four isolated tests passed for readiness, timeout cancellation, preservation of remote errors, and cancellation failure. These control-flow tests use simulated calls, not real failed services: `../../diagnostics/test_retriever_startup.py`. The production behavior uses Modal's [documented FunctionCall get/cancel interfaces](https://modal.com/docs/sdk/py/latest/FunctionCall); the dependency choice is consistent with [PyArrow 19 installation guidance](https://arrow.apache.org/docs/19.0/python/install.html) and was verified in the actual image.

The previous GRPO call `fc-01M7V5GXEKMD8FA1Q4XW7CYFD4` and its Retriever child were both observed as TERMINATED after the user stopped the application. This repair did not cancel them. Notebook cells/outputs, datasets, model settings, and existing run directories were preserved. A new smoke attempt should use a fresh run name instead of overwriting `grpo-smoke-v1` or claiming to resume trained weights. Pre-edit backups are in `../../.revision_backups/retriever_fix_20261119_1/`.

## LoRA target configuration type repair (2026-11-19)

The user's next smoke run (`grpo-smoke-v2`) completed Retriever warmup, then failed during Actor rollout initialization. The real error was `Cannot convert 'ListConfig' to string`, field `target_modules`, in veRL's HFModelConfig structured conversion. The parent call was FAILURE and its Retriever child was SUCCESS. Evidence: `../../diagnostics/grpo_v2_failure_20261119.log`.

The pinned [veRL 0.6.1 HFModelConfig](https://github.com/verl-project/verl/blob/v0.6.1/verl/workers/config/model.py) declares target_modules as Optional[str]. Both Actor and Critic CLI overrides now use the equivalent PEFT regular-expression string `(?:.*[.])?(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`. The `[.]` spelling avoids changing the expression through Hydra's quoted CLI parsing. The original TARGET_MODULES list remains unchanged for adapter initialization; existing prepared adapters and datasets do not need to be regenerated. No framework code or installed package versions were changed.

Verification:

- All 22 local CPU glue tests passed, including positive/negative checks that the expression preserves exact-name and dotted-suffix selection.
- An actual temporary CPU sandbox reused the verified training image. For both GRPO and PPO, it reproduced the original list-to-string failure as a regression control, then passed Hydra composition, HFModelConfig conversion, Critic construction through its normal `_target_` path, and Rollout configuration construction using the corrected overrides.
- Tiny randomly initialized Qwen models confirmed that PEFT list and expression configurations select the same seven layers for CAUSAL_LM and TOKEN_CLS. Locally saved list-based test adapters reloaded successfully. This verifies compatibility of the format, not the user's full-size cloud adapter files.
- The final log records `TARGET_MODULES_CPU_CHECK_PASSED`, `SANDBOX_EXIT 0`, and `SANDBOX_TERMINATED`: `../../diagnostics/target_modules_cpu_20261119.log`. Reproduction script: `../../diagnostics/check_target_modules_cpu.py`. Earlier diagnostic attempts are retained separately; the final check corrected test-only offline setup and Critic-construction routing, and caught the CLI backslash issue before this final pass.

The Chinese test notebook now submits and reads `grpo-smoke-v3`; reuse settings for workbook/data preparation remain unchanged. Its saved outputs are preserved, including the failed v2 evidence. No new GPU training or application deployment was performed. Source inspection and CPU checks do not establish a successful full-size GPU training run. Pre-edit files are in `../../.revision_backups/target_modules_fix_20261119/`.

## vLLM CUDA Graph CLI repair (2026-11-20)

The user's `grpo-smoke-v3` run loaded the Actor model and initialized LoRA, then failed in the vLLM server argument parser: `argument --cuda-graph-sizes: invalid int value: '[1, 2, 4, 8, 16, 32]'`. Ray's worker-exit error and Modal's oversized CalledProcessError were secondary wrappers. The stopped application's log is retained in `../../diagnostics/grpo_v3_failure_20261120.log`.

The pinned [veRL launcher](https://github.com/verl-project/verl/blob/v0.6.1/verl/workers/rollout/vllm_rollout/vllm_async_server.py) converts each CLI value with `str(value)`, which incorrectly serializes this list for vLLM's integer-list option. The assignment overrides now set `cudagraph_capture_sizes=null` and pass the same six sizes as a JSON string through `engine_kwargs.vllm.compilation_config`, supported by [vLLM 0.10.0](https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/config.py). CUDA Graph remains enabled; model, training, memory and batch settings remain unchanged. No library source or dependency versions were changed.

The runner retains the complete console log and raises a bounded RuntimeError containing relevant error lines instead of serializing the entire training command. A simulated subprocess regression checks that the underlying parser error remains visible and the complete log is preserved. All 24 local CPU glue tests pass.

Actual temporary Modal CPU verification passed in the existing training image: the test reproduced the original parser failure, then exercised the installed veRL method's argument-building/parser prefix and vLLM AsyncEngineArgs construction for GRPO, PPO, direct inference and search evaluation. It verified the six capture sizes, LoRA selection, memory fraction, sequence limit, token batch limit and context length. The test explicitly supplies CPU platform discovery because the CUDA wheel cannot detect a GPU in this sandbox; it stops before distributed workers and engine initialization. Earlier diagnostic-only failures (frozen test configuration, device discovery and a None-versus-False assertion) are retained separately. This is real CLI compatibility verification, not GPU engine or training verification.

Evidence: `../../diagnostics/check_vllm_cli_cpu.py` and `../../diagnostics/vllm_cli_cpu_20261120.log`, recording `VLLM_CLI_CPU_CHECK_PASSED`, `SANDBOX_EXIT 0` and `SANDBOX_TERMINATED`. No GPU was allocated or training started by this repair. The Chinese notebook now submits and reads `grpo-smoke-v4`, preserving all existing outputs and metadata. Backups: `../../.revision_backups/vllm_cli_fix_20261120/`. Reopen the saved notebook before Restart Kernel / Run All so an old editor buffer does not restore the previous retry name.

## Search contract and integrated failure handling (2026-11-20)

The v4 cloud log confirms that the full retriever index loaded and vLLM reached pre-training validation. It does not establish any completed optimizer update. Its repeated backend HTTP errors were 500, not 400. The backend exception rejected requests outside an extra one-query restriction; the complete rejected payload was not logged, so its exact contents cannot be recovered. Separate model calls supplied a string instead of query_list, which the native tool rejected before HTTP. These are distinct failures.

Changes:

- Keep the pinned veRL, vLLM, native SearchTool, PPO/GRPO trainers, and dependency versions. Accept the native nonempty list-of-strings request contract, including multiple queries, with one ordered result group per query. Validate both requests and results at the bridge and backend; do not silently coerce malformed types.
- Remove ineffective schema fields that veRL 0.6.1 drops (`items`, `minItems`, `maxItems`). Put an explicit array example in the retained description. Runtime validation enforces element types. Malformed model requests receive HTTP 400 without backend invocation; the installed veRL client does not retry them. Infrastructure failures or invalid backend responses set a fatal flag, log request/error details, and stop the trainer instead of waiting for all tool retries. Remote search calls have a bounded wait and explicit cancellation on failure.
- Poll that fatal flag while streaming trainer logs, including silent periods. Stop the subprocess group on Linux and retain console.log plus search_requests.jsonl. This does not stop the entire separately deployed Modal application.
- Write current tools.yaml and agents.yaml into each fresh run's runtime_config directory. Reuse prepared data/model/adapters without reusing stale tool configuration. Resume requires the original run's saved runtime configuration; legacy failed runs without it must start fresh.
- Both test notebooks use four development questions for smoke validation, skip pre-training validation, and keep two updates plus per-step validation/save. The subset is copied into the run directory with its question IDs recorded; the prepared data, formal Settings, effective training batch, benchmark, and performance sweep are unchanged. New run names: grpo-smoke-v5 and ppo-smoke-v2. Saved notebook outputs are retained as historical evidence.

Verification: 28 local CPU tests passed. In the actual training image, a temporary CPU sandbox passed all 32 regression tests, the installed veRL schema round trip, real loopback HTTP requests and native result formatting for multiple queries, rejection of invalid elements without backend retries, and Hydra composition for GRPO/PPO/direct/search evaluation. It also verified the four-question subset without changing the source development file. The HTTP test substitutes the retrieval engine; it does not exercise a live Modal Retriever RPC or the Wikipedia index. The expected deliberate HTTP 400 error appears in the otherwise successful diagnostic log. Evidence: `../../diagnostics/search_contract_cpu_20261120.log` and its `check_search_contract_cpu.py` entry point. Sandbox exit was 0 and it was terminated. No GPU was allocated, model downloaded, or training started for this repair.

Adjacent source audit: veRL 0.6.1 collect_lora_params supports both values of layered_summon, and the worker passes the resulting parameters into rollout.update_weights. Inspection did not establish that the current setting causes a synchronization failure, so no speculative weight-sync change was made. Actual GPU update, weight synchronization, checkpoint save/reload, and a complete multi-turn training run remain unverified. The CPU checks must not be presented as end-to-end training success.

Sources: [native search HTTP protocol](https://github.com/verl-project/verl/blob/v0.6.1/verl/tools/utils/search_r1_like_utils.py), [tool schema](https://github.com/verl-project/verl/blob/v0.6.1/verl/tools/schemas.py), [LoRA parameter collection](https://github.com/verl-project/verl/blob/v0.6.1/verl/utils/fsdp_utils.py), [worker weight synchronization](https://github.com/verl-project/verl/blob/v0.6.1/verl/workers/fsdp_workers.py).

Backups: `../../.revision_backups/search_contract_fix_20261120/modal/`. Reload the notebooks from disk before restarting the kernel. Run All launches paid smoke jobs followed by the existing sequential performance sweep; this edit has not launched those jobs or redeployed the full app.

## Bounded recovery after retriever preemption (2026-11-20)

Observed user-run evidence now supersedes the earlier absence of GPU smoke results: the Chinese notebook contains successful GRPO v5 and PPO v2 completion records, each with two steps and a final adapter. The first performance probe, perf-grpo-mb2-v1, logged five completed steps before failing. Modal logs explicitly report retriever preemption at 07:58:08 EST, replacement readiness at 07:59:14, and the bridge timeout at 07:59:53. No later performance group was submitted. This is not the previous string/list contract failure.

The recovery policy now waits for the same spawned FunctionCall, polling results every 30 seconds within a 600-second total result budget. Poll timeouts do not resubmit or cancel the call. Final expiry, cancellation, or a returned backend exception cancels that call once, with a ten-second cleanup bound. The bridge timeout is 620 seconds and the native SearchTool HTTP timeout is 650 seconds. These are bounded initial budgets, not guarantees that every interruption will recover. A pending result longer than the former 175-second budget is logged separately from an actual container replacement.

The actual SDK CPU check exposed an important distinction: `FunctionCall.get(timeout=...)` raises Python's built-in TimeoutError when no result is available, not modal.exception.TimeoutError. The application now handles the actual exception. A TimeoutError raised inside the retrieval engine is wrapped as a RuntimeError at the RPC boundary so it cannot be mistaken for a pending result. Other backend exceptions still fail promptly when returned. Startup failures that produce no result remain bounded by the overall wait.

Each loaded Retriever creates a new instance ID and includes it in successful responses. The bridge compares that ID against its warmup result and logs replacement events. Performance reports retain raw metrics, but any observed replacement or extended wait marks an otherwise completed probe `interrupted_measurement`, with comparison_eligible=false. The whole probe is conservatively excluded even if the event occurred in warmup. Long waiting alone is not proof of preemption. The sweep stops for review rather than silently using that result or automatically rerunning it.

Both test notebooks now reuse matching successful smoke completion records, validating algorithm, settings, preparation name, validation count and final adapter path. They do not rerun the previously successful small training jobs during Run All. Performance names use series v2. Old outputs, the failed v1 run, prepared data and the formal notebook remain unchanged. The cached completion check trusts the prior successful completion record; it does not reload adapter weights. Resume and overwrite behavior were not loosened.

Verification: 39 local regression tests and four startup checks passed. Actual temporary CPU Modal execution verified repeated waits followed by success for one real FunctionCall, with one submission. The existing training image passed all 43 regression tests, schema/HTTP integration checks and four-mode Hydra composition. Virtual-clock tests cover expiry at 600 seconds, cancellation, immediate backend errors and performance exclusion; no real preemption was induced. Both notebook smoke-reuse paths were exercised with saved-result fixtures and forbidden submission stubs. All source cells parse and historical outputs are unchanged.

Evidence: `../../diagnostics/check_timeout_recovery_cpu.py` and `../../diagnostics/timeout_recovery_cpu_20261120_final.log` (`REAL_MODAL_SAME_CALL_POLLING_PASSED`, `CPU_REGRESSION_TESTS_PASSED 43`, `SANDBOX_EXIT 0`, `SANDBOX_TERMINATED`). Earlier diagnostic logs are retained: the first caught the SDK exception mismatch; the second exposed missing imports in the temporary diagnostic function's packaging. The diagnostic function was then serialized independently and the final run passed. Those temporary apps ended; no GPU, model download, formal app deployment or training was started for this repair.

Backup: `../../.revision_backups/search_timeout_recovery_20261120/modal/`. Reload the test notebook from disk, restart its kernel and Run All to deploy the new runtime, reuse matching smoke records and start the v2 performance sweep. An existing failed v2 name will still be rejected rather than overwritten. GPU behavior of this newest recovery policy remains unverified.


## 2026-11-20: batched exact retrieval and performance deadlines

- Replaced serial per-query encoding/search with bounded batches (32 unique queries), restoring request order and duplicate results. Same corpus/index/encoder/pooling/top-k. Added encoding/index/passage/RPC timing.
- Retained the 3600-second cloud cap; introduced a 3480-second monotonic work deadline and bounded HTTP shutdown. A deadline stops the trainer process group and writes a failed report. Approximately every 60 seconds, flushed metrics and a non-comparable running report are committed. Hard interruption still cannot guarantee the most recent interval.
- Preserved notebook outputs and formal notebook bytes. Test notebooks use v3, display partial reports on failure, and require the same retrieval-execution revision when comparing completed runs.
- Local WSL CPU regression suite: 46 tests passed. Real PyTorch and FAISS 1.10.0, deterministic test encoder and synthetic 101-vector index; 37 ordered queries including repeats, a 32/3 unique-query chunk boundary, empty index, matching top-k IDs and scores within 2e-6. No downloaded E5 model or full corpus was tested. Tests also exercise silent-child termination, partial report preservation, HTTP cancellation, protocol validation and deployment orchestration with simulated services.
- Syntax checked all Python files and notebook code cells. No new deployment, cloud job, GPU training or full-index throughput measurement. Speedup is not established by correctness tests. Existing dependency versions unchanged; test-only FAISS/openpyxl installed in /tmp/week08-batch-check-deps, not the user's virtual environment.
- Design references: https://github.com/PeterGriffinJin/Search-R1/issues/54 ; https://github.com/facebookresearch/faiss/wiki/FAQ#why-are-searches-with-a-single-query-vector-slow ; https://modal.com/docs/guide/volumes .


## 2026-11-20: supersedes the 58-minute budget above

User explicitly rejected the one-hour cost guard. Removed the runner work deadline and set the performance function to Modal's 86400-second maximum. Retained periodic snapshots and added completed/target step and elapsed-minute log messages. Extended waiting alone no longer invalidates completion; confirmed retriever replacement remains excluded from normal comparison. Added a simulated-clock regression crossing one hour, and retained real-failure reporting and batch-result tests. Cloud execution and full-index speedup remain unverified; no deployment or training started.


## 2026-11-20: fresh-account image build import failure

Observed image im-6cJWuK7lsv25vGJguPGIoO failed before check_retriever_dependencies executed: deployment module line 6 imported search_r1_framework.performance before add_local_dir made that package available. Removed that top-level import and kept the 86400-second decorator constant self-contained. Runtime imports remain inside functions. Both build hooks share this entrypoint.

Verification: 25 local tests passed (24 glue tests plus isolated deployment import). The isolated process blocks project packages, torch, verl and faiss, with Modal mocked; the pre-fix file fails this exact check and the repaired file passes. This verifies import ordering, not a real cloud image build. No deployment or cloud compute started by this repair.


## 2026-11-20: completed work with warnings and cross-request batching

Observed cloud evidence before this change: perf-grpo-mb2-v3 completed all ten updates, then the wrapper raised interrupted_measurement after a recovered Retriever preemption. Its saved error field was null and complete_measurement was true. That historical report remains unchanged.

- Cloud completion now requires every expected step and complete metrics. Recovered preemption produces completed_with_warnings, returns the saved report and permits subsequent candidates. Real worker errors and missing steps still raise. Legacy interrupted_measurement reports can be recognized as completed only if the same complete-step checks pass. Warning reports are excluded from clean speed rankings.
- SearchBatcher combines concurrent HTTP requests before Modal RPC: 50 ms collection window, target cap 32 queries, one in-flight RPC. Different top-k requests remain separate; result order and duplicates are restored. DenseRetriever still chunks a larger individual request. Backend timings are assigned to one response per merged batch to avoid double-counting. Search batch sizes appear in request logs. Both formal training and performance tests use the shared bridge; no model, corpus, index, package version or training batch setting changed.
- Both test notebooks screen GRPO/PPO micro-batches 2/4/8, two warmup plus two measured steps (24 updates maximum). v4 separates the changed retrieval revision. Matching completed v4 reports, including warnings, are reused. Incomplete, failed or mismatched existing runs stop without overwrite/retry. v3 is read as historical evidence, never included in a strict v4 speedup calculation. Very short screening is noisy and does not establish an optimum.
- Verification: 54 local WSL CPU tests passed, including actual FAISS synthetic-index tests, async request merging/order/top-k/cancellation/failure propagation, real wrapper orchestration with simulated workers, and both notebook plans executed against simulated Modal calls. Warning reports continue the sweep, second execution submits no duplicate jobs, and a real failure stops later submissions. Python/notebook syntax checked. Notebook outputs were preserved; formal notebook bytes remain unchanged.
- No deployment or GPU job was started. Full-index speedup and the new cloud path are not yet measured. Backup: ../../.revision_backups/coalesced_search_20261120_115958/modal/.


## 2026-11-20: Run All reuse after the saved workbook error

The user-saved Chinese notebook confirmed FileExistsError at the workbook upload cell, before training. The v4 performance parameters were already present; the missed repeated-setup path caused the new failure. Preserved the latest error output separately in ../../.revision_backups/resume_runall_20261120_120851/saved_error_zh.ipynb and retained cell outputs in the edited notebook.

Both test notebooks now compare cloud workbook bytes and skip identical uploads. prepare_data validates existing dataset hashes, settings, model path and initial adapter files, returning without download/model initialization when reusable. Different or incomplete artifacts fail without overwrite. Retrieval preparation already reuses its manifest and assets. Run switches remain True. The single complete v4 plan runs remaining GRPO/PPO 2/4/8 candidates sequentially, then summarizes; no automatic extra round is scheduled. Existing completed reports are reused, and v3 is historical reference only.

Read-only cloud checks confirmed matching finished.json files for grpo-smoke-v5 and ppo-smoke-v2. Both will be reused under unchanged settings. 57 local CPU tests passed, including notebook upload wiring and repeat-preparation checks. No deployment, paid GPU test or training was started by this repair. Cloud speed and cost of the revised pipeline remain unmeasured.


## 2026-11-20: bounded optimization notebook

Added `assignment_08_optimization_tests_zh.ipynb` and `run_optimization_tests.py` for a user-run extension of the completed v4 probes. It redeploys the stopped app, reads existing micro-batch 8 baselines, replays fixed recorded queries at batch sizes 8/16/32, then tests GRPO/PPO micro-batches 16/32. It does not prepare data, repeat smoke tests, or start formal training. Existing notebooks and outputs were left unchanged.

The user explicitly authorized at most USD 14 and one hour for this experiment. The default controller allows 55 minutes including deployment and reserves cleanup time. The estimated resource charge includes a 50% reserve; this is not an account-level billing cap. A fixed output directory prevents accidental paid resubmission. The restarted app is stopped in cleanup; cleanup failure is reported explicitly. Optional absolute deadlines now propagate only from performance tests into retriever warmup and the veRL subprocess runner; default formal-training deadlines are unchanged. Reports retain partial results.

Validation: 23 local CPU/simulated tests passed (optimization session, search batcher, build import, search wait). Included full four-candidate ordering, OOM skipping, duplicate-run prevention, budget refusal, and cloud-runner process-group termination at the deadline. Modified Python and Notebook cells parsed successfully. A broader host-side run encountered missing PyTorch in the host runtime for an existing FAISS test; no dependencies were installed to rerun that unrelated test. No new cloud deployment, GPU training, or retrieval benchmark was executed for this change. The earlier read-only/preflight attempt found the stopped app and submitted no compute calls.


## 2026-11-20: correct pending-poll exception and noninteractive cleanup

Observed saved Notebook failure at 19:30:18 UTC: deployment succeeded in 2.349 seconds, then the first Retriever warmup result poll raised builtin TimeoutError after 30 seconds. The new controller incorrectly caught modal.exception.TimeoutError, so it cancelled a pending cold start. Cleanup also omitted --yes and the Modal CLI refused a noninteractive stop. No GPU training or measured retrieval round had started. This is a regression in the new orchestration, not evidence of a new dependency or veRL failure. The previous mock incorrectly aliased the two exception types and did not exercise pending polls.

Verified the user's actual WSL Modal 1.6.4 source: _Invocation.poll_function raises builtin TimeoutError for an empty unfinished poll; modal.exception.TimeoutError is a distinct type. Added poll_function_call to the existing search_wait module and used it in the controller. Pending polls keep the same call; remote execution failures propagate; the global deadline remains effective. Added --yes to cleanup. The Notebook reloads the wait module before reloading the controller.

Validation: all 65 local WSL CPU tests passed, including actual installed SDK polling code with simulated responses (no network/remote compute), three pending polls followed by success, remote execution timeout propagation, deadline termination, and the cleanup command. The previously failed app-stop operation was completed successfully; a live container listing returned []. No new paid tests were submitted. Archived the failed local results as optimization_results_failed_20261120_193018 and preserved Notebook outputs and pre-fix files under .revision_backups/optimization_poll_fix_20261120.


## 2026-11-20: user removed experiment time and cost cutoffs

Supersedes the bounded-session policy above. Removed the controller's --minutes/--budget-usd options, 55-minute deadline, cost rejection, remaining-time checks, and candidate skipping based on time. Removed experiment deadline parameters/checks from the cloud entrypoints and veRL runner. Result polling now continues on the same call until success, an actual error, or user interruption. The hourly resource estimate is informational only. Existing query/transport fault timeouts, Modal-required function timeouts, OOM handling, duplicate-run protection, and completion/error cleanup remain. Notebook source text and calls updated; old saved outputs preserved as historical evidence.

All 65 local WSL CPU regression tests passed, including 121 consecutive simulated 30-second pending polls without cancellation, a high cost estimate that does not block any candidate, and an uncapped subprocess launch. Modified source files and Notebook cells parsed successfully; static checks found no experiment deadline or budget/time arguments in the active controller and Notebook code. No cloud deployment or new paid test was started. Backup: ../.revision_backups/remove_optimization_limits_20261120.


## 2026-11-20: exact cross-request query cache and actual CPU replay

Added a per-loaded-retriever LRU cache of at most 2048 exact (query text, top-k) keys. The fixed encoder, corpus, index, pooling, and search algorithm remain unchanged. Cache misses use the existing batched exact search; hits reuse ranked documents. Results are copied before returning. Different text or top-k remains a miss, failures are not cached, and container replacement starts an empty cache. Metrics expose cache hits and actual searched unique queries. Retrieval revision is now coalesced-exact-cache-v3, so historical pre-cache performance reports cannot be silently reused as equivalent results. Training settings and notebook outputs were not changed.

Validation: all 65 local WSL CPU regression tests passed, including real FAISS synthetic-index tests for ordering, duplicate queries, result mutation isolation, different top-k/text, LRU eviction, and uncached exceptions. Modified Python modules parse successfully.

An actual Modal CPU-only replay used the existing E5 encoder and full Wikipedia index, with 32 saved queries (22 unique), top-k 3, and request groups of eight. Uncached: 21.5600 seconds, 26 searched queries across groups, zero cache hits. Initially empty cache: 20.6880 seconds, 22 searched queries, six hit occurrences. Replaying the same queries with a populated cache: 0.000517 seconds inside the retriever, zero index searches, 32 hits. All ranked document IDs matched; score differences are recorded in the report. Timing excludes network round trips and container startup. The populated-cache phase is an all-repeat case, not an end-to-end training speedup estimate. The approximately 4% difference in the first pass is a single short observation, not a robust speedup guarantee.

No GPU or training job was started. Deployment succeeded and the application was stopped after the replay. Evidence: ../../diagnostics/cache_replay_20261120/report.json, call.json, cleanup.json. Backup: ../../.revision_backups/exact_retrieval_cache_20261120/. Cache behavior during full GPU training has not been measured.


## 2026-11-20: actual micro-batch-16 long-sequence capacity failure

User authorized a targeted GPU capacity check. Added an explicit PerformanceProbe.long_sequence_check option (default false), test-only dataset/agent helpers, and actual installed-framework CPU preflight. Formal Settings and formal Notebook remain unchanged. Test-only prompt filler uses ordinary attended tokens to reach 1024 prompt tokens; native vLLM min_tokens=4096 and ignore_eos force real full-length generation. No max_tokens argument is duplicated: installed veRL owns that argument. Native veRL actor/critic updates remain unchanged. Normal probes retain their existing two-second GPU sampler; this capacity probe samples every 0.2 seconds. Stress timings must not be described as ordinary QA throughput.

68 local WSL CPU tests passed. Actual cloud CPU preflight passed both algorithm Hydra compositions and the installed vLLM sampling API. GRPO completed two normal steps, then generated 1024+4096-token records and failed in the first full-length actor loss.backward with CUDA OOM. The failed allocation requested 23.18 GiB with 13.07 GiB free. The 0.2-second device samples peaked at 72,888 MiB (71.1797 GiB); sampling can miss instantaneous peaks. Normal step 1: actor update 3.7372 s, whole step 135.3948 s. Normal step 2: actor update 3.0646 s, whole step 99.1818 s. Both reported Torch reserved peak 50.9824 GiB. These are startup/warmup steps on different questions, not a controlled speed comparison against earlier runs. No complete full-length update or step duration exists.

The check intentionally stopped on this real capacity failure. PPO was not submitted, final validation was not reached, and no retry or smaller-batch experiment was started. Thus micro-batch 16 is NOT verified safe at the configured length boundary. No inference that 8 is already safe is justified. No trained adapter/checkpoint from this diagnostic is used for formal work. The app was stopped successfully; total local orchestration elapsed 9.3387 minutes. Evidence: ../../diagnostics/capacity_check_20261120/{summary.json,analysis.json,capacity-grpo-mb16-long-v1-performance.json,capacity-grpo-mb16-long-v1-performance_steps.json,capacity-grpo-mb16-long-v1-console.log}. The saved runtime report's generic no-validation note predates a local wording correction; the test configuration did schedule a final 16-question validation, but it was never reached.

Backup: ../../.revision_backups/long_sequence_check_20261120/. Deployment command and CPU/GPU call IDs are preserved in the diagnostic output directory. No time/cost cap was introduced; a real OOM caused termination.


## 2026-11-20: natural 20-question length sample and export repair

The user requested 20 randomly selected training questions with an initial input cap of 384, without forced full-length generation. The existing 2000-question training set was sampled without replacement (seed 42). All sampled complete prompts were 262-281 tokens; the earlier full-training-set audit found a maximum of 347. The sampled task used the initial adapter, one stochastic response per question, native multi-turn search and a non-forced 4096 response ceiling. It was validation-only: no optimizer update, no checkpoint, and no benchmark scoring run. Formal Settings and Notebook outputs remain unchanged.

69 local CPU tests passed before submission, and the actual cloud CPU Hydra/import/tokenizer preflight passed. Both validation batches (16+4) finished generation, but veRL's final _dump_generations failed on a NumPy int64 JSON value. This is an incomplete export, NOT a successful 20-question report. Ray deduplicated console lines; only 12 distinct per-question length records survive there. The largest visible trajectory is 1173 tokens, which cannot be described as the maximum over the selected 20. The GPU sampler recorded a peak of 31,388 MiB (30.6523 GiB), with no sampler errors. This includes framework initialization and generation only; it does not measure backward-pass capacity. Cloud GPU function body ran 293.79 seconds; whole orchestration took 6.837 minutes. The application was stopped after failure. No GPU retry was submitted.

The pinned veRL 0.6.1 exporter passes NumPy scalar values directly to json.dumps. Added a guarded image-build compatibility fix at that common JSON boundary: np.generic and np.ndarray become ordinary JSON-compatible scalars/lists, including nested ground-truth answer arrays; unsupported objects still raise. The framework's generation, reward, and update algorithms remain unchanged. The image build exercised the actual installed _dump_generations method on CPU with integer/float/bool/string NumPy values and nested arrays, verifying numeric preservation and rejection of unsupported values. That check passed. It covers formal evaluation as well as this diagnostic, instead of changing each metric's meaning or converting numbers to strings.

Evidence: ../../diagnostics/natural_length_sample_20261120/{summary.json,length_report.json,partial_analysis.json,console.log,gpu_samples.jsonl,selection.json}. The original failed report is preserved. A complete second GPU sample requires the user's response to the pending rerun question; absence of a response is not permission. Backup: ../../.revision_backups/natural_length_sample_20261120/.


## 2026-11-20: selected formal H200 configuration

User approved H200, prompt_tokens=384, response_tokens=2560, and micro_batch_per_gpu=16 for formal work. Updated shared Settings defaults and only the formal run_job GPU declaration; diagnostic H100/H200 functions retain their explicit hardware. Effective update batch remains 2 questions x 16 responses = 32 records, accumulated over two micro-batches. Training remains 2,000 iterations per algorithm; evaluation and checkpoint intervals remain 100. This selection does not authorize or start a formal cloud run.

The formal English notebook now documents these settings and verification limits, uses the already tested byte-comparing workbook upload, waits for each enabled task through the shared polling helper, saves call IDs to formal_calls.jsonl, and preserves explicit action switches. The common runtime already includes dependency fixes, retrieval contract/batching/cache fixes, error forwarding, and NumPy JSON-export compatibility. Added progress.json and periodic Volume commits for formal work, with absent/transient metrics handled without turning them into training errors. Updated both deployment guides. Existing assets can be reused because the preparation identity excludes length and micro-batch settings.

Validation: 72 local CPU regression tests passed after updating the old batch-count expectation; a further progress-startup/transient-read regression passed with the two formal configuration/sequencing regressions (73 distinct tests total). Tests use simulated Modal calls; all Python sources and formal notebook cells parse. Saved notebook outputs were preserved (the formal notebook had none). No deployment, formal GPU execution, or benchmark evaluation was performed for this change. Backup: ../../.revision_backups/formal_h200_384_2560_mb16_20261120/.

Prior evidence: paired GRPO tests on the same two questions at 384+2048 completed on H100/16, H200/16, H200/32. Full-length mean step times were 34.52/34.00/34.15 seconds; sampled GPU memory peaks were 58.98/58.99/108.96 GiB. H200/32 passed that boundary but did not show a meaningful whole-step speed improvement. The CPU sampler could not read /sys/fs/cgroup/cpu.stat; no CPU-utilization peak is available. Those tests do not verify 2560, PPO on H200, convergence, or checkpoint recovery. Evidence: ../../diagnostics/h200_pair_20261120/analysis.json and summary.json.


## 2026-11-20: two independent H200 training jobs, 100 steps each

User confirmed PPO and GRPO each run 100 iterations on separate H200 GPUs. Shared Settings.steps is now 100; prompt=384, response=2560, micro-batch=16, and effective update batch=32 are unchanged. run_job allows at most two containers, each with one H200. The single formal notebook submits both algorithms before waiting; their run names and output/checkpoint directories differ. Prepared inputs and initial adapters are shared read-only during training. Final scoring remains on the same 125-question benchmark. The CPU retriever remains shared and may limit parallel throughput; no claim of a twofold speedup or a five-hour completion guarantee is made.

The shared polling helper observes each already-submitted call without resubmission. A real error stops the local workflow and reports any other pending call IDs; it does not silently cancel the other training job. Call IDs persist in formal_calls.jsonl. Setup and Direct Inference remain prerequisites, and final evaluations retain explicit switches. Existing diagnostics are separate from formal training.

All 75 local CPU tests passed, including simulated submit-before-wait ordering, exact 100-step settings, H200/two-container deployment declarations, polling success and failure without cancellation, and prior runtime regressions. Python sources and formal notebook cells parse. These tests do not verify live dual-GPU scheduling, shared-retriever performance under two training jobs, the 2560 length boundary, full checkpoint recovery, or final benchmark scores. No deployment or paid GPU run was started. Backup: ../../.revision_backups/formal_parallel_h200_20261120/.

## 2026-11-20: legacy builder in the course workspace

The minimal parallel test stopped during deployment, before submitting any GPU calls. The active workspace was neu-info5100-oak-spr-2025. Its legacy builder invoked python while bootstrapping the registry image, before the chained run_commands could create the executable alias. The failed attempt remains in ../../diagnostics/formal_minimal_parallel_20261120/.

Moved the existing Python 3.12 alias and version checks into from_registry(setup_dockerfile_commands=...). This preserves the image's installed Python environment and does not change training algorithms, package pins, GPU selection, or the workspace-wide builder setting. The actual installed SDK's generated command order passed local checks for builders 2024.10 and 2026.10; the five formal workflow tests also passed. Backup: ../../.revision_backups/course_builder_20261120/.

Actual deployment in the explicitly selected course profile succeeded in 345.923 seconds. Image-build dependency/native-import checks and the installed veRL JSON exporter check passed. Evidence: ../../diagnostics/course_builder_fix_20261120.log. This verifies image construction and deployment, not GPU training.

Read-only volume inspection found no /framework/qwen3b-framework-v1 or /retrieval in week08-search-r1-data. Existing unrelated directories were left untouched. The minimal test controller now selects the course profile explicitly and checks both preparation manifests before starting the retriever or GPU jobs. No new data preparation or GPU training was started in this repair; the one-step parallel test remains pending prepared assets.
