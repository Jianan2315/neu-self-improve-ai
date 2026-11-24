# Flow-GRPO training and tests

`05_agentflow_flow_grpo.ipynb` is the full run: 10 sequential planner updates on
one GPU selected by the notebook's `GPU` parameter, followed automatically by Bamboogle inference for 125 questions
and seeds 42/43/44. GPT-4o grades the 375 saved benchmark answers locally.
The same model and AdamW optimizer stay alive across all ten updates. Executor,
verifier, answer generator and tool helpers continue to use the frozen base.

## Full run

- `TRAINING_STEPS=10` is the chosen course-project budget, not a paper-specified training duration.
- Each update uses a new batch of 16 Search-R1 and 16 DeepMath questions, with 8
  trajectories per question. The ten batches contain 320 distinct questions and
  produce 2,560 training trajectories. Exact normalized Bamboogle matches are excluded.
- The current test settings are retained: rollout batch 32, tool workers 32,
  reward workers 8, update batch 1, and exact greedy-request reuse enabled.
- Training allows 3 interaction turns with planner temperature 0.5. Evaluation
  allows 10 turns with temperature 0.7, batch size 32, and the trained planner.
- A six-hour training estimate excludes final evaluation and grading. Hardware,
  generated lengths, tool responses and API waits affect the actual time.

Run the notebook locally with `RUN_TRAINING=True`, `RESUME_TRAINING=False`,
`RUN_JUDGE=True` and an unused `RUN_NAME`. One remote call performs all updates
and benchmark inference. The notebook does not stream Modal's logs.

After each completed update and each completed evaluation trial, the Modal Volume
`week09-flow-grpo-checkpoints` stores `<RUN_NAME>/checkpoint.pt`. This single latest
checkpoint contains planner adapter weights, AdamW state, random-generator states,
selected training batches, training results and completed evaluation trials. The
file is replaced before committing the Volume; older steps are not separate files.

To resume an interrupted run, use the same Modal workspace, run name and settings,
with `RUN_TRAINING=True` and `RESUME_TRAINING=True`. Completed updates and evaluation
trials are skipped. An unfinished update or trial is repeated. Resume requires at
least one saved checkpoint. Remote API results can change when work is repeated.
If all inference results are already local, set `RUN_TRAINING=False` to display them
or finish grading without allocating a GPU.

Local files, relative to the notebook folder:

| Path | Contents |
| --- | --- |
| `outputs/<RUN_NAME>/manifest.json` | Training/evaluation settings, model identity, prompts and tool descriptions. |
| `outputs/<RUN_NAME>/training_result.json` | Each completed update's training questions, eight answers per question, rewards, gradient/update metrics, times and memory peaks. |
| `outputs/<RUN_NAME>/sequential_response.json` | Three benchmark trials with seeds, measurements, questions, references, interaction histories, final answers and grading decisions. |

The final local training and answer files are written when the remote call returns.
Intermediate progress is saved in the cloud checkpoint. Training timing tables cover
completed updates; they exclude initial loading, checkpoint writes, failed attempts
and benchmark evaluation. The checkpoint is a PyTorch dictionary; restore its
`adapter` with PEFT's `set_peft_model_state_dict` after creating the same base model
and LoRA configuration. It is not a full copy of the 7B base weights.

## One-step tests

Run `03_flow_grpo_step_test.ipynb` in local Jupyter. It calls one H200 on Modal, collects the complete training batch, obtains GPT-4o rewards, and applies one AdamW update to the planner's LoRA adapter. The notebook displays timings and memory peaks. Post-training benchmark evaluation is omitted.

`04_flow_grpo_parallel_test.ipynb` runs the same complete update with concurrent tool
execution and reward judging, batched likelihood/gradient computation, and reuse of
identical greedy Qwen requests. It uses a separate Modal App, `week09-flow-grpo-parallel-test`, and a
separate local run directory.

Set `RUN_TRAINING=False` to read the saved result for `RUN_NAME`, without preparing
training data or calling Modal or the scoring API. To train again, set
`RUN_TRAINING=True` and choose an unused `RUN_NAME`.

The notebook does not enable Modal's live log output. It prints one waiting message,
saves the returned result, and displays the final tables. Stage and batch logs remain
available in Modal. Exceptions still propagate to the notebook.

## Files and functions

| File | Main functions and responsibilities |
| --- | --- |
| `__init__.py` | Package description. |
| `data.py` | `training_questions`: stream a seeded mixture from the Search-R1 and DeepMath training splits before allocating a GPU; exclude duplicate questions and exact normalized Bamboogle matches. `training_batches`: partition distinct questions into the requested per-step source mixture. |
| `policy.py` | `PolicyGenerator`: enable LoRA for planner requests, disable it for every other role, and optionally merge identical greedy requests within a call. `action_log_probs`: compute sampled planner-token likelihoods in padded batches, excluding padding and prompt tokens from the loss. `update_planner`: accumulate gradients over all planner turns and call the optimizer once. |
| `objective.py` | `group_advantages`: normalize rewards within each question's rollout group. `turn_loss`: clipped policy loss and sampled reference KL, averaged over that turn's tokens. |
| `training.py` | `load_policy`: initialize Qwen and the planner adapter. `train_step`: collect trajectories, judge rewards, update the planner and measure each stage, optionally reusing an existing model and optimizer. `judge_rewards`: grade final answers with bounded request concurrency and place each reward back in its question/trajectory position. |
| `modal_app.py` | `training_step`: allocate the H200 and reuse the existing model cache and isolated Python execution tool. |
| `parallel_tools.py` | `ConcurrentToolset.execute_many`: run independent tool tasks in worker threads and return results in command order. `ask`: queue Qwen helper prompts for batched generation by the coordinating thread. |
| `parallel_modal_app.py` | `training_step`: run the shared training procedure with `ConcurrentToolset` in a separate Modal App. |
| `run.py` | `run_training`: run sequential updates with persistent model/optimizer state and then evaluate the trained planner. `save_checkpoint`: replace and commit the latest complete state. `evaluate_trial`: generate one benchmark trial with the trained planner and frozen other modules. |
| `run_modal_app.py` | `create_app`: register `train_and_evaluate` with the notebook's GPU and API secret. The function mounts the model cache and checkpoint Volume and executes the full run in App `week09-flow-grpo`. |

## Configuration

The [paper](https://arxiv.org/html/2510.05592v2) specifies 32 questions per training batch, 8 trajectories per question, at most 3 interaction turns during training, temperature 0.5, planner output limit 2048, learning rate 1e-6 and KL coefficient 0.001. These settings are used here. One optimizer update is the requested test scale; the paper does not specify an exact total training-step count or identify step 60 as its best checkpoint.

The test uses 16 [Search-R1](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) questions and 16 [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K) questions. The 1:1 mixture, dataset revisions, seed, buffer shuffle, LoRA rank 16/alpha 32, clipping 0.2, gradient norm limit 1.0, zero AdamW weight decay, and rollout concurrency 32 are implementation choices. The source revisions and selected questions are saved with the results. Only question/answer columns are read; source solution traces are unused.

`ROLLOUT_BATCH_SIZE=32` controls concurrent trajectories, not the number of questions in an optimizer update. The default batch contains 256 trajectories, collected in 8 groups of 32 trajectories (four questions with eight trajectories each). Model parameters remain unchanged until all trajectories have rewards and all gradients have accumulated. Each trajectory runs at most 3 planner/executor/verifier turns before final-answer generation. In notebook 03, tool requests run sequentially as in the existing AgentFlow implementation.

In notebook 04, `TOOL_WORKERS=32` permits up to 32 concurrent tool tasks within a
group. Network requests and Python sandbox calls can overlap. Ready Qwen helper
prompts are batched, and only the coordinating thread calls the shared model and
switches its adapter state. Each turn waits for its tool results before verification;
the eight groups still run sequentially. Concurrency does not guarantee a proportional speedup
or identical generated answers. Actual context lengths and helper batches affect
memory use; API rate limits can also limit throughput.

Notebook 04 adds these implementation settings:

| Setting | Meaning |
| --- | --- |
| `REWARD_WORKERS=8` | Up to eight independent reward-judge API requests at once, after trajectory collection. Every answer is still judged separately. |
| `UPDATE_BATCH_SIZE=1` | One planner turn at a time per likelihood or gradient calculation. This does not change the 32-question training batch or the single optimizer update. |
| `REUSE_IDENTICAL_REQUESTS=True` | Merge complete, identical non-planner conversations within the same generation call at temperature zero; return a separate answer entry for each original request. Planner sampling is never merged. There is no cache across calls or updates. |

Notebook 03 retains one reward worker, one planner turn per update micro-batch,
and no request reuse unless these settings are explicitly added. Batching can
change floating-point results and generated text; equal seeds do not guarantee
identical trajectories across the two implementations.

Training uses the current OpenAI search/embedding tools and GPT-4o reward judge through `.env`. The executor, verifier, answer generator and tool helpers use the frozen base model with the adapter disabled. Only the planner's LoRA weights receive gradients. Reference answers go only to the reward judge. A judge/API failure stops the run instead of becoming a zero reward.

## Update and measurements

The loss averages in this order: action tokens within each turn, turns within each trajectory, trajectories within each question, then questions. Rewards are normalized across the 8 trajectories for the same question using sample standard deviation plus 1e-6. Equal rewards give zero advantages. KL uses the sampled k3 estimator. Old, current and frozen-reference token likelihoods all use the planner's sampling temperature. Exact prompts and sampled token IDs, including the first EOS, are held in CPU memory until the update.

Gradient accumulation processes up to `UPDATE_BATCH_SIZE` planner turns at a time.
Attention masks exclude padding; the loss keeps each turn's token mean and the
original question/trajectory/turn weights. All micro-batch gradients accumulate
before one gradient clipping operation and one optimizer step. Transformer gradient
checkpointing and chunked vocabulary projections reduce activation memory. Larger
update batches require more activation memory. One shared base model serves all roles.

The four measured stages are model loading, rollout generation (including tool calls), training reward judging, and parameter update (including old/reference likelihood preparation). `server_total_seconds` includes all four and their cleanup overhead. `client_call_seconds` also includes Modal startup and transfer. `peak_gpu_reserved_gib` is the maximum PyTorch allocator reservation, not GPU utilization or total device memory reported by NVIDIA. The GPU remains allocated during tool and judge calls, so those waits contribute to billed container time.

Notebook 04 also records `qwen_generation_seconds` in the rollout stage: time
inside all Qwen generation calls, including GPU synchronization. It excludes
tokenization, decoding, and external tool calls. Subtracting it from the rollout
duration therefore does not isolate network waiting. Compare runs with matching
questions, prompts, token limits and GPU hardware when assessing the timing change.
With request reuse enabled, the rollout stage also contains `qwen_requests_reused`:
the number of duplicate requests whose answers were copied from an identical request
in the same call. This counts avoided requests, not seconds saved.

`adapter_change_l2` measures the actual adapter parameter change. A zero change is reported as such; tied rewards can produce no learning. Training reward is not post-training benchmark accuracy.

## Local result

Paths are relative to the notebook folder:

| Path | Contents |
| --- | --- |
| `outputs/<RUN_NAME>/training_step.json` | Settings and model identity; actual GPU and library versions; stage times and memory peaks; update metrics; each selected question, reference answers, and its 8 generated answers with reward decisions. |

An existing run directory stops execution with `run name already used`. The test writes one result file after the remote step returns. No training checkpoint is saved by this timing test.
