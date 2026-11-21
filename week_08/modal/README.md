# Week 08: Search-R1 with PPO and GRPO

This project adapts Search-R1 to **Qwen2.5-3B-Instruct** using **LoRA**, with PPO and GRPO implemented through veRL 0.6.1. Training and inference run on Modal. A shared CPU retriever provides Wikipedia passages through an E5 encoder and FAISS index.

The course-provided [tiny-zero-lora](https://github.com/suhabe/tiny-zero-lora) repository is the project starting point; this implementation uses veRL for both training methods. It is a limited-compute adaptation, not a reproduction of every setting in the Search-R1 paper.

## Current status and results

Both methods completed **100 training steps** and saved their final adapters and recoverable checkpoints. All three methods completed evaluation on the same **125 Bamboogle questions**. Evaluation uses one deterministic response per question and normalized exact-match accuracy (EM).

| Method | Correct / Total | EM | Recorded run time |
|---|---:|---:|---:|
| Direct Inference, with offline answer-tag correction | 6 / 125 | 4.8% | 3 min 16 sec |
| PPO + LoRA with search | 15 / 125 | 12.0% | 12 min 13 sec |
| GRPO + LoRA with search | 23 / 125 | 18.4% | 7 min 54 sec |

Direct Inference originally scored 0% because 122 responses lacked the required `<answer>...</answer>` tags. Wrapping the complete saved responses in these tags, without changing their content, and rescoring them produced 4.8%. No new inference was performed. The notebook preserves the original output and explains this correction.

PPO training took **2 hr 10 min 39 sec**; GRPO took **1 hr 54 min 4 sec**. They ran concurrently on separate H200 GPUs. The times above are recorded job durations, excluding deployment and queueing.

These are results from one run. Some PPO evaluation tool calls could not be parsed; missing or invalid final answers receive zero scores. The Direct result uses the documented offline format correction, while the PPO and GRPO results retain their original scoring. The comparison therefore reflects the current end-to-end pipelines, not an isolated measurement of the training algorithm alone.

## Formal configuration

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Training methods | PPO and GRPO, 100 steps each |
| Prepared training pool | 2,000 questions |
| Questions per step | 1 |
| Responses per training question | 8 |
| Micro-batch per GPU | 8 response records |
| LoRA | Rank 16, alpha 32, dropout 0.05 |
| Learning rates | Actor: 2e-5; Critic: 1e-5; constant schedule |
| Initial prompt / response limits | 384 / 2,560 tokens |
| Assistant turns / tool response limit | 5 turns / 512 characters |
| Training validation | 10 questions, before training and at step 100 |
| Checkpoint saving | Every 10 steps; retain the latest two |
| Gradient checkpointing | Disabled |
| Training hardware | One H200 per method; shared CPU retriever |

A 100-step run processes 100 question occurrences and generates 800 training responses per method; it does not traverse the entire 2,000-question pool.

## Files

| File | Purpose |
|---|---|
| `assignment_08_verl.ipynb` | Main notebook: configuration, training, evaluation and saved results |
| `search_r1_framework_modal.py` | Modal application and cloud environments |
| `search_r1_retriever.py` | Wikipedia retrieval implementation |
| `search_r1_framework/` | Configuration, data preparation, agents, scoring, execution and recovery modules |
| `requirements-local.txt` | Local notebook and Modal client dependencies |
| `requirements-cloud.txt` | Cloud training dependencies, installed during deployment |
| `Bamboogle Prerelease.xlsx` | Evaluation dataset |

Keep these files together. Model weights, prepared training data and the retrieval index are stored in Modal Volumes and are not bundled with the local source files.

## Running the notebook

From this directory, install the local dependencies and authenticate with the intended Modal workspace:

```sh
python -m pip install -r requirements-local.txt
python -m modal setup
python -m jupyterlab assignment_08_verl.ipynb
```

Use the notebook's deployment cell to create or update the application. Cloud dependencies are installed by Modal; do not install `requirements-cloud.txt` into the local notebook environment.

For a new workspace, enable workbook upload, data preparation and retrieval preparation before running training or evaluation. For a workspace with completed preparation, leave those actions disabled and reuse the existing assets.

The saved notebook is currently in **evaluation mode**: both training actions are `skip`, and Direct, PPO and GRPO evaluation are enabled. Saved outputs can be inspected without executing any cells. Existing evaluation names cannot be reused for a new run; change the evaluation run names only if intentionally repeating inference. Cloud execution incurs usage charges.

For new training, choose unused `PPO_RUN` and `GRPO_RUN` names and set:

```python
TRAINING_ACTIONS = {"ppo": "new", "grpo": "new"}
```

The notebook submits both training jobs before waiting. For evaluation, keep the names of the completed training runs and set both training actions to `skip`; the evaluator loads their final adapters without further training.

## Saved results and recovery

The `week08-search-r1-framework-results` Volume stores each run's logs, completion records, checkpoints and evaluation outputs. `summary.json` contains evaluation scores; `evaluation/0.jsonl` contains the individual responses. Stopping the Modal app does not delete these files.

To resume an interrupted training run, keep the same workspace, preparation, run name and settings, and set its action to `resume`. Use `skip` for a method that already finished. Recovery requires a verified complete checkpoint and does not silently start from scratch. A different workspace does not automatically inherit the existing Volumes.
