# Week 09 — Bamboogle experiments

Open the notebooks in local Jupyter:

- `01_qwen7b_baseline.ipynb`: Qwen2.5-7B-Instruct direct answering.
- `02_agentflow.ipynb`: AgentFlow with tool use and an untrained planner. See [agent_flow/README.md](agent_flow/README.md) for its files and API setup.
- `03_flow_grpo_step_test.ipynb`: one complete Flow-GRPO training update on H200, including training rewards, stage timings and memory peaks. See [flow_grpo/README.md](flow_grpo/README.md).
- `04_flow_grpo_parallel_test.ipynb`: the same one-step test with concurrent tool tasks and batched Qwen helper requests.
- `05_agentflow_flow_grpo.ipynb`: ten consecutive Flow-GRPO training updates on the GPU selected in the notebook, then automatic Bamboogle inference and grading for three seeds. Saves the latest cloud checkpoint after each update; see [flow_grpo/README.md](flow_grpo/README.md#full-run).

## Completed experiment results

All three methods were evaluated on the same 125 Bamboogle questions with seeds
42, 43 and 44, producing 375 answers per method. All answers were graded with
`gpt-4o-2024-08-06` using the same Appendix E.3-based instructions and reference
answers. Accuracy below is the mean across the three trials; SD is the sample
standard deviation across those trials, in percentage points.

| Method | Correct answers, seeds 42 / 43 / 44 (each out of 125) | Our accuracy, mean ± SD | Paper accuracy | Our mean minus paper |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct direct answering | 44 / 49 / 48 | 37.6% ± 2.1 | 12.0% | +25.6 pp |
| AgentFlow, untrained planner | 49 / 59 / 51 | 42.4% ± 4.2 | 58.4% | −16.0 pp |
| AgentFlow with Flow-GRPO, 10 updates | 61 / 57 / 53 | 45.6% ± 3.2 | 69.6% | −24.0 pp |

Paper values are the Bamboogle results in
[AgentFlow Table 1](https://arxiv.org/html/2510.05592v2#S4.T1).
The saved runs are, respectively:

- `outputs/qwen7b-direct-paper-judge-v1/`
- `outputs/qwen7b-agentflow-paper-prompts-v1/`
- `outputs/qwen7b-flow-grpo-10steps-h200-v2/`

Each run's `manifest.json` contains its settings and prompts;
`sequential_response.json` contains its answers and grading decisions. The trained
run also contains `training_result.json` with the ten completed updates.

### Interpretation

In this experiment, untrained AgentFlow exceeded direct answering by **4.8
percentage points**. The trained planner exceeded untrained AgentFlow by **3.2
percentage points**, equivalent to 12 additional correct answers across 375
responses. The training comparison was positive for seeds 42 and 44 but negative
for seed 43. These results show a higher average score after the limited training
run, rather than an improvement in every trial or an established general benefit.

The ordering of the three methods matches the paper, but the absolute accuracies
and improvement sizes do not. The paper's training gain is 11.2 percentage points,
compared with 3.2 here. This is a resource-limited implementation and evaluation;
it does not reproduce the paper's reported accuracy levels. The untrained and
trained evaluations share the module prompts, search backend and grading protocol,
but differ in hardware and execution scheduling, so the comparison is not a
strictly isolated test of the learned weights.

### Implementation differences and possible explanations

The following are relevant differences or unresolved matches in settings we
actually use. Their effects on the accuracy gaps have not been isolated by
controlled experiments.

1. **Direct-answer prompt.** Our baseline uses the locally chosen instruction:
   “Answer the factual question using your own knowledge. You may reason briefly,
   then give a concise final answer.” We did not find an explicit prompt identified
   as the direct-answer baseline's template in the paper. Appendix E's AgentFlow
   module and tool templates do not establish what was sent to that separate
   baseline. A prompt difference is therefore a plausible explanation for part of
   the **37.6% versus 12.0%** gap, but the author's actual prompt and its effect
   remain unverified. Our three trial scores, 35.2%, 39.2% and 38.4%, provide no
   evidence that this large gap is explained simply by unlucky random seeds.
2. **Generation settings and prompt adaptations.** Our direct baseline uses
   temperature 0.7 and a 512-token output limit; the paper does not clearly specify
   the corresponding complete direct-answer settings. For AgentFlow, the four
   module prompts follow Appendix E with interface adaptations. Tool descriptions,
   some helper prompts, and the 512-token executor/verifier/final-answer limits are
   implementation choices. These can affect reasoning, tool selection and when
   the agent stops. The detailed adaptations are listed in
   [agent_flow/README.md](agent_flow/README.md#paper-settings-and-implementation-choices).
3. **Search evidence.** Our AgentFlow uses OpenAI web search with `gpt-5.4-mini`
   instead of the authors' Gemini-based Google Search tool. Retrieved pages and
   summaries can differ, changing the evidence available to both untrained and
   trained AgentFlow. This difference does not apply to direct answering, which
   uses no search tools.
4. **Training budget and parameter updates.** We completed 10 updates on one H200,
   training only the planner's rank-16 LoRA adapter. Each update used 32 questions
   and eight trajectories per question: 320 distinct training questions and 2,560
   trajectories in total. The selected batches contained equal numbers of Search-R1
   and DeepMath questions. Ten updates, the 1:1 mixture and LoRA configuration are
   local budget/implementation choices; they are not established matches to the
   authors' complete training run. The paper reports eight A100 GPUs. The limited
   update budget and adapter training may contribute to the smaller training gain;
   GPU count alone does not explain accuracy, and training differences cannot
   explain the untrained AgentFlow gap.

An earlier run with locally written prompts was reported at approximately 61.8%,
but its answer files are no longer present in the current results directory and
its complete evaluation conditions could not be rechecked. It is excluded from
the formal comparison. No additional prompt-only experiment was run because of
the remaining compute budget, so this project does not establish that prompt
changes outperform Flow-GRPO training.

## Files

| Path | Contents |
| --- | --- |
| `agent_flow/README.md` | AgentFlow file responsibilities, tool setup and implementation choices. |
| `baseline/README.md` | Python file responsibilities, main functions and execution flow. |
| `baseline/__init__.py` | Python package initialization and description. |
| `baseline/config.py` | Model identifier, answer/grading prompts and saved-format version. |
| `baseline/modal_app.py` | Modal image, model-cache volume, resource allocation and remote entry point. |
| `baseline/inference.py` | Model loading, batched generation, sequential trials and GPU-memory/timing measurements. |
| `baseline/results.py` | Benchmark reading, experiment files and saved grading decisions. |
| `baseline/evaluation.py` | GPT-4o grading and accuracy statistics. |
| `bamboogle.json` | 125 questions, their identifiers and reference answers. |
| `outputs/<RUN_NAME>/` | Saved settings, answers, timing and evaluation results. |
| `compatibility/` | Existing veRL/Flow-GRPO compatibility investigation. |

`bamboogle.json` was converted from `week_08/modal/Bamboogle Prerelease.xlsx` (`Sheet1`, question column A and reference-answer columns E/F).

## Running

1. Use an authenticated local Modal client (`python -m modal setup` if needed).
2. Use the folder containing the notebook as the Jupyter working directory, and run the configuration cell.
3. Set `RUN_INFERENCE=True` to generate answers. The inference cell stops with `run name already used` if `outputs/<RUN_NAME>/` already exists.
4. Put your OpenAI API key in `.env` beside the notebook, then set `RUN_JUDGE=True` to score saved answers. Keep `RUN_INFERENCE=False` to use existing answers. The scoring cell reads `.env` each time it runs.

```dotenv
OPENAI_API_KEY=your_api_key_here
```

AgentFlow uses the same `OPENAI_API_KEY` for search, embeddings and scoring. Enable Responses, Embeddings and Chat Completions access; see [AgentFlow setup](agent_flow/README.md#running).

`.env` is excluded from Git.

## Results and repeat runs

Paths below are relative to the notebook folder. Replace `<RUN_NAME>` with the configured run name.

| Relative path | Contents |
| --- | --- |
| `outputs/<RUN_NAME>/manifest.json` | Initial experiment settings, model identity and revision, and generation/grading instructions. AgentFlow also records its module instructions and tool descriptions. |
| `outputs/<RUN_NAME>/sequential_response.json` | Three trial groups. Each trial contains its seed, measurements and answer records with questions, reference answers, generated answers and grading results. AgentFlow answers also contain their interaction histories. Shared GPU information and overall timings are stored outside the trial groups. |

Scoring reads `outputs/<RUN_NAME>/sequential_response.json` and saves each decision in the corresponding answer's `judge` field in that file. The notebook calculates accuracy from those decisions. Already graded answers are skipped.

Answers are stored once inside the result file. Each update is written to a temporary file and then replaces the result file. Inference errors stop the notebook before the answer file is written. The run directory and manifest remain; use a new RUN_NAME or delete that directory before another inference attempt.

To run inference again, choose an unused `RUN_NAME` or delete the previous run directory yourself. To grade or display an existing run, keep its `RUN_NAME` and set `RUN_INFERENCE=False`.

For baseline regrading with the updated paper instructions, set `REUSE_ANSWERS_FROM`
to the original run, use an unused `RUN_NAME`, and keep `RUN_INFERENCE=False`.
This creates a copy with ungraded answers and preserves the original run. To resume
the copy, set `REUSE_ANSWERS_FROM=None`. Generation timings in the copy remain those
of the source run. See [baseline/README.md](baseline/README.md#regrading-saved-answers).

The cloud App names are `week09-qwen7b-sequential-baseline` and `week09-agentflow`; `RUN_NAME` names the local experiment directory. GPU requests use `A100-80GB`; the actual allocated GPU name is recorded with the results.
