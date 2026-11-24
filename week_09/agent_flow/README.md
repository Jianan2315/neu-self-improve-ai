# AgentFlow code

Open `../02_agentflow.ipynb` in local Jupyter. The notebook runs three sequential
trials on Bamboogle, scores the answers and displays accuracy. One frozen
Qwen2.5-7B-Instruct model on Modal serves the planner, executor, verifier, solution
generator and the Qwen-based tools.

## Files

Paths are relative to the notebook folder.

| File | Contents |
| --- | --- |
| `agent_flow/__init__.py` | Package description. |
| `agent_flow/prompts.py` | Instructions for the modules and descriptions of the five tools. |
| `agent_flow/flow.py` | Planning, tool-call parsing, execution, verification, memory updates and final answers. |
| `agent_flow/tools.py` | Base Generator, Python Coder, OpenAI Search, Wikipedia Search and Web Search. |
| `agent_flow/inference.py` | Batched Qwen generation, one model load and sequential trials. |
| `agent_flow/modal_app.py` | GPU environment and entry point; isolated CPU execution for generated Python calculations. |
| `agent_flow/results.py` | Experiment manifest and access to the shared result-storage functions. |

Shared functions come from `baseline/inference.py` (model loading and EOS handling),
`baseline/results.py` (JSON reading/writing and grading storage),
`baseline/evaluation.py` (GPT-4o scoring and statistics) and `baseline/config.py`
(model identity and scoring instructions). Keep `baseline/` beside `agent_flow/`.

## Running

Use the notebook folder as the working directory. Select the Modal profile and an
unused `RUN_NAME` in configuration. With inference enabled, an existing run directory
stops execution with `run name already used`. For saved answers, retain the run name
and set `RUN_INFERENCE=False`.

Add this key to `.env` beside the notebook:

```dotenv
OPENAI_API_KEY=your_openai_key
```

OpenAI Search uses `gpt-5.4-mini` through the Responses API with the `web_search`
tool required and low reasoning effort. It returns evidence and cited source URLs.
Wikipedia Search and Web Search use OpenAI `text-embedding-3-small`; scoring uses
GPT-4o. All three services use the same `OPENAI_API_KEY`, with Responses, Embeddings
and Chat Completions permissions. The inference cell supplies this key through a
temporary Modal Secret. It is not included in saved settings or answers.
Set `RUN_JUDGE=True` to score generated answers.

API access and quotas must be enabled for the selected services. Authentication,
quota and service failures stop the run; they are not converted into incorrect
answers. Invalid model-generated plans or commands become feedback in the memory
so the planner can revise them. An inaccessible individual webpage also becomes
tool feedback.

Wikipedia search retries HTTP 429 up to three times after the initial request.
It follows `Retry-After` when valid; otherwise it waits 5, 10 and 20 seconds.
If the final attempt fails, the error stops the run.

## Execution and saved results

For each seed, the planner chooses a tool and sub-goal, the executor creates its
arguments, the tool returns evidence, and the verifier decides whether to continue.
The next planner call reads this accumulated memory. On STOP or the turn limit,
the solution generator produces the final answer. Reference answers are added
locally after inference and are used only for grading.

The same-stage Qwen requests are processed in batches of up to `BATCH_SIZE`.
Tool calls are synchronous. The three seeds run in order and share one loaded model.

| Relative path | Contents |
| --- | --- |
| `outputs/<RUN_NAME>/manifest.json` | Model identity, experiment settings, module instructions, tool descriptions and grading instructions. |
| `outputs/<RUN_NAME>/sequential_response.json` | Three trial groups containing answers and interaction histories; shared GPU, library and timing information. Each answer's `judge` field contains its grading result. |

Each answer's `trajectory` is the memory used by the agent: one entry per round,
containing the planner output, executor command, tool result and verifier output.
The answer's `input_tokens`, `output_tokens` and `finish_reason` describe the final
solution-generator call. Trial token totals include every local Qwen call, including
planning and tool summarization. External OpenAI search and embedding tokens are not
included in those Qwen totals.

`trial_seconds` includes Qwen generation, search and other tool work.
`generation_seconds` includes only Qwen's generation calls. `peak_gpu_reserved_gib`
is the peak GPU memory reserved by PyTorch in that trial. The final statistics are
calculated in the notebook without creating a separate summary or CSV file.

## Paper settings and implementation choices

The four-module loop, five tools, ten-turn evaluation limit, planner temperature
0.7, deterministic Qwen tool engines, three-trial evaluation and embedding-based page
retrieval follow [AgentFlow, sections 3.1, C.2 and Appendix E](https://arxiv.org/html/2510.05592v2).
Retrieval uses 200-word chunks, 20-word overlap, the top ten matches and at most
one million words per page.

The internet search backend uses [OpenAI Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search)
with [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini), replacing
the authors' Gemini-based Google Search tool. This is an implementation difference;
the requested search model is recorded in the manifest.

The planner, executor, verifier and solution-generator instructions follow Appendix
E.1.1-E.1.4. Requests supply the question, tool metadata and action history under the
paper's context labels. The Python Coder instruction follows E.2.2. Tool descriptions
follow E.2 with the actual input/output contracts listed in `prompts.py`. Grading
follows E.3, using JSON output and accepting any of the supplied reference answers.

Interface adaptations:

- OpenAI Search replaces Google Search. It accepts `query` and returns evidence and
  source URLs; citations are requested automatically, with no `add_citations` option.
- The executor emits one `execution = tool.execute(...)` call with literal keyword
  arguments and a string query. The paper's list-query example is replaced by an
  example supported by this interface.
- The verifier and solution generator read the planner's reasoning from the existing
  action history. There is no separate initial-analysis call or placeholder.
- Base Generator accepts text only. Wikipedia Search selects at most three of ten
  candidates and returns page summaries and URLs, without an `other pages` section.
  Web Search returns a URL and summary in a dictionary. Their descriptions match
  these actual results rather than promising unused paper fields.
- The Base Generator, Wikipedia page-selector and passage-summarizer instructions
  are local choices; E.2 does not give separate complete templates for those calls.
  Python Coder adds the permitted functions and sandbox restrictions to E.2.2.

Seeds 42/43/44, a Qwen
batch size of 32, 512-token limits for supporting modules and final answers,
and the 60-second HTTP timeout are local choices. The planner's 2048-token limit
uses the paper's stated training limit also for this evaluation. The executor
accepts one literal `tool.execute(...)` call rather than arbitrary Python statements.
Python Coder uses an isolated CPU sandbox with a five-second CPU limit. Context is
not silently truncated. These choices and live search changes can affect accuracy;
the result is not guaranteed to equal the paper's reported score.

The paper-aligned prompts require a new inference run. Earlier results from the
shortened prompts remain results of that earlier configuration.
