# Baseline code

The notebook coordinates configuration, inference, grading and presentation. The six Python files below provide its supporting functions. Paths in the table are relative to the notebook folder.

## Files and responsibilities

| File | Responsibility | Main definitions |
| --- | --- | --- |
| [`baseline/__init__.py`](__init__.py) | Initializes the Python package; contains the package description. | No experiment functions. |
| [`baseline/config.py`](config.py) | Shared model identifier, answer instructions, grading instructions and saved-format version. | `MODEL_ID`, `SYSTEM_PROMPT`, `JUDGE_PROMPT`, `SCHEMA`. |
| [`baseline/modal_app.py`](modal_app.py) | Modal image, library versions, GPU allocation, model-cache volume and remote entry point. | `app`, `answer_trials()`. |
| [`baseline/inference.py`](inference.py) | Loads the model once, generates one batch per seed, and runs the trials in order. Collects answers, token counts, GPU memory measurements and timing. | `load_model()`, `infer()`, `generate_batch()`, `trim_generated_tokens()`, `run_sequential()`. |
| [`baseline/results.py`](results.py) | Reads the benchmark, creates experiment files, copies answers for regrading, adds reference answers to returned responses, and saves grading decisions in the result JSON. | `load_benchmark()`, `prepare_run()`, `prepare_regrading()`, `prepare_response()`, `load_records()`, `response_records()`, `save_judgment()`, `read_json()`, `write_json()`. |
| [`baseline/evaluation.py`](evaluation.py) | Requests and parses GPT-4o grading decisions, then calculates trial accuracies and their mean and sample standard deviation. | `judge_answer()`, `summary()`. |

Run name, seeds, GPU choice, temperature, answer-length limit and grading switch are configured in the notebook.

## Execution flow

1. **Local setup:** the notebook reads `bamboogle.json` through `results.load_benchmark()`.
2. **Experiment setup:** `results.prepare_run()` rejects an existing `outputs/<RUN_NAME>/` directory, creates a new directory and writes its settings. With inference disabled, the notebook either copies answers for regrading or reads an existing run.
3. **Cloud inference:** the notebook calls `modal_app.answer_trials()`. It passes the trial jobs to `inference.run_sequential()`, which loads one model and calls `infer()` for each seed. `infer()` uses `generate_batch()` to generate answers.
4. **Local saving:** after all trials succeed, `results.prepare_response()` adds question identifiers and reference answers to the returned answers. `results.write_json()` saves the combined result. Inference errors propagate to the notebook without saving partial answers.
5. **Local grading:** the notebook calls `evaluation.judge_answer()` with the run's saved scoring instructions for an ungraded answer, then `results.save_judgment()` writes the decision back into the result file. Already graded answers are skipped.
6. **Local presentation:** the notebook uses the loaded answers; `evaluation.summary()` calculates statistics; the notebook displays tables and the accuracy chart.

GPU libraries are imported inside the inference functions when those functions execute in the cloud. Importing the supporting modules locally does not load the model.

## Regrading saved answers

Set `RUN_INFERENCE=False`, `REUSE_ANSWERS_FROM` to the original run name, and
`RUN_NAME` to an unused name. `prepare_regrading()` copies the manifest and answers
into the new directory, records `source_run`, replaces the scoring instructions and
judge model, and clears the copied scores. The original run is unchanged. Generation
settings and timing measurements remain those of the original inference.

Run the notebook to grade the copied answers with `RUN_JUDGE=True`. To resume or
display this new run later, retain its `RUN_NAME` and set `REUSE_ANSWERS_FROM=None`.
No GPU inference is needed for regrading.

The scoring instructions follow [Appendix E.3](https://arxiv.org/html/2510.05592v2),
with JSON output and alternative reference answers adapted to the local interface.
The direct-answer system prompt, temperature 0.7 and 512-token output limit are
implementation choices; the paper does not specify this complete direct-answer
configuration. Changing the judge does not change the saved Qwen answers.

## Saved files

These paths are relative to the notebook folder:

| Path | Contents |
| --- | --- |
| `outputs/<RUN_NAME>/manifest.json` | Experiment settings, model identity and version, and answer/grading instructions. |
| `outputs/<RUN_NAME>/sequential_response.json` | Answers from all trials, reference answers, grading decisions and explanations, timing and GPU measurements. |

`results.py` owns file reads and writes. `evaluation.py` returns grading decisions and calculated statistics to the notebook. Grading updates use a temporary file followed by replacement of the result file.

Model identity and revision are stored only in the manifest. The result JSON stores actual GPU details, library versions, common input measurements and model-loading time once under `output`. Each trial stores its trial number, seed, generation measurements and answers. Each answer stores its question identifier, question, reference answers, response, token counts, finish reason and grading decision. `response_records()` adds trial and seed to temporary table rows in memory; these copies are not written into the answer entries.
