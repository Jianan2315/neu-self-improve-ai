# Local environment-designer loop

The model reads previous results and proposes a scene configuration. A fixed generator creates the map and Lua, a runner executes the real game, and results return to the model. Invalid proposals receive feedback and bounded retries; the runner does not silently substitute a handcrafted scene.

Read [SETUP.md](SETUP.md), [RESULTS.md](RESULTS.md), and [evidence/README.md](evidence/README.md).

## Run

After preparing dependencies, run `./run-designer.ps1` here. Defaults: local Ollama at localhost:11434, qwen3:8b, two rounds, at most three proposals per round. No API key is used. Options include `-Rounds 1`, `-RuntimeRoot 'YOUR_RUNTIME_DIRECTORY'`, and `-CheckOnly` to verify paths without starting a model or game.

Runtime selection: explicit -RuntimeRoot, environment variable OPENRA_RUNTIME_ROOT, ignored `.runtime-path` file, then `.runtime` in this directory. New output goes into ignored `designer-runs/<timestamp>/`. Errors and rejected proposals are preserved.

`./baseline/run.ps1` is the earlier fixed-scene entry point, not the designer loop. `baseline/scenario.json` retains the artillery baseline. New designer scenes use two light tanks.

## Directory layout

- Project root: the automatic designer entry point, shared execution code, prompt, Lua template, dependencies and documentation.
- `baseline/`: the fixed-scene launcher and configuration, usable to reproduce or refresh the initial baseline.
- `tests/`: development checks with simulated inputs/results. They are run explicitly and are not imported by the normal experiment entry points.
- `evidence/`: curated inputs and results from actual experiments, including the baseline history used by the designer.

The automatic designer uses `evidence/baseline/history.json` by default. It does not need the test files or the fixed-baseline launcher to run. Keep `pursuit.py` in the project root: both the fixed baseline and automatic designer use its map generation, execution and evaluation functions.

## Components

| File | Responsibility |
|---|---|
| designer_loop.py | Local model calls, constrained output, checks, bounded repair, execution/feedback orchestration |
| designer_context.md | Domain capabilities and constraints supplied to the model |
| pursuit.py | Validate configuration, generate map, run engine, evaluate events |
| scenario.lua.template | First-hit trigger, retreat and scene telemetry |
| runtime.ps1 | Machine-local dependency discovery |
| tests/test_pursuit.py / tests/test_designer_loop.py | Validation and feedback-propagation checks |

Both units are 1tnk, A=(16,18), B=(21,18), seed=1234. A initially holds fire, then attacks. At its first positive hit, the script queues retreat. B receives no external movement/attack orders.

The fixed checks are A starting cell [16, 18], B starting cell [21, 18], and first escape destination [3, 18]. Each is checked directly against its own coordinate. Round 1 uses only that destination. Later rounds retain each prior destination and append a turning destination chosen by the model; no fixed coordinate is imposed on the new point. Every point is checked against x=3..108 and y=3..49. The host sets the curriculum; the model also chooses attack_tick in 1–10, max_ticks in 600–1500, and sample_interval in 5–10. At most three rounds are supported.

The native Stop order is sent to A after the Lua retreat trigger to clear its persistent turret target, then the same route is requeued. No engine, weapon, health or speed modifications are made. B remains uncontrolled by the test driver. Native command timing is logged.

Validity requires one A hit, the retreat trigger, observations and an ending event, and no B damage before the first A hit. Death or absence of pursuit alone does not invalidate a scene. Reaching the first point, reaching later points, actually executing a turn, and survival are recorded outcomes, not required pass conditions. A valid experiment can advance despite an incomplete route.

Final factual fields returned by the model are checked against logs; the program renders the factual summary. Free-text reasons and next questions are not verified causal explanations.

## Checks and scope

Run `python -m unittest discover -s tests` here. Fourteen checks passed using Python's standard library; they do not replace real execution.

The default seed [evidence/baseline/history.json](evidence/baseline/history.json) is committed, so the ignored old runs-final folder is unnecessary. For a fresh baseline, run `./baseline/run.ps1` and pass its history via -SeedHistory.

Current scope is configuration design through a fixed template: no arbitrary environment code, obstacles, teleportation, multiple targets, policy training, or direct measurement of internal aggro targets. Added turns are structural complexity, not proven training benefit.

The current prompts request English explanations and questions, and the factual summary is rendered in English. Submitted historical model-call records are labeled English translations; see evidence/README.md for provenance.

A fresh two-round run with the current English prompts and coordinate checks completed on September 26, 2026. Both experiments and the final fact review passed. See RESULTS.md and evidence/english-two-rounds/ for original English call records and actual game results.
