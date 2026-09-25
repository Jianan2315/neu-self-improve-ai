# Observed results and limitations

Validation date: September 25, 2026. The actual two-round loop used local qwen3:8b and the pinned OpenRA-RL/OpenRA runtime, without policy training or cloud calls.

## Library investigation

The multi-session backend loads generated maps via reset(map_data=...) and advances simulation through gRPC. Its existing action interface provides native Stop and queued Move orders. Lua damage/death callbacks and position/health telemetry provide the event record.

Initial smoke tests verified movement, stopping, reset and custom-map damage. Five selected library test modules passed 447 checks, including mocked tests. The submitted prototype has a separate reproducible set of 14 standard-library checks. Neither set replaces the actual game runs below.

## Actual two-round designer run

Common conditions: two 1tnk light tanks, A=(16,18), B=(21,18), seed=1234, attack_tick=10, max_ticks=1200, sample_interval=5. B receives no external movement/attack orders.

| Measurement | Round 1 | Round 2 |
|---|---|---|
| Scene | designer_r01_a02 | designer_r02_a01 |
| Escape destinations | (3,18) | (3,18), (3,15) |
| Planned turns | 0 | 1 |
| First A hit on B | tick 30 | tick 30 |
| Total A hits on B | 1 | 1 |
| Reached destinations | 1, at tick 176 | 2, at ticks 176 and 216 |
| End reason | attacker_dead | attacker_dead |
| Experimental validity | Passed | Passed |
| Runner process cleanup | Confirmed | Confirmed |

The first proposal chose the wrong boundary and was automatically rejected; the model corrected it after feedback. The model selected the second round's new destination (3,15), which was actually reached. Results then returned to the model for factual review and a suggested next question. See [status](evidence/designer-two-rounds/status.json), [history](evidence/designer-two-rounds/history.json) and [final review](evidence/designer-two-rounds/final-review.json).

Reaching the route did not mean survival: A died in both experiments. The scheduled curriculum is a host constraint, not an autonomous discovery of an optimal training curriculum.

## Control bug corrected

Lua Stop cancels an activity but may leave AttackFollow's persistent turret target. A light tank continued firing while retreating, failing the one-hit requirement. The driver now sends A a native Stop after the Lua retreat trigger and requeues the same route; the engine clears the persistent target for that order. Engine source, unit attributes and B's orders were not changed.

Both final runs confirm one A hit. The validity condition was retained, not relaxed to accept a failure. Command timing is recorded in controller-actions.json. This is an intervention on fixed test actor A, not on the observed pursuit behavior of B.

## Reliability boundaries

- The model made configuration errors and once reversed the attack roles in free text. Final factual fields are now checked; the program renders the factual summary. The original erroneous output is retained and labeled in the evidence guide.
- Model reasons and next questions are suggestions, not verified causal explanations; they can be shallow or repetitive.
- Telemetry does not directly expose B's internal target lock or disengagement cause.
- This restricted configuration generator has no obstacles, teleportation, multiple targets, arbitrary environment-code generation or policy learning.
- Two scenes and one model setup do not establish broad reliability or cross-machine determinism.

## Repository cleanup after validation

Personal paths were removed from the entry points and the seed moved to committed evidence/. Simulation and designer decision behavior were not changed by this cleanup; path resolution and lightweight tests were rechecked. Curated provenance records the original executed source hashes and the strengthened fact-review version. These hashes do not claim that the later repository-layout cleanup reran the entire game experiment.

## English-language submission

After the recorded experiments, prompts and deterministic summaries were changed to English, and historical Chinese prose was translated for publication. This language change does not alter the simulator or scenario validation rules, but a model may produce different proposals under an English prompt. The recorded two-round results belong to the original prompts; no full game rerun is claimed for this language-only revision. See evidence/README.md and provenance.json for translation details.
