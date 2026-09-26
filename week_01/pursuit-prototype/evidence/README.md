# Curated evidence

These are selected records from actual runs, not illustrative outputs. Full local run directories remain ignored by Git.

- `english-two-rounds/`: September 26 rerun using the current English prompts and explicit coordinate checks. Two proposals, two real executions and the final factual review all passed on the first attempt. Retained call records are verbatim English originals; only provenance path fields were normalized.
- `baseline/`: earlier artillery-versus-light-tank experiment. history.json is the default designer input; spec, result, events and trajectory support it. This baseline used the earlier Lua-only retreat controller.
- `designer-two-rounds/`: successful two-light-tank loop, originally timestamped 20260925-114716-973. Includes the rejected first proposal, its correction, two real executions and final feedback.
- `designer-two-rounds/calls/`: saved requests, responses, parsed output and timing. Chinese text in these publication copies has been translated into English; they are not byte-for-byte raw transcripts. `final_review_1` and `final-review-unverified-original.json` contain an incorrect account of the attack roles and are **not accepted conclusions**. `verified_review_1` and final-review.json contain the strengthened fact check.
- Each experiment directory contains configuration, result, events, trajectory and native control commands.
- `provenance.json`: original runtime/source identification. Personal path fields were normalized, and translation metadata identifies the English publication copies. Hashes refer to the tested source versions, not the later submission-layout edits.

Omitted as redundant: repeated full spatial observations, engine-support folders, generated maps, duplicate engine logs and development backups. Retained event streams/results were copied without rewriting. Map hashes remain for comparison against a regenerated map using the corresponding tested source version.

All prompts concern this synthetic experiment. No credentials or model weights are included.

## Translated publication copies of the September 25 run

This translation section applies to designer-two-rounds/, not english-two-rounds/. Chinese prose in historical prompts, model responses, parsed proposals, history and final reviews was translated into English for submission. The original prompts asked for Chinese responses; their translations preserve that instruction to describe the actual run accurately. The current program asks for English responses instead and has now been tested in the separate english-two-rounds/ run.

These are translations of an earlier run, not results from rerunning the English prompts. The mistranscribed attack roles in the rejected free-text review are intentionally preserved as an error in the English translation; the verified review and game events remain the authoritative facts. Scenario settings, measured results, events, timestamps and timing/token-count metadata were not changed. Timing and token counts describe the original calls, not the translated text.

The original files remain in ignored local run/backup directories. The english_publication section in designer-two-rounds/provenance.json lists the translated files with their original and translated SHA-256 hashes. Existing source hashes continue to identify the earlier executed code.
