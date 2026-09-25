# Week 01 — Environment Designer for RTS Pursuit Scenarios

This assignment implements a bounded, SPADE-inspired environment designer using **OpenRA-RL** and local **Qwen3 8B**. The domain is unit pursuit in OpenRA's Red Alert mod, not Red Alert 2. **No policy is trained.**

## Review order

1. [Implementation and entry points](pursuit-prototype/README.md)
2. [Dependencies and Windows setup](pursuit-prototype/SETUP.md)
3. [Actual results and limitations](pursuit-prototype/RESULTS.md)
4. [Curated experiment evidence](pursuit-prototype/evidence/README.md)

## Requirement mapping

| Assignment requirement | Implementation |
|---|---|
| Domain-specific environment library with recent development | OpenRA-RL, pinned to an April 22, 2026 commit, within six months of the September 25, 2026 validation date. See dependencies.json. |
| Study the library and run experiments | Inspected backend, map-loading and action interfaces; ran real games. |
| Environment designer loop | Model proposal → validation/repair → map generation → actual execution → feedback to the model. |
| Progressively more complex environments | A bounded curriculum adds a route turn. The model chooses the appended destination; the second scene actually reached both destinations. |
| No need to train a policy | Designer weights and test actor controls stay fixed. |

This is a restricted configuration-design implementation, not the paper's full free-form code generator or shared-model training system. A more complex route does not establish greater learning value or task difficulty.

Third-party checkouts, SDKs, virtual environments, model weights and repeated runs are local dependencies/output, not submission files. Source and setup instructions do not require the author's personal directory layout.
