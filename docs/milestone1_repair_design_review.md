# Milestone 1 repair design review (round 3, reviewing draft 3)

Verdict: CHANGES_REQUESTED (0 HIGH, 3 MEDIUM, 3 NIT)

Reviewed: `docs/MILESTONE1_REPAIR_DESIGN.md` against `docs/MILESTONE1_PRE_REPAIR_INVENTORY.md` and the original request. No builds or tests were run.

## Findings

1. MEDIUM. Wrong last-batch size (section 4.3, "Fixed quantities").
   The design says 42 steps per epoch with "last batch of 389 rows kept". 20,997 = 41 x 512 + 5, so the last batch has 5 rows, not 389. The step count 42 and total 1050 are right, but a test or assertion written from "389" would fail, or the coder will doubt the split size.
   Fix: replace with "41 full batches of 512 plus a final batch of 5 rows (20,997 = 41*512 + 5), no drop_last", and assert `batch_sizes[-1] == 5` in a test.

2. MEDIUM. Report corrections 10a-10e are still not specified (sections 9.4, 9.4a, open risk 6).
   The design admits it does not hold the verbatim 10a-10e. Its five entries are a guess built from the step brief. This reviewer's copy of the request is also truncated after item 6, so the "specifies the report corrections 10a-10e" requirement cannot be confirmed. The runtime gate (a `warning` at implementation step 2) defers the problem rather than fixing it, and there is no confirmation that the provisional entries match the user's items.
   Fix: before approval, the orchestrator must supply the verbatim items 7-11 (including 10a-10e) in `docs/MILESTONE1_REQUEST_ITEMS_7_11.md`. The designer then fills section 9.4b with a one-to-one table (item, files, lint or check) and removes "provisional" and open risk 6. Until that file exists, this requirement stays open.

3. MEDIUM. Order of the resume gate and the data fingerprint is under-specified (section 3.4).
   `check_resume` must run first in `main`, before any write. But the `data` component includes the scaler min/scale hashes, which need the CSV, the saved split and a scaler fit. The design does not say these are computed in memory with no writes. It also gives no exit code for "run directory has content but no `fingerprint.json`", and it does not say where `split_ids.json`/`preprocessing` are read from at gate time (the run-dir copy or the source file). That could let data preparation write files before the gate, which breaks the "reject before modifying" requirement.
   Fix: state that `compute_fingerprint` is read-only and takes the split from `cfg["split"]["source"]` (hash-verified). It computes the scaler in memory and writes nothing. `prepare_data_adam` runs only after `check_resume` passes. Use exit 3 for every refusal at the gate (mismatch or missing `fingerprint.json`). Add a Group C test that spies on all writes (for example a patched `open`/`os.replace`) during a rejected launch.

4. NIT. The paper wording check does not say how the "visual check" is done (section 4.1). pypdf text is the only tool named.
   Fix: record the extracted sentence, page number and a count of "AdamW" matches in the PDF text (expected 0) in the protocol, and say rendering is not required.

5. NIT. The mojibake examples in section 6 differ from the inventory (`ΓÇô`, `╬╡`, `┬▒`) and are themselves garbled.
   Fix: copy the sequences from the inventory, or drop the examples, and rely on the grep-based procedure.

6. NIT. The smoke directory's fingerprint and gate behavior is not stated (section 5.2). The full runner refuses to run inside it, but the design does not say whether the smoke writes a `fingerprint.json`.
   Fix: say the smoke writes its own `fingerprint.json` with the same function and is subject to the same gate.

## Verified assumptions
- The finalized housekeeping material is in HEAD per the inventory, and the design reuses it instead of rebuilding.
- `requirements-lock-windows.txt` exists and pins `torch==2.5.1+cpu`.
- `configs/milestone1_paper_described.json` has `dropout: 0.1`.
- In `paper_described_baseline.py`, `run_smoke` starts at line 823, with per-class `c_idx[:10]` and `c_idx[10:15]` at lines 842-843. The design's "near lines 841-846" is accurate.
- The design never edits historical runs, configs or recorded hashes, and it routes covered files to new notes.
- The design keeps A (65.808333 / 44.416667 / 41.641667, seeds 1, 7, 13, 29) as checker assertions.
- The Adam config is correct: lr 0.001, weight_decay 0.0001, betas (0.9, 0.999), eps 1e-8. The `decoupled_weight_decay` handling is coherent: pass `False` if the parameter exists, otherwise omit it.
- The shared split (20,997/4,500/4,500, seed 2025), 25 epochs and batch 512 are retained.
- The smoke is partition-safe, with ID assertions, saved selection and fresh weights.
- The domain policy matches the request: intersection bounds, projection of init and every update, an infeasible error, epsilon-zero identity, and no clamping of clean inputs.
- The test list covers the requested cases.
- The isolated checker, with its self-test, is specified.

## Unverified / wrong assumptions
- Wrong: the final batch of 389 rows (finding 1).
- Unverified: that 10a-10e match the user's text (finding 2).
- Unverified: the real-data infeasibility scan, hash coverage of README/REPORT/COMPARISON, and the compute time. The design discloses all three.
- Unverified by me: the torch 2.5.1 Adam signature. I did not run it, per the no-run rule.
