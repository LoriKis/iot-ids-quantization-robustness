# Milestone 1 repair design: round-3 recheck (draft 4)

Scope: confirm only the three round-3 MEDIUM findings. Other sections were not re-verified. Informational; downstream steps proceed.

## Result: all three resolved

1. Last batch size. Resolved. Sections 4.3 and 12.1 (Group B test 7) now state 20,997 = 41*512 + 5, 42 steps per epoch, no `drop_last`, 1,050 total steps. The test asserts `len(batch_sizes) == 42`, `batch_sizes[:41] == [512]*41`, `batch_sizes[-1] == 5` and `sum == 20997`. An integration check on optimizer step counts (42 after epoch 1, 1050 after epoch 25) is also specified.

2. Items 10a-10e. Resolved. `grep` finds no "provisional" in the design. Section 9.4b is a one-to-one table (10a to 10e) giving verbatim wording, carrying files, old wording with source lines, and the enforcing check `CL-10a`..`CL-10e`. I compared the five quoted items to `docs/MILESTONE1_REQUEST_ITEMS_7_11.md` and they match. The request-items file hash is pinned in the preflight.

3. Resume gate. Resolved. Section 3.4 specifies:
   - `compute_fingerprint` is read-only and in-memory. It verifies the split hash before use and fits the scaler in memory.
   - `check_resume` runs first, before any write: lock, status, log, snapshots and `prepare_data_adam`.
   - Exit code 3 for every refusal: component mismatch, content without `fingerprint.json`, checkpoint fingerprint mismatch, and unreadable `fingerprint.json`.
   - The smoke directory is gated the same way.
   - Group C test 8 is a no-writes spy test. It patches `open`, `os.replace`, `os.makedirs` and the other write paths, and asserts exit code 3 with an empty spy log.

## Residual NITs (non-blocking)

- The 10e row lists R, K, P, N but not M (README). Add M if the README should carry the reload-coverage statement.
- Checkpoint fingerprint refusal (c) reads an `epoch-NN.pt` at the gate. Make sure the read-only open is exempt from the spy's write-mode filter, which it is by the stated mode rule.
- The regexes and line numbers in 9.4b were confirmed by grep at design time but the forbidden regexes have not been run against real files. The Group E fixtures cover this.

Verdict for these three findings: RESOLVED. No HIGH or MEDIUM findings remain from them.
