# Milestone 1 correction notes

**2026-10-23 — local repair takeover.** Variant A (released-code baseline) is
preserved. Variant B remains the historical AdamW paper-described experiment;
its executed source snapshot and full run are unavailable on this Windows
machine. Variant C is the new corrected conventional-Adam experiment under
`docs/MILESTONE1_PAPER_DESCRIBED_ADAM_PROTOCOL.md`.

- **10a:** Lower accuracy does not prove that Table 4 used a different
  classification task. Table 4 correspondence remains unresolved.
- **10b:** Table 4 F1 is not called binary attack-positive F1 without verified
  evidence of that definition. Variant C reports macro-F1 and weighted-F1
  separately.
- **10c:** Adam and AdamW are not equivalent. Variant B is labelled AdamW;
  variant C uses `torch.optim.Adam` with conventional L2 weight decay.
- **10d:** Duplicate feature overlap is measured. Its origin is not asserted
  without evidence, and the declared row split is not replaced by a group split.
- **10e:** Checkpoint reload claims state their actual scope: variant C reloads
  clean logits for all 4,500 test rows; adversarial conditions are not
  reloaded unless explicitly reported.

The released baseline and paper-described configurations differ in several
settings and evaluation populations, so their results are not a single-factor
ablation. Epsilon `0.10` has different feature-space meaning under their two
scalers. No exact Table 4 reproduction or independent reviewer approval is
claimed.

The completed conventional-Adam run is
`runs/milestone1-paper-described-adam-rerun/`. The original
`runs/milestone1-paper-described-adam/` attempt remains preserved. The later
checkpoint-reload failure-handling patch changes current source code after that
run; preserved executed-source snapshots, fingerprints, manifests, checkpoints,
and evidence remain historical records and are not rewritten.
