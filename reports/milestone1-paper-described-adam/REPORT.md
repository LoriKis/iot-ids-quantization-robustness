# Corrected paper-described Adam experiment

Variant C uses conventional `torch.optim.Adam`; it is not an exact reproduction
of Table 4. Historical AdamW results remain variant B. The completed run is
`runs/milestone1-paper-described-adam-rerun/`. The similarly named
`runs/milestone1-paper-described-adam/` directory is the preserved first
attempt and is not reused.

| Condition | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | MCC | ASR (%) |
|---|---:|---:|---:|---:|---:|
| Clean | 63.644 +/- 0.940 | 58.305 +/- 1.016 | 58.326 +/- 1.016 | 0.628 +/- 0.009 | 0.000 +/- 0.000 |
| FGSM | 15.422 +/- 4.278 | 12.919 +/- 3.981 | 12.916 +/- 3.993 | 0.124 +/- 0.044 | 77.266 +/- 5.509 |
| PGD | 7.922 +/- 1.660 | 6.524 +/- 1.092 | 6.509 +/- 1.085 | 0.045 +/- 0.018 | 89.030 +/- 3.744 |

The test population is the shared 4,500-row saved split. ASR uses the
clean-correct denominator. The overlap audit finds 2,804 of 4,500 test rows
(62.31%) with identical feature patterns in training; row-disjoint IDs
therefore do not establish duplicate-free or independent-device
generalization. The historical smoke audit found 61 distinct rows overlapping
the saved test partition, so the test set was not completely untouched
throughout development. Feature-space adversarial examples are not guaranteed
to be protocol-valid network traffic. Attack policy, environment, and
unresolved Table 4 correspondence are documented in the protocol and
correction notes.

Experiment C generates fresh Gaussian training noise independently for each
batch and epoch using declared derived local generators. This differs from
Experiment A, which materializes its noisy training tensor once before
training and reuses it across epochs while shuffling batches. Experiment B's
executed source is unavailable, so its historical noise and shuffle execution
cannot be independently verified.

This is a completed practical baseline and paper-described comparison study;
it does not claim that every official Milestone 1 requirement was fully met or
that an external reviewer approved the whole project.

## Evidence scope

Included prediction tables and manifests support metric and alignment recomputation. Checkpoint-dependent claims require the local run directory; historical AdamW checkpoint claims remain unavailable here.
