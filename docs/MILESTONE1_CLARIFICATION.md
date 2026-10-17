# Clarification draft — not sent

We are reproducing the CICIoT Standard Transformer baseline for Milestone 1.
Table 4 reports clean accuracy/F1 98.68/98.63 and FGSM accuracy/F1/ASR
98.19/97.95/0.81. The paper describes a binary setting, while our assignment asks
for multiclass; the public artifact instead uses a balanced 27-class CSV and
different scaling, splits, training and PGD settings.

Could you confirm whether Table 4 is binary, multiclass, or aggregated, and
identify the intended published multiclass reference if different? Please share
the corresponding generating script/command and configuration, processed data
with feature/class mapping and train/validation/test row IDs (or an exact sampling
and split recipe), and Standard Transformer checkpoint(s)/selection rule. We also
need its Transformer sequence construction, Phase 1 seed list and aggregation
procedure, F1 averaging, exact ASR numerator/denominator, and per-seed results.
For PGD, Section 5.4 gives seven steps with alpha=.025 and random start, whereas
Section 3.3 says alpha=epsilon/K; please confirm the executed settings and clipping.
The paper promises splits/checkpoints, but the public v1.0 release has no attached
assets. The anonymous README and six relevant scientific source files match the
public release; its nested file listings and processed CSV request returned 403.

These materials would let us reproduce the stated experiment without substituting
the released revision experiment or inventing missing scientific settings.
