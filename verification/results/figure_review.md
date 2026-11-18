# Figure review (post-run, by eye)

The original PNGs were opened and compared with the recomputed values in [`verify_m2.txt`](verify_m2.txt). They were **not**
recomputed pixel by pixel. The recomputed series were plotted separately into [`../figures/`](../figures/fgsm_recomputed.png)
(labelled post-run regeneration; the originals stay in `reports/milestone2/original/figures/`).

What the originals show (1650 x 630 px, two panels each: accuracy left, macro-F1 right; x axis "epsilon (feature space)" with ticks 0.00, 0.05, 0.10;
title says "test subset n=270 (seed 1, 25 epochs)"; INT8 models drawn dashed and labelled "approx. BPDA gradient"):

- FGSM, accuracy: FP32-ordinary starts at about 68.5 and falls to about 26 and 7; INT8-ordinary about 66.7, 28.5, 7.4;
  FP32-adversarial about 60.7, 48.5, 36; INT8-adversarial about 60, 49.3, 37. Matches the recomputed table (68.52/26.30/6.67, 66.67/28.52/7.41,
  60.74/48.52/35.93, 60.00/49.26/37.04).
- PGD-5, accuracy: about 68.5, 20.7, 4.4 (FP32-ordinary); 66.7, 18.5, 4.8 (INT8-ordinary); 60.7, 48.5, 31.9 (FP32-adversarial);
  60, 48.9, 33 (INT8-adversarial). Matches (68.52/20.74/4.44, 66.67/18.52/4.81, 60.74/48.52/31.85, 60.00/48.89/32.96).
- Macro-F1 panels start at about 64.5, 62.4, 54.8, 54 and show the same ordering.
- The starting point (epsilon 0) of every curve is the subset score (about 68.5 for FP32-ordinary), not the 61.96 % full-test score.
- Reading limits: dots are read by eye, so values are approximate. The exact values are in the tables of `REPORT.md` and `verify_m2.json`.
