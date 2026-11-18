# Post-run adaptation note (documentation only)

**This folder is post-run work. It contains no code that was executed on the VM and no code written for this repository.**
The files under [`../source/`](../source/README.md) are the code that existed on the VM (V4 base and deadline overlay, kept
separate). They are **not** this folder. No wrapper was adapted or written for the repository layout, because the review script
([`../../../verification/verify_m2.py`](../../../verification/verify_m2.py)) only reads saved files.

**No end-to-end reproduction from this repository layout was performed.** Nothing here was trained, evaluated, attacked or benchmarked on the laptop.

## What a later person would have to adapt (untested, NOT PERFORMED HERE)

The overlay and V4 code assume the VM layout (a V4 folder and a sibling overlay folder, V4 located through `V4_ROOT`; see
[`../source/deadline-v1-overlay/HANDOFF.md`](../source/deadline-v1-overlay/HANDOFF.md)). To rerun, someone would need to:

1. Rebuild a V4 tree: the M2 modules need V4's `research/attacks.py` and `research/common.py` (copied under `source/v4-base/research/`; they differ from the repository's M1 files, so they must not simply overwrite them), plus the unchanged shared modules that are already in this repository.
2. Obtain the dataset (see [`docs/SETUP.md`](../../../docs/SETUP.md)) and the upstream PAVE sources, which are not copied.
3. Place the overlay as a sibling folder, set `V4_ROOT`, and use a Linux Python environment as in `source/v4-base/requirements-linux-cpu.txt`.
4. Expect different wall-clock times and efficiency numbers on other hardware; training uses no deterministic-algorithm mode, so bit-exact reruns are not claimed by the original protocol.

The steps above are an outline from reading the files, not a tested procedure. Any wrapper written later should live only in this folder, carry the header
"POST-RUN ADAPTATION, not the code executed on the VM", and be described that way.
