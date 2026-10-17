# PAVE Adversarial Robustness Research

Investigation of the adversarial robustness of a Transformer-based network
intrusion detection system using the PAVE framework and the CICIoT2023 dataset.

## Upstream Repository

**Source:** https://github.com/robotumbel/PAVE_artifact.git  
**Location:** `upstream/PAVE_artifact/` — **READ-ONLY**. No researcher script
may create, edit, or delete files inside this directory.

## Directory Layout

```
project_root/
├── upstream/
│   └── PAVE_artifact/          ← cloned upstream (read-only)
├── experiments/
│   └── milestone1/             ← researcher wrapper scripts
├── results/
│   └── milestone1/             ← captured evaluation output
├── notes/
│   ├── milestone1_environment.txt   ← git commit hash + Python version
│   ├── milestone1_packages.txt      ← pip freeze snapshot
│   └── pave_reproduction_notes.md  ← discovery report + audit notes
└── README.md
```

## Quickstart (Milestone 1)

```bash
# 1. Clone (already done if you are reading this)
git clone https://github.com/robotumbel/PAVE_artifact.git upstream/PAVE_artifact

# 2. Bootstrap environment
bash experiments/milestone1/setup_environment.sh

# 3. Run full pipeline
bash experiments/milestone1/run_all.sh
```

Results land in `results/milestone1/`. Provenance logs land in `notes/`.

## Upstream Attribution

The PAVE framework is the work of its original authors. This repository
contains only wrapper scripts and captured outputs; the upstream code is
unmodified. See `upstream/PAVE_artifact/` for the original license.

## Pinned Commit

See `notes/milestone1_environment.txt` for the exact commit hash used.
