# Attribution and licensing

This project builds on [PAVE_artifact](https://github.com/robotumbel/PAVE_artifact),
revision `e73911bbdb241f3fb6c5de346139010c4aadefd4`, released by the PAVE authors.
The accompanying paper is [PAVE, DOI 10.1016/j.hcc.2026.100426](https://doi.org/10.1016/j.hcc.2026.100426).

The upstream repository labels its license MIT, with the literal copyright notice
`Copyright (c) 2026 [Author]`. The [unmodified license text](licenses/PAVE-MIT.txt)
is included here, including that placeholder; we do not invent a replacement
copyright holder.

Upstream code is restored into the ignored `upstream/PAVE_artifact/` directory.
The Milestone 1 runner imports its Standard Transformer, preprocessing, training,
sampling and attack functions, adding explicitly documented instrumentation.
The saved training adapter is derived from that code and is covered by the
retained upstream notice. Source hashes are in the baseline configuration.

Our group developed the orchestration, corrected research pipeline, verification,
configurations, documentation and result packaging in this repository. PAVE's
MIT license does not by itself license the group's original contributions.
The group has not yet chosen an overall project license; resolve that decision
before advertising a reuse license for the entire project.

The code license does not establish rights to redistribute CICIoT dataset payloads
or the publisher's paper. Those payloads are excluded from Git; provenance and
acquisition instructions are retained.
