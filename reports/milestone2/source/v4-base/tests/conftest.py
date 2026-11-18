"""Local training guard: with M2_LOCAL_NO_TRAIN=1 no test may train, build a full-size model or run FGSM/PGD evaluations."""
import os

import pytest

from m2_guard_support import install_local_guard


@pytest.fixture(autouse=True)
def _m2_local_no_train(request, monkeypatch):
    if os.environ.get("M2_LOCAL_NO_TRAIN") == "1" and request.node.get_closest_marker("vm_only") is None:
        install_local_guard(monkeypatch)
    yield

@pytest.fixture(autouse=True)
def _packaged_git(monkeypatch):
    """In the git-free VM package the Milestone 1 modules' own `git` helper cannot run `git`; serve the fixed table.

    Only active when PACKAGE_IDENTITY.json exists and .git does not (never in the source repository).
    """
    import sys
    from research import common
    if common._packaged():
        for name in ("research.milestone1", "research.artifact_baseline", "research.paper_described_baseline"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "git"):
                monkeypatch.setattr(module, "git", common.git)
    yield
