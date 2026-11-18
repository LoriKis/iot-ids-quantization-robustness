"""Pure preflight/smoke helpers with injected inputs (no torch quantization, no model, no real-data inference)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import m2b_plan as mp
from research import m2b_preflight as pf

ROOT = Path(__file__).resolve().parents[1]
CPUINFO = "processor\t: 0\nflags\t\t: fpu sse avx avx2 fma\n"


def status(checks, name):
    return next(c["status"] for c in checks if c["name"] == name)


def test_recommended_workers_formula():
    assert pf.recommended_workers(8, 32) == 4
    assert pf.recommended_workers(16, 16) == 5
    assert pf.recommended_workers(1, 4) == 1
    assert pf.recommended_workers(2, 2.0) == 1


def test_platform_checks():
    ok = pf.check_platform("Linux", "x86_64", "3.11.9", CPUINFO, "2.35")
    assert status(ok, "platform.os") == status(ok, "platform.python") == status(ok, "platform.avx2") == pf.PASS
    assert status(ok, "platform.avx512") == pf.WARN
    bad = pf.check_platform("Windows", "AMD64", "3.12.1", "", None)
    assert status(bad, "platform.os") == pf.FAIL and status(bad, "platform.python") == pf.FAIL and status(bad, "platform.avx2") == pf.FAIL
    assert pf.summarize(bad) == pf.FAIL and pf.summarize(ok) == pf.WARN == pf.WARN


def test_version_pins():
    pins = pf.parse_requirements("# c\n--extra-index-url x\ntorch==2.5.1+cpu\nnumpy==1.26.4  # pin\nscikit-learn==1.5.2\n")
    assert pins == {"torch": "2.5.1+cpu", "numpy": "1.26.4", "scikit-learn": "1.5.2"}
    installed = {p: None for p in pf.IMPORT_NAMES}
    installed.update(torch="2.5.1+cpu", numpy="1.26.3", pandas="2.2.3")
    checks = pf.check_versions(pins, installed)
    assert status(checks, "import.torch") == pf.PASS and status(checks, "import.numpy") == pf.FAIL
    assert status(checks, "import.pandas") == pf.WARN and status(checks, "import.scipy") == pf.FAIL


def test_dataset_facts():
    header = ["a", "Label"]
    good = pf.check_dataset_facts(pf.DATA_SIZE_BYTES, "x", header, pf.DATA_ROWS, "x", header)
    assert {c["status"] for c in good} == {pf.PASS}
    bad = pf.check_dataset_facts(10, "y", ["b"], 5, "x", header)
    assert {c["status"] for c in bad} == {pf.FAIL}


def test_resources():
    gb = 10**9
    ok = pf.check_resources(50 * gb, 32 * gb, 8, 2, 2, ["0.1", "0.1", "0.1"])
    assert {c["status"] for c in ok} == {pf.PASS}
    low = pf.check_resources(1 * gb, 4 * gb, 2, 2, 2)
    assert status(low, "resources.disk") == pf.FAIL and status(low, "resources.ram") == pf.FAIL and status(low, "resources.cores") == pf.FAIL
    assert status(pf.check_resources(50 * gb, 20 * gb, 64, 8, 2), "resources.ram") == pf.FAIL  # 8 workers x 3 GB = 24 GB


def test_threads_note():
    assert pf.check_threads(8, 4, 8)["status"] == pf.WARN
    assert pf.check_threads(4, 4, 4)["status"] == pf.PASS


def test_probe_writable(tmp_path):
    checks = pf.probe_writable(tmp_path / "out", (tmp_path / "out" / "logs",))
    assert [c["status"] for c in checks] == [pf.PASS, pf.PASS]
    assert list((tmp_path / "out").glob(".preflight-probe*")) == []
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert pf.probe_writable(blocker / "sub")[0]["status"] == pf.FAIL


def test_script_check_without_bash(tmp_path):
    (tmp_path / "scripts" / "vm").mkdir(parents=True)
    (tmp_path / "scripts" / "vm" / "a.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    assert pf.check_scripts(tmp_path, bash="")["status"] == pf.WARN
    assert pf.check_scripts(tmp_path / "none", bash="")["status"] == pf.WARN


def test_smoke_plan_is_bounded_and_covers_six_sources():
    cfg = json.loads((ROOT / "configs" / "m2b" / "campaign_v2.json").read_text(encoding="utf-8"))
    smoke = pf.smoke_config(cfg)
    assert cfg["e4"]["epsilons"] != [0.05] and smoke["e4"]["epsilons"] == [0.05]
    assert smoke["attack_row_policy"]["validation"] == {"per_class": 10}
    units = pf.filter_smoke_units(mp.build_plan(smoke, [1], "validation", mp.BASE_STAGES, pgd_budget="pgd20"))
    attacks = [u for u in units if u.kind == "attack"]
    assert {u.params["source"] for u in attacks} == {"fp32_direct", "bpda_int8", "transfer_fp32_to_int8", "transfer_fp32_to_int8_targetsel",
                                                     "surrogate_fakequant", "surrogate_fakequant_targetsel"}
    assert {(u.params["attack"], float(u.params["eps"])) for u in attacks} == {("fgsm", 0.01), ("fgsm", 0.05), ("pgd20", 0.05)}
    assert not any(u.seed != 1 for u in units) and len({u.id for u in units}) == len(units)
    assert sum(u.kind == "e4" for u in units) == 4 and sum(u.kind == "train" for u in units) == 2


def test_no_not_implemented_left_in_m2b_modules():
    for path in (ROOT / "research").glob("m2b_*.py"):
        assert "raise NotImplementedError" not in path.read_text(encoding="utf-8"), path.name
