"""make_manifest and verify_overlay on temporary overlay and fake V4 folders only. No torch, no research.*."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from m2_deadline import make_manifest, manifest as mf, verify_overlay


def _w(path, text):
    path.write_bytes(text.encode("utf-8"))  # LF on every platform


def build_v4(root):
    (root / "research").mkdir(parents=True)
    _w((root / "research" / "a.py"), "print('a')\n")
    _w((root / "README.md"), "v4\n")
    entries = {rel: mf.sha256_file(root / rel) for rel in ("research/a.py", "README.md")}
    (root / "MANIFEST.sha256").write_bytes(mf.manifest_text(entries).encode("utf-8"))
    (root / ".venv").mkdir()
    _w((root / ".venv" / "ignored.txt"), "x\n")
    (root / "output").mkdir()
    _w((root / "output" / "ignored.txt"), "x\n")
    return root


def build_overlay(root):
    (root / "configs").mkdir(parents=True)
    (root / "m2_deadline").mkdir()
    (root / "artifacts" / "m2-deadline-v1").mkdir(parents=True)
    _w((root / "configs" / "m2_deadline_v1.json"), '{"run_id":"x"}\n')
    _w((root / "m2_deadline" / "mod.py"), "X = 1\n")
    _w((root / "HANDOFF.md"), "# handoff\n")
    (root / "artifacts" / "m2-deadline-v1" / "big.bin").write_bytes(b"\x00" * 10)
    return root


def ok_runner(*a, **k):
    return SimpleNamespace(returncode=0, stdout="RESULT: PASS (manifest only)\n", stderr="")


@pytest.fixture
def pair(tmp_path):
    return build_overlay(tmp_path / "overlay"), build_v4(tmp_path / "v4")


def made(pair):
    overlay, v4 = pair
    assert make_manifest.main(["--overlay", str(overlay), "--v4-root", str(v4)]) == 0
    return overlay, v4


def verify(pair, runner=ok_runner):
    overlay, v4 = pair
    return verify_overlay.main(["--overlay", str(overlay), "--v4-root", str(v4)], runner=runner)


def v4_state(v4):
    state = {}
    for dirpath, _dirs, files in os.walk(str(v4)):
        for name in files:
            p = os.path.join(dirpath, name)
            state[p] = (os.stat(p).st_mtime_ns, mf.sha256_file(p))
    return state


def test_make_and_verify_ok_and_v4_untouched(pair, capsys):
    overlay, v4 = pair
    before = v4_state(v4)
    made(pair)
    assert verify(pair) == 0
    assert capsys.readouterr().out.strip().endswith("OK")
    assert v4_state(v4) == before


def test_manifest_format_sorted_lf_and_skips(pair):
    overlay, _v4 = made(pair)
    data = (overlay / "OVERLAY_MANIFEST.sha256").read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    lines = data.decode().splitlines()
    rels = [line[66:] for line in lines]
    assert rels == sorted(rels)
    assert all(len(line[:64]) == 64 and line[64:66] == "  " for line in lines)
    assert "OVERLAY_MANIFEST.sha256" not in rels
    assert not any(r.startswith("artifacts/") for r in rels)
    assert {"OVERLAY_IDENTITY.json", "HANDOFF.md", "m2_deadline/mod.py", "configs/m2_deadline_v1.json"} <= set(rels)


def test_identity_content(pair):
    overlay, v4 = made(pair)
    ident = json.loads((overlay / "OVERLAY_IDENTITY.json").read_text(encoding="utf-8"))
    assert ident["expects_v4_manifest_sha256"] == mf.sha256_file(v4 / "MANIFEST.sha256")
    assert ident["config_sha256"] == mf.sha256_file(overlay / "configs" / "m2_deadline_v1.json")


def test_tampered_file_fails(pair, capsys):
    overlay, _v4 = made(pair)
    _w((overlay / "m2_deadline" / "mod.py"), "X = 2\n")
    assert verify(pair) == 3
    assert "mismatched: m2_deadline/mod.py" in capsys.readouterr().out


def test_extra_file_fails(pair, capsys):
    overlay, _v4 = made(pair)
    _w((overlay / "extra.txt"), "hi\n")
    assert verify(pair) == 3
    assert "unlisted: extra.txt" in capsys.readouterr().out


def test_missing_file_fails(pair, capsys):
    overlay, _v4 = made(pair)
    (overlay / "HANDOFF.md").unlink()
    assert verify(pair) == 3
    assert "missing: HANDOFF.md" in capsys.readouterr().out


def test_artifacts_and_pycache_do_not_count_in_verify(pair):
    overlay, _v4 = made(pair)
    _w((overlay / "artifacts" / "m2-deadline-v1" / "new.log"), "x\n")
    (overlay / "m2_deadline" / "__pycache__").mkdir()
    (overlay / "m2_deadline" / "__pycache__" / "mod.cpython-39.pyc").write_bytes(b"\x00")
    assert verify(pair) == 0


def test_crlf_after_manifest_fails(pair, capsys):
    overlay, _v4 = made(pair)
    (overlay / "HANDOFF.md").write_bytes(b"# handoff\r\n")
    assert verify(pair) == 3
    assert "crlf: HANDOFF.md" in capsys.readouterr().out


def test_make_manifest_refuses_crlf(pair, capsys):
    overlay, v4 = pair
    (overlay / "HANDOFF.md").write_bytes(b"# handoff\r\n")
    assert make_manifest.main(["--overlay", str(overlay), "--v4-root", str(v4)]) == 3
    assert "HANDOFF.md" in capsys.readouterr().err
    assert not (overlay / "OVERLAY_MANIFEST.sha256").exists()


@pytest.mark.parametrize("cache", ["__pycache__", ".pytest_cache"])
def test_make_manifest_refuses_stray_cache(pair, capsys, cache):
    overlay, v4 = pair
    (overlay / "m2_deadline" / cache).mkdir()
    assert make_manifest.main(["--overlay", str(overlay), "--v4-root", str(v4)]) == 3
    assert cache in capsys.readouterr().err
    assert not (overlay / "OVERLAY_IDENTITY.json").exists()


def test_make_manifest_refuses_symlink(pair):
    overlay, v4 = pair
    try:
        os.symlink(str(overlay / "HANDOFF.md"), str(overlay / "link.md"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")
    assert make_manifest.main(["--overlay", str(overlay), "--v4-root", str(v4)]) == 3


def test_usage_errors_exit_2(tmp_path, pair):
    overlay, v4 = pair
    assert make_manifest.main(["--overlay", str(tmp_path / "nope"), "--v4-root", str(v4)]) == 2
    assert verify(pair) == 2  # no manifest written yet


def test_identity_mismatch_fails(pair, capsys):
    overlay, v4 = made(pair)
    (v4 / "MANIFEST.sha256").write_bytes(b"")  # a different V4 manifest than the one recorded
    assert verify(pair) == 3
    assert "differs from the expected" in capsys.readouterr().out


def test_v4_unlisted_is_warning_only(pair, capsys):
    overlay, v4 = made(pair)
    (v4 / "research" / "__pycache__").mkdir()
    (v4 / "research" / "__pycache__" / "a.pyc").write_bytes(b"\x00")
    assert verify(pair) == 0
    out = capsys.readouterr().out
    assert "WARN" in out and "research/__pycache__/a.pyc" in out and out.strip().endswith("OK")


def test_v4_mismatch_and_missing_fail(pair, capsys):
    overlay, v4 = made(pair)
    _w((v4 / "README.md"), "changed\n")
    assert verify(pair) == 3
    assert "V4 mismatched: README.md" in capsys.readouterr().out
    (v4 / "README.md").unlink()
    assert verify(pair) == 3
    assert "V4 missing: README.md" in capsys.readouterr().out


def test_second_check_warn_when_only_unlisted(pair, capsys):
    made(pair)

    def runner(*a, **k):
        return SimpleNamespace(returncode=1, stdout="manifest: 2 listed\n  unlisted: stray.txt\nRESULT: FAIL\n", stderr="")

    assert verify(pair, runner) == 0
    assert "reports only unlisted files" in capsys.readouterr().out


def test_second_check_failure_exits_3(pair, capsys):
    made(pair)

    def runner(*a, **k):
        return SimpleNamespace(returncode=1, stdout="  mismatched: README.md\nRESULT: FAIL\n", stderr="")

    assert verify(pair, runner) == 3
    assert "verify_package --manifest-only: FAIL" in capsys.readouterr().out


def test_second_check_unparseable_failure_exits_3(pair):
    made(pair)

    def runner(*a, **k):
        return SimpleNamespace(returncode=1, stdout="garbage\n", stderr="")

    assert verify(pair, runner) == 3


def test_second_check_skipped_cases(pair, capsys):
    made(pair)

    def import_error(*a, **k):
        return SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'x'")

    def not_found(*a, **k):
        raise FileNotFoundError("python")

    for runner in (import_error, not_found):
        assert verify(pair, runner) == 0
        assert "SKIPPED" in capsys.readouterr().out


def test_second_check_skipped_when_script_missing_and_real_runner(pair, capsys):
    made(pair)  # fake V4 has no scripts/vm/verify_package.py; the real subprocess runner is not started
    assert verify(pair, runner=None) == 0
    assert "SKIPPED" in capsys.readouterr().out
