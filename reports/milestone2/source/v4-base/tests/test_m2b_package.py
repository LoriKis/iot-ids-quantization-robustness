"""Git-free provenance shim, portable PAVE verification and the package builder (synthetic trees only)."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from research import common

# Real `git hash-object` / `git write-tree` values for the synthetic tree below (computed with git, hard-coded here).
HELLO_BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"   # b"hello\n"
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"   # b""
X_BLOB = "c1b0730e0133447badcfd47fd144e254807b06e1"       # b"x"
TREE_WITH_EXEC = "cb42698757be97f3497101cd332724eb2fe00e75"  # e.sh recorded as 100755
TREE_ALL_644 = "f240fb4a20e05efe7047ab6877ba09f5e430b061"    # same blobs, e.sh as 100644
FILES = {"a.txt": (b"hello\n", "100644"), "e.sh": (b"", "100755"), "sub/z.txt": (b"hello\n", "100644"),
         "sub.txt": (b"x", "100644")}


def _identity(tree_id=TREE_WITH_EXEC, commit=None, **overrides):
    files = {rel: {"mode": mode, "git_blob_id": common.git_blob_id(data), "sha256": hashlib.sha256(data).hexdigest()}
             for rel, (data, mode) in FILES.items()}
    identity = {"source": {"head": "a" * 40, "branch": "work"},
                "pave": {"commit": commit or common.PIN, "tree_id": tree_id, "files": files}}
    identity.update(overrides)
    return identity


def _write_tree(root: Path, crlf_for=()):
    for rel, (data, _mode) in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\n", b"\r\n") if rel in crlf_for else data)


def test_blob_and_tree_ids_match_git():
    assert common.git_blob_id(b"hello\n") == HELLO_BLOB
    assert common.git_blob_id(b"") == EMPTY_BLOB
    assert common.git_blob_id(b"x") == X_BLOB
    entries = {rel: (mode, common.git_blob_id(data)) for rel, (data, mode) in FILES.items()}
    assert common.git_tree_id(entries) == TREE_WITH_EXEC
    # the recorded mode enters the tree id
    entries["e.sh"] = ("100644", EMPTY_BLOB)
    assert common.git_tree_id(entries) == TREE_ALL_644


def test_verify_pave_portable_accepts_synthetic_tree_and_crlf_checkout(tmp_path):
    _write_tree(tmp_path)
    assert common.verify_pave_portable(tmp_path, _identity(), use_cache=False)
    crlf = tmp_path / "crlf"
    _write_tree(crlf, crlf_for=("a.txt",))
    identity = _identity()
    identity["pave"]["files"]["a.txt"]["sha256"] = hashlib.sha256(b"hello\r\n").hexdigest()  # working-tree bytes
    assert common.verify_pave_portable(crlf, identity, use_cache=False)


@pytest.mark.parametrize("mutate,message", [
    (lambda root, ident: (root / "a.txt").write_bytes(b"changed"), "SHA-256"),
    (lambda root, ident: ident["pave"].update(tree_id="0" * 40), "tree id"),
    (lambda root, ident: ident["pave"].update(commit="1" * 40), "pin"),
    (lambda root, ident: ident["pave"]["files"]["e.sh"].update(mode="120000"), "mode"),
    (lambda root, ident: (root / "extra.txt").write_text("surprise"), "not in the recorded identity"),
    (lambda root, ident: (root / "a.txt").unlink(), "missing"),
])
def test_verify_pave_portable_rejects_differences(tmp_path, mutate, message):
    _write_tree(tmp_path)
    identity = _identity()
    mutate(tmp_path, identity)
    with pytest.raises(RuntimeError, match=message):
        common.verify_pave_portable(tmp_path, identity, use_cache=False)


def test_tree_id_without_recorded_exec_mode_is_rejected(tmp_path):
    _write_tree(tmp_path)
    with pytest.raises(RuntimeError, match="tree id"):
        common.verify_pave_portable(tmp_path, _identity(tree_id=TREE_ALL_644), use_cache=False)


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    root = tmp_path / "pkg"
    (root / "upstream" / "PAVE_artifact").mkdir(parents=True)
    _write_tree(root / "upstream" / "PAVE_artifact")
    (root / "PACKAGE_IDENTITY.json").write_text(json.dumps(_identity()), encoding="utf-8")
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "UPSTREAM", root / "upstream" / "PAVE_artifact")
    monkeypatch.setattr(common, "_PAVE_VERIFIED", {})
    return root


def test_git_shim_serves_fixed_table_without_dot_git(packaged):
    assert common._packaged()
    assert common.git("rev-parse", "HEAD", cwd=common.ROOT) == "a" * 40
    assert common.git("branch", "--show-current", cwd=common.ROOT) == "work"
    assert common.git("status", "--porcelain", cwd=common.ROOT).startswith("PACKAGED:")
    assert common.git("status", "--porcelain", "--untracked-files=all", cwd=common.ROOT).startswith("PACKAGED:")
    assert common.git("rev-parse", "HEAD", cwd=common.UPSTREAM) == common.PIN
    assert common.git("status", "--porcelain", "--untracked-files=all", cwd=common.UPSTREAM) == ""
    with pytest.raises(RuntimeError, match="not available"):
        common.git("log", "-1", cwd=common.ROOT)
    with pytest.raises(RuntimeError, match="not available"):
        common.git("diff", cwd=common.UPSTREAM)


def test_git_shim_refuses_upstream_answers_when_tree_differs(packaged):
    (common.UPSTREAM / "a.txt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        common.git("rev-parse", "HEAD", cwd=common.UPSTREAM)


def test_shim_inactive_when_dot_git_exists_or_identity_absent(packaged):
    (packaged / ".git").mkdir()
    assert not common._packaged()
    (packaged / ".git").rmdir()
    (packaged / "PACKAGE_IDENTITY.json").unlink()
    assert not common._packaged()


def test_real_checkout_unchanged_when_dot_git_exists():
    # In the working repository .git exists, so the original subprocess path runs.
    if (common.ROOT / ".git").exists():
        assert not common._packaged()


def test_no_m2_module_imports_milestone1():
    pattern = re.compile(r"^\s*(from\s+\.milestone1\b|from\s+research\.milestone1\b|import\s+research\.milestone1\b|"
                         r"from\s+research\s+import\s+.*\bmilestone1\b)", re.M)
    offenders = [p.name for p in sorted((common.ROOT / "research").glob("m2*.py")) if pattern.search(p.read_text("utf-8"))]
    assert offenders == []

# ------------------------------------------------------------------------------------------------ package builder
import importlib.util
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
def _load(name, rel):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
pkg = _load("package_m2_vm_under_test", "scripts/package_m2_vm.py")
VP = _load("verify_package_under_test", "scripts/vm/verify_package.py")
GOOD_SH = "#!/usr/bin/env bash\nset -euo pipefail\necho hi\n"
def _fake_repo(base: Path) -> Path:
    repo = base / "repo"
    files = {
        "research/a.py": "x = 1\n", "research/__pycache__/a.cpython-311.pyc": "junk", "research/notes.txt": "not python",
        "tests/t.py": "def test_x():\n    pass\n", "configs/c.json": "{}\n", "configs/secret.local.json": "{}\n",
        "scripts/vm/01_x.sh": GOOD_SH, "scripts/setup.ps1": "x", "scripts/helper.py": "y = 2\n",
        "docs/VM_RUNBOOK.md": "# runbook\n", "docs/environment-windows.json": "{}\n", "docs/other.md": "doc\n",
        "reports/r.md": "report\n", "artifacts/m2a-pilot/models/m.json": "{}\n", "artifacts/m2a-pilot/a.json": "{}\n",
        "artifacts/m2a-pilot/calibration_x.pt": "w", "artifacts/pilot-verified/x.json": "{}\n", "requirements-linux-cpu.txt": "numpy==1\n",
        "README.md": "readme\n", "model.pt": "w", "work/w.txt": "w", ".env": "SECRET=1", "output/o.txt": "o",
        "requirements-teammate-mac.txt": "mac\n", "scripts/vm/old.log": "log"}
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    (repo / ".git").mkdir()
    for name in ("verify_package.py", "check_shell.py"):
        shutil.copy2(ROOT / "scripts" / "vm" / name, repo / "scripts" / "vm" / name)
    return repo
def _build(tmp_path):
    repo = _fake_repo(tmp_path) if not (tmp_path / "repo").exists() else tmp_path / "repo"
    return repo, pkg.build(repo, tmp_path, "PKG", strict=False)
def test_excluded_is_anchored_and_pave_free():
    for rel in ("artifacts/m2a-pilot/models/a.json", "docs/environment-windows.json", "scripts/setup.ps1", "model.pt",
                "x/y/__pycache__/z.py", "a/.env", "work/x.txt", "requirements-teammate-mac.txt", "reports/a.log"):
        assert pkg.excluded(rel), rel
    for rel in ("reports/milestone2b/split_v2/split_m2.json", "scripts/vm/a.sh", "data/x.csv", "configs/m2/data.json",
                "upstream/PAVE_artifact/data/balanced_CICIoT.csv", "notes/work.md", "artifacts/m2a-pilot/a.json"):
        assert not pkg.excluded(rel), rel
def test_build_copies_allow_list_and_excludes_forbidden(tmp_path):
    repo, result = _build(tmp_path)
    assert not result["failures"], result["failures"]
    final = Path(result["final_path"])
    assert final.name == "PKG" and final.is_dir() and not any(p.name.startswith("PKG.partial") for p in tmp_path.iterdir())
    for rel in ("research/a.py", "tests/t.py", "configs/c.json", "scripts/vm/01_x.sh", "scripts/helper.py", "VM_RUNBOOK.md",
                "docs/other.md", "reports/r.md", "artifacts/m2a-pilot/a.json", "README.md", "requirements-linux-cpu.txt",
                "MANIFEST.sha256", "PACKAGE_IDENTITY.json"):
        assert (final / rel).is_file(), rel
    for rel in ("research/__pycache__", "research/notes.txt", "configs/secret.local.json", "scripts/setup.ps1",
                "docs/environment-windows.json", "artifacts/m2a-pilot/models", "artifacts/m2a-pilot/calibration_x.pt",
                "artifacts/pilot-verified", "model.pt", "work", ".env", "output", ".git", "requirements-teammate-mac.txt",
                "scripts/vm/old.log"):
        assert not (final / rel).exists(), rel
    assert not any("__pycache__" in str(p) for p in final.rglob("*"))
    assert "Trained weights are generated on the VM" in (final / "PACKAGE_IDENTITY.json").read_text(encoding="utf-8")
def test_never_overwrites_and_picks_next_suffix(tmp_path):
    repo, first = _build(tmp_path)
    before = (Path(first["final_path"]) / "MANIFEST.sha256").read_bytes()
    marker = Path(first["final_path"]) / "keep.txt"
    marker.write_text("mine")
    second = pkg.build(repo, tmp_path, "PKG", strict=False)
    third = pkg.build(repo, tmp_path, "PKG", strict=False)
    assert Path(second["final_path"]).name == "PKG-v2" and Path(third["final_path"]).name == "PKG-v3"
    assert marker.read_text() == "mine" and (Path(first["final_path"]) / "MANIFEST.sha256").read_bytes() == before
def test_manifest_format_verify_and_tamper(tmp_path):
    _repo, result = _build(tmp_path)
    final = Path(result["final_path"])
    raw = (final / "MANIFEST.sha256").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode().splitlines()
    assert lines == sorted(lines, key=lambda l: l[66:].encode())
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S.*", l) for l in lines) and not any("MANIFEST.sha256" in l[66:] for l in lines)
    clean = VP.check_manifest(final)
    assert not (clean["missing"] or clean["mismatched"] or clean["unlisted"] or clean["symlinks"])
    (final / "research/a.py").write_text("x = 2\n", newline="\n")
    (final / "extra.txt").write_text("new")
    (final / "tests/t.py").unlink()
    (final / "output").mkdir()
    (final / "output/ignored.txt").write_text("run output")  # output/ is outside the manifest by design
    bad = VP.check_manifest(final)
    assert bad["mismatched"] == ["research/a.py"] and bad["unlisted"] == ["extra.txt"] and bad["missing"] == ["tests/t.py"]
    assert VP.main(["--root", str(final), "--manifest-only"]) == 1
def test_symlink_is_detected(tmp_path):
    _repo, result = _build(tmp_path)
    final = Path(result["final_path"])
    try:
        (final / "link.txt").symlink_to(final / "README.md")
    except (OSError, NotImplementedError):
        pytest.skip("this OS cannot create symlinks")
    assert VP.check_manifest(final)["symlinks"] == ["link.txt"]
    assert pkg.find_links(final) == ["link.txt"]
def test_absolute_path_scanner_finds_planted_strings(tmp_path):
    drive_style = "C" + ":" + "\\Users\\" + "x" + "\\data"
    user_style = "Users" + "/" + "lo" + "ran" + "/file"
    (tmp_path / "research").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "research/active.py").write_text(f"P = r'{drive_style}'\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/also.py").write_text(f"P = '{user_style}'\n")
    (tmp_path / "docs/old.md").write_text(f"see {drive_style}\n")
    (tmp_path / "research/clean.py").write_text("P = 'relative/path'\n")
    hits = pkg.scan_windows_paths(tmp_path)
    assert hits["active"] == ["research/active.py", "scripts/also.py"] and hits["historical"] == ["docs/old.md"]
def test_repository_code_has_no_absolute_windows_paths():
    files = [p.relative_to(ROOT).as_posix() for area in ("tests", "scripts", "research") for p in (ROOT / area).rglob("*.py")
             if "__pycache__" not in p.parts]
    assert files
    assert pkg.scan_windows_paths(ROOT, files)["active"] == []
def test_crlf_checker_catches_crlf(tmp_path):
    (tmp_path / "scripts/vm").mkdir(parents=True)
    (tmp_path / "scripts/vm/a.sh").write_bytes(b"#!/usr/bin/env bash\r\nset -euo pipefail\r\n")
    (tmp_path / "scripts/vm/b.sh").write_bytes(b"#!/usr/bin/env bash\nset -euo pipefail\n")
    (tmp_path / "VM_RUNBOOK.md").write_bytes(b"ok\n")
    assert pkg.find_crlf(tmp_path) == ["scripts/vm/a.sh"]
def test_shell_checker_flags_missing_strict_mode_and_unbalanced_if(tmp_path):
    cs = _load("check_shell_under_test", "scripts/vm/check_shell.py")
    bad = tmp_path / "bad.sh"
    bad.write_text('#!/usr/bin/env bash\nif true; then\necho "x\n', newline="\n")
    problems = cs.static_problems(bad)
    assert any("pipefail" in p for p in problems) and any("if/fi" in p for p in problems) and any("quotes" in p for p in problems)
    good = tmp_path / "good.sh"
    good.write_text(GOOD_SH + "if true; then echo a; fi\n", newline="\n")
    assert cs.static_problems(good) == []
def test_runbook_command_checker_resolves_and_rejects(tmp_path):
    rc = _load("check_runbook_under_test", "scripts/vm/check_runbook_commands.py")
    (tmp_path / "scripts/vm").mkdir(parents=True)
    (tmp_path / "scripts/vm/01_x.sh").write_text(GOOD_SH + "# --flag\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research/m2b_campaign.py").write_text('sub.add_parser("run")\nadd_argument("--phase")\n')
    ok = ["bash scripts/vm/01_x.sh", "WORKERS=2 bash scripts/vm/01_x.sh --flag", ".venv/bin/python -m research.m2b_campaign run --phase validation",
          "nohup bash scripts/vm/01_x.sh > output/log.txt 2>&1 &", "tail -f output/log.txt", "kill -TERM <pid>"]
    bad = ["bash scripts/vm/99_missing.sh", "bash scripts/vm/01_x.sh --nope", ".venv/bin/python -m research.m2b_campaign nosuch",
           ".venv/bin/python -m research.m2b_campaign run --bogus", ".venv/bin/python -m research.absent run", "curl http://x"]
    assert [rc.check(rc.clean(c), tmp_path) for c in ok] == [None] * len(ok)
    assert all(rc.check(rc.clean(c), tmp_path) for c in bad)
