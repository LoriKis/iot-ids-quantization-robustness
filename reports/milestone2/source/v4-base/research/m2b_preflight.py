"""M2B preflight and smoke (DESIGN 4.12, 4.15). Neither stage touches the test partition.

`preflight` runs no training and no inference on the real dataset: platform, pinned imports, the INT8 backend (a random
8x8 Linear and a tiny QAT prepare+convert without optimizer steps), dataset identity, PAVE identity, split audit, config
hashes, fingerprints, resources and writable paths. It writes output/<run>/preflight.json including recommended_workers.

`smoke` is the bounded VM check: train and validation subsets only, 2-epoch V1/V3, PTQ, clean, a few attack units, a mini
step 0, an optional QAT/FP32-FT run, an interrupt-and-resume check, a fingerprint-refusal check and a report build. It runs
in output/<run>-smoke/ whose state kind is "smoke" (a smoke state is never reused by `run`).

Each check returns {"name", "status": PASS|WARN|FAIL, "detail"}. The pure check functions take injected inputs so they can be
tested on any machine; the functions that need Linux or torch quantization are pending VM verification.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .m2b_io import durable_write_json, sha256_file, utc_now

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
DATA_SIZE_BYTES = 5_444_531
DATA_ROWS = 29_997
MIN_DISK_GB, MIN_RAM_GB = 5.0, 8.0
RAM_PER_WORKER_GB = 3
PYTHON_SERIES = (3, 11)
REQUIREMENTS = "requirements-linux-cpu.txt"
IMPORT_NAMES = {"torch": "torch", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
                "matplotlib": "matplotlib", "psutil": "psutil", "pytest": "pytest"}
SMOKE_PER_CLASS = {"train": 100, "validation": 20, "attack": 10}
SMOKE_FGSM_EPS = (0.01, 0.05)
SMOKE_PGD_EPS = (0.05,)


def check(name, status, detail) -> dict:
    return {"name": name, "status": status, "detail": detail}


# ------------------------------------------------------------------------------------------ pure checks
def recommended_workers(cores: int, ram_gb: float) -> int:
    """max(1, min(cores // 2, ram_gb // 3)): two threads per training worker and about 3 GB each."""
    return max(1, min(int(cores) // 2, int(ram_gb) // RAM_PER_WORKER_GB))


def check_platform(system, machine, python_version, cpuinfo_text, glibc=None) -> list:
    out = [check("platform.os", PASS if system == "Linux" else FAIL, f"{system} {machine}; supported: Linux x86_64"),
           check("platform.arch", PASS if machine in ("x86_64", "AMD64") else FAIL, machine)]
    major_minor = tuple(int(p) for p in python_version.split(".")[:2])
    out.append(check("platform.python", PASS if major_minor == PYTHON_SERIES else FAIL,
                     f"Python {python_version}; the pinned stack is Python {PYTHON_SERIES[0]}.{PYTHON_SERIES[1]}.x"))
    flags = set()
    for line in (cpuinfo_text or "").splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    out.append(check("platform.avx2", PASS if "avx2" in flags else FAIL,
                     "avx2 present" if "avx2" in flags else "avx2 missing or /proc/cpuinfo unreadable: the fbgemm INT8 kernels need it"))
    out.append(check("platform.avx512", PASS if any(f.startswith("avx512") for f in flags) else WARN,
                     "avx512 present" if any(f.startswith("avx512") for f in flags) else "no avx512 (fbgemm still runs; timings differ)"))
    out.append(check("platform.glibc", PASS if glibc else WARN, glibc or "glibc version not detected"))
    return out


def parse_requirements(text: str) -> dict:
    """{package: pinned version} for `name==version` lines (comments, options and blank lines skipped)."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def check_versions(pins: dict, installed: dict) -> list:
    """installed: {package: version or None}. Only the packages in IMPORT_NAMES are compared."""
    out = []
    for package in IMPORT_NAMES:
        want, have = pins.get(package), installed.get(package)
        if have is None:
            out.append(check(f"import.{package}", FAIL, "not importable"))
        elif want is None:
            out.append(check(f"import.{package}", WARN, f"{have} installed; no pin in {REQUIREMENTS}"))
        elif have == want:
            out.append(check(f"import.{package}", PASS, have))
        else:
            out.append(check(f"import.{package}", FAIL, f"installed {have}, pinned {want}"))
    return out


def check_dataset_facts(size, sha, header, rows, expected_sha, expected_header) -> list:
    return [check("data.size", PASS if size == DATA_SIZE_BYTES else FAIL, f"{size} bytes (expected {DATA_SIZE_BYTES})"),
            check("data.sha256", PASS if sha == expected_sha else FAIL, f"{sha} (frozen {expected_sha})"),
            check("data.header", PASS if header == expected_header else FAIL,
                  "header equals the 39 feature names plus Label" if header == expected_header else f"header differs: {header[:3]}..."),
            check("data.rows", PASS if rows == DATA_ROWS else FAIL, f"{rows} rows (expected {DATA_ROWS})")]


def check_resources(disk_free_bytes, ram_total_bytes, cores, workers, threads_per_worker, loadavg=None) -> list:
    disk_gb, ram_gb = disk_free_bytes / 1e9, ram_total_bytes / 1e9
    need_ram = max(MIN_RAM_GB, RAM_PER_WORKER_GB * workers)
    return [check("resources.disk", PASS if disk_gb >= MIN_DISK_GB else FAIL, f"{disk_gb:.1f} GB free (need {MIN_DISK_GB:.0f} GB)"),
            check("resources.ram", PASS if ram_gb >= need_ram else FAIL,
                  f"{ram_gb:.1f} GB total (need {need_ram:.0f} GB for {workers} worker(s))"),
            check("resources.cores", PASS if cores >= workers * threads_per_worker else FAIL,
                  f"{cores} cores for {workers} worker(s) x {threads_per_worker} thread(s)"),
            check("resources.load", PASS, f"1/5/15-minute load {loadavg}" if loadavg else "load average not available")]


def check_threads(nproc, physical_cores, torch_threads) -> dict:
    smt = physical_cores is not None and nproc > physical_cores
    return check("threads.smt", WARN if smt else PASS,
                 f"nproc {nproc}, physical cores {physical_cores}, torch default threads {torch_threads}"
                 + ("; SMT is on, timings of two workers can interfere (efficiency runs use one thread per child)" if smt else ""))


def probe_writable(folder, extra_dirs=()) -> list:
    """Create, fsync, atomically rename and delete a probe file in each directory."""
    out = []
    for d in (folder, *extra_dirs):
        d = Path(d)
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp, final = d / ".preflight-probe.tmp", d / ".preflight-probe"
            with open(tmp, "wb") as handle:
                handle.write(b"probe")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, final)
            final.unlink()
            out.append(check(f"writable.{d.name}", PASS, f"{d}: create, fsync, atomic rename and delete work"))
        except OSError as error:
            out.append(check(f"writable.{d.name}", FAIL, f"{d}: {error}"))
    return out


def check_scripts(root, bash=None) -> dict:
    scripts = sorted((Path(root) / "scripts" / "vm").glob("*.sh"))
    bash = bash if bash is not None else shutil.which("bash")
    if not scripts:
        return check("scripts.bash_n", WARN, "no scripts/vm/*.sh found")
    if not bash:
        return check("scripts.bash_n", WARN, "bash not found; syntax not checked")
    bad = []
    for script in scripts:
        result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        if result.returncode != 0:
            bad.append(f"{script.name}: {result.stderr.strip()[:200]}")
    return check("scripts.bash_n", FAIL if bad else PASS, "; ".join(bad) if bad else f"bash -n passed for {len(scripts)} script(s)")


def summarize(checks) -> str:
    states = {c["status"] for c in checks}
    return FAIL if FAIL in states else (WARN if WARN in states else PASS)


# ------------------------------------------------------------------------------------------ environment-dependent checks
def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _glibc():
    try:
        return platform.libc_ver()[1] or None
    except Exception:  # noqa: BLE001
        return None


def _installed_versions() -> dict:
    import importlib
    versions = {}
    for package, module in IMPORT_NAMES.items():
        try:
            versions[package] = importlib.import_module(module).__version__
        except Exception:  # noqa: BLE001 - reported as not importable
            versions[package] = None
    return versions


def check_int8_backend() -> list:
    """fbgemm must be supported; an 8x8 Linear is quantized and run (quantized::linear in the profiler trace) and a tiny
    QAT prepare+convert runs without optimizer steps. No fallback to another engine. Pending VM verification."""
    out = []
    try:
        import torch
        import torch.nn as nn
        from torch.ao import quantization as aq
        from torch.profiler import ProfilerActivity, profile
    except ImportError as error:
        return [check("int8.import", FAIL, str(error))]
    engines = list(torch.backends.quantized.supported_engines)
    if "fbgemm" not in engines:
        return [check("int8.fbgemm", FAIL, f"fbgemm not in supported engines {engines}; the frozen workflow needs fbgemm")]
    torch.backends.quantized.engine = "fbgemm"
    out.append(check("int8.fbgemm", PASS, f"supported engines {engines}; engine set to fbgemm"))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant, self.fc, self.dequant = aq.QuantStub(), nn.Linear(8, 8), aq.DeQuantStub()

        def forward(self, x):
            return self.dequant(self.fc(self.quant(x)))
    try:
        generator = torch.Generator().manual_seed(0)
        model = Tiny().eval()
        model.qconfig = aq.get_default_qconfig("fbgemm")
        aq.prepare(model, inplace=True)
        for _ in range(4):
            model(torch.randn(16, 8, generator=generator))
        aq.convert(model, inplace=True)
        probe = torch.randn(4, 8, generator=generator)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.inference_mode():
                result = model(probe)
        names = {e.key for e in prof.key_averages()}
        ok = bool(torch.isfinite(result).all()) and "quantized::linear" in names
        out.append(check("int8.ptq_linear", PASS if ok else FAIL,
                         "random 8x8 Linear: PTQ converted, finite output, quantized::linear in the profiler trace" if ok
                         else f"finite={bool(torch.isfinite(result).all())}; quantized ops seen: {sorted(n for n in names if 'quant' in n)}"))
    except Exception as error:  # noqa: BLE001
        out.append(check("int8.ptq_linear", FAIL, f"{type(error).__name__}: {error}"))
    try:
        qat = Tiny().train()
        qat.qconfig = aq.get_default_qat_qconfig("fbgemm")
        aq.prepare_qat(qat, inplace=True)
        qat(torch.randn(4, 8))  # one forward for the observers; no optimizer step
        qat.eval()
        aq.convert(qat, inplace=True)
        out.append(check("int8.qat_convert", PASS, "tiny QAT prepare + convert works (no optimizer steps)"))
    except Exception as error:  # noqa: BLE001
        out.append(check("int8.qat_convert", FAIL, f"{type(error).__name__}: {error}; the QAT stage would be blocked"))
    return out


def check_pave(root) -> list:
    from . import common
    identity = Path(root) / common.IDENTITY_NAME
    try:
        if identity.is_file():
            common.verify_pave_portable(Path(root) / "upstream" / "PAVE_artifact", json.loads(identity.read_text(encoding="utf-8")),
                                        use_cache=False)
            detail = f"PAVE tree id and file hashes equal {common.IDENTITY_NAME}"
        else:
            common.verify_upstream()
            detail = f"{common.IDENTITY_NAME} absent: verified through git at the pinned revision"
        model_sha = common.sha256(Path(root) / "upstream" / "PAVE_artifact" / "code" / "models" / "standard_transformer.py")
        return [check("pave.identity", PASS, detail), check("pave.model_file", PASS, f"standard_transformer.py sha256 {model_sha}")]
    except Exception as error:  # noqa: BLE001
        return [check("pave.identity", FAIL, f"{type(error).__name__}: {error}")]


def check_dataset(root, data_cfg) -> list:
    path = Path(root) / data_cfg["dataset"]["path"]
    try:
        size, sha = path.stat().st_size, sha256_file(path)
        with open(path, "rb") as handle:
            header = handle.readline().decode("utf-8").rstrip("\r\n").split(",")
            rows = sum(1 for _ in handle)
    except OSError as error:
        return [check("data.file", FAIL, f"{path}: {error}")]
    expected = list(data_cfg["dataset"]["feature_names"]) + ["Label"]
    return check_dataset_facts(size, sha, header, rows, data_cfg["dataset"]["sha256"], expected)


def check_split(root, data_cfg_path, data_cfg) -> list:
    from . import m2_split
    out = []
    try:
        cfg = m2_split.load_config(data_cfg_path)
        split_dir = Path(root) / cfg["split"]["output_dir"]
        audit = m2_split.audit_existing(cfg, split_dir, config_path=data_cfg_path)
        freeze = audit.get("freeze")
        out.append(check("split.audit", PASS, f"split_v2 audited read-only: manifest {audit['manifest_sha256']}, rows {audit['partition_rows']}, "
                                              f"0 shared groups, all classes in every partition"))
        out.append(check("split.freeze", PASS if freeze else FAIL,
                         f"split_freeze.json: {freeze['artifacts_verified']} artifact hashes verified" if freeze else "split_freeze.json missing"))
    except Exception as error:  # noqa: BLE001
        out.append(check("split.audit", FAIL, f"{type(error).__name__}: {error}"))
    original = any(Path(root, "reports", "milestone2a").glob("**/split_m2.json"))
    out.append(check("split.original_present", PASS if original else WARN,
                     "original M2A split artifacts present" if original else "original M2A split not found"))
    return out


def check_configs(root, campaign_path) -> list:
    from .m2b_campaign import load_campaign
    out = []
    try:
        cfg = load_campaign(campaign_path, root)
        bad = [n for n, e in cfg["siblings"].items() if sha256_file(Path(root) / e["path"]) != e["sha256"]]
        out.append(check("config.siblings", FAIL if bad else PASS,
                         f"sibling configuration hash differs: {bad}" if bad else f"campaign_v2.json valid; {len(cfg['siblings'])} sibling hashes match"))
        training = Path(root) / cfg["siblings"]["training"]["path"]
        out.append(check("config.training", PASS if training.parts[-3:] == ("configs", "m2", "training.json") else FAIL,
                         f"{training.name} is the unchanged M2 recipe (25 epochs, batch 512)"))
    except Exception as error:  # noqa: BLE001
        out.append(check("config.campaign", FAIL, f"{type(error).__name__}: {error}"))
    return out


def check_fingerprints(root, campaign_path, run_dir) -> list:
    from .m2b_state import CampaignState, compute_fingerprints, require_matching
    out = []
    try:
        current = compute_fingerprints(root, campaign_path)
        out.append(check("fingerprints.compute", PASS, "components: " + ", ".join(f"{k}={v[:8]}" for k, v in current["components"].items() if isinstance(v, str))))
        state = CampaignState(Path(run_dir) / "state")
        if state.exists():
            require_matching(state.read_fingerprints(), current, None)
            out.append(check("fingerprints.state", PASS, "existing campaign state is compatible with the current fingerprints"))
        else:
            out.append(check("fingerprints.state", PASS, "no campaign state yet (fresh run)"))
    except Exception as error:  # noqa: BLE001
        out.append(check("fingerprints.state", FAIL, f"{type(error).__name__}: {error}"))
    return out


def collect_checks(args, root) -> tuple:
    """All preflight checks and the facts that drive recommended_workers. Pending VM verification."""
    from .m2b_campaign import resolve_paths
    run_dir, campaign_path = resolve_paths(args, root)
    root = Path(root)
    cfg = json.loads(campaign_path.read_text(encoding="utf-8"))
    data_cfg_path = root / cfg["siblings"]["data_v2"]["path"]
    data_cfg = json.loads(data_cfg_path.read_text(encoding="utf-8"))
    cpuinfo, meminfo, loadavg = _read("/proc/cpuinfo"), _read("/proc/meminfo"), _read("/proc/loadavg")
    mem_kb = next((int(l.split()[1]) for l in (meminfo or "").splitlines() if l.startswith("MemTotal")), 0)
    cores = os.cpu_count() or 1
    ram_gb = mem_kb * 1024 / 1e9
    workers = max(1, getattr(args, "workers", 1))
    checks = check_platform(platform.system(), platform.machine(), platform.python_version(), cpuinfo, _glibc())
    req = root / REQUIREMENTS
    pins = parse_requirements(req.read_text(encoding="utf-8")) if req.is_file() else {}
    if not req.is_file():
        checks.append(check("import.requirements", FAIL, f"{REQUIREMENTS} missing"))
    checks += check_versions(pins, _installed_versions())
    checks += check_int8_backend()
    checks += check_dataset(root, data_cfg)
    checks += check_pave(root)
    checks += check_split(root, data_cfg_path, data_cfg)
    checks += check_configs(root, campaign_path)
    checks += check_fingerprints(root, campaign_path, run_dir)
    usage_path = run_dir if run_dir.exists() else run_dir.parent
    usage_path.mkdir(parents=True, exist_ok=True)
    checks += check_resources(shutil.disk_usage(usage_path).free, mem_kb * 1024, cores, workers, cfg["threads"]["training"],
                              (loadavg or "").split()[:3] or None)
    mpl = Path(os.environ.get("MPLCONFIGDIR") or (Path.home() / ".config" / "matplotlib"))
    checks += probe_writable(run_dir, (run_dir / "logs", mpl))
    checks.append(check_scripts(root))
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        import torch
        torch_threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch_threads = None
    checks.append(check_threads(cores, physical, torch_threads))
    facts = {"cores": cores, "ram_gb": round(ram_gb, 2), "recommended_workers": recommended_workers(cores, ram_gb)}
    return run_dir, checks, facts


def run(args, root) -> int:
    """`m2b_campaign preflight`. Exit 0 unless a check FAILs (exit 2)."""
    run_dir, checks, facts = collect_checks(args, root)
    status = summarize(checks)
    report = {"schema": 1, "created_utc": utc_now(), "status": status, "kind": "preflight", "verified_on": platform.node(),
              "note": "no model was trained and no real-data inference ran", **facts, "checks": checks}
    run_dir.mkdir(parents=True, exist_ok=True)
    durable_write_json(run_dir / "preflight.json", report)
    for c in checks:
        print(f"{c['status']:<5} {c['name']:<26} {c['detail']}")
    print(f"preflight {status}: recommended_workers = {facts['recommended_workers']} "
          f"(cores {facts['cores']}, RAM {facts['ram_gb']} GB); written {run_dir / 'preflight.json'}")
    return 2 if status == FAIL else 0


# ------------------------------------------------------------------------------------------ smoke
class DeadlineStop:
    """stop_flag that also trips when the smoke time budget is used up (so no unit is started after the deadline)."""

    def __init__(self, path, minutes):
        from .m2b_campaign import StopFlag
        self.flag, self.deadline = StopFlag(path), time.monotonic() + 60 * minutes
        self.event, self.path = self.flag.event, self.flag.path

    def expired(self) -> bool:
        return time.monotonic() > self.deadline

    def __call__(self) -> bool:
        return self.flag() or self.expired()

    def set(self) -> None:
        self.flag.set()


def smoke_config(cfg: dict) -> dict:
    """Campaign config copy for the smoke run: a short E4 grid, smoke attack rows. epsilon_grid stays; units are filtered."""
    smoke = copy.deepcopy(cfg)
    smoke["e4"]["epsilons"] = [0.05]
    smoke["attack_row_policy"] = {"validation": {"per_class": SMOKE_PER_CLASS["attack"]}, "test": "all"}
    return smoke


def smoke_attack_config(attack_cfg: dict) -> dict:
    out = copy.deepcopy(attack_cfg)
    out["e4"]["epsilons"] = [0.05]
    return out


def filter_smoke_units(units):
    """Keep FGSM at SMOKE_FGSM_EPS and PGD20 at SMOKE_PGD_EPS (all sources); everything else of the plan is kept."""
    keep = []
    for u in units:
        if u.kind == "attack":
            eps, attack = float(u.params["eps"]), u.params["attack"]
            if not ((attack == "fgsm" and eps in SMOKE_FGSM_EPS) or (attack == "pgd20" and eps in SMOKE_PGD_EPS)):
                continue
        keep.append(u)
    return keep


def _smoke_arrays(root, cfg, partitions):
    """Subsets of train and validation only (the test partition is never loaded)."""
    from . import m2_split
    from .m2_pilot import make_subset
    from .paper_described_adam import derived_seed
    data_cfg = json.loads((Path(root) / cfg["siblings"]["data_v2"]["path"]).read_text(encoding="utf-8"))
    split_dir = Path(root) / data_cfg["split"]["output_dir"]
    if "test" in partitions:
        raise m2_split.PartitionGuardError("the smoke test never reads the test partition")
    full = m2_split.load_arrays(split_dir / "split_m2.json", split_dir / "scaler_minmax.json", ("train", "validation"))
    seed = derived_seed("smoke-subset", json.loads((split_dir / "split_m2.json").read_text(encoding="utf-8"))["split_seed"])
    return {p: {k: v for k, v in make_subset(full[p], SMOKE_PER_CLASS[p], seed).items() if k != "local"}
            for p in ("train", "validation")}


def _restart_check(root, env, cfg, run_dir, straight_hash, seed) -> list:
    """V1 train interrupted after epoch 1 and resumed equals the uninterrupted smoke V1 (WARN, not FAIL, if not bit-equal)."""
    from . import m2_training as mt
    out = []
    train = env.arrays()["train"]
    folder = run_dir / "restart_check"
    ckpt = folder / "ckpt"
    kwargs = dict(adversarial=False, epochs=env.epochs, threads=cfg["threads"]["training"], checkpoint_dir=ckpt, keep_last=2)
    try:
        try:
            mt.train_unit(env.train_cfg, train["x"], train["y"], seed, folder, stop_flag=lambda: True, **kwargs)
            return [check("restart.interrupt", FAIL, "training did not stop at the epoch boundary when the stop flag was set")]
        except mt.TrainingInterrupted:
            out.append(check("restart.interrupt", PASS, "stopped after epoch 1 with an atomic checkpoint"))
        resumed = mt.train_unit(env.train_cfg, train["x"], train["y"], seed, folder, stop_flag=None, **kwargs)
        equal = resumed["state_hash"] == straight_hash
        out.append(check("restart.resume_equal", PASS if equal else WARN,
                         "resumed V1 final state hash equals the uninterrupted run" if equal
                         else "resumed state hash differs from the uninterrupted run; bit-exactness is not claimed"))
    except Exception as error:  # noqa: BLE001
        out.append(check("restart.resume", FAIL, f"{type(error).__name__}: {error}"))
    return out


def _fingerprint_refusal_check(state, current, run_dir) -> dict:
    """An altered copy of the recorded fingerprints must be refused with FingerprintMismatchError."""
    from .m2b_state import FingerprintMismatchError, require_matching
    recorded = copy.deepcopy(state.read_fingerprints())
    name = next(iter(recorded["components"]))
    recorded["components"][name] = "0" * 64
    copy_path = Path(run_dir) / "fingerprint_refusal" / "fingerprints_altered.json"
    durable_write_json(copy_path, recorded)
    try:
        require_matching(json.loads(copy_path.read_text(encoding="utf-8")), current, None)
    except FingerprintMismatchError as error:
        return check("fingerprint.refusal", PASS, f"altered component {name!r} refused: {str(error)[:120]}")
    return check("fingerprint.refusal", FAIL, "an altered fingerprint was accepted")


def run_smoke(args, root) -> int:
    """`m2b_campaign smoke`: bounded train+validation-only check of every executor. Pending VM verification."""
    from . import m2_attacks as ma
    from . import m2b_exec as ex
    from . import m2b_plan as mp
    from . import m2b_report
    from .m2b_campaign import (EXIT_CONFIG, EXIT_FAILED, EXIT_OK, Ctx, ExitError, get_executors, load_campaign, resolve_paths,
                               run_units, setup_logging)
    from .m2b_state import CampaignState, compute_fingerprints
    if not 5 <= args.max_minutes <= 240:
        raise ExitError(EXIT_CONFIG, "--max-minutes must be 5..240")
    base_dir, campaign_path = resolve_paths(args, root)
    run_dir = base_dir.with_name(base_dir.name + "-smoke")
    cfg = load_campaign(campaign_path, root)
    seed = cfg["seeds"][0]
    smoke_cfg = smoke_config(cfg)
    started = time.monotonic()
    current = compute_fingerprints(root, campaign_path)
    state = CampaignState(run_dir / "state")
    state.init(f"{args.run_name}-smoke", "smoke", [seed], opt_ins=["qat"] if args.with_qat else (), fingerprints=current)
    if state.read_state()["kind"] != "smoke":
        raise ExitError(EXIT_CONFIG, "the smoke directory holds a non-smoke state")
    setup_logging(run_dir, "smoke")
    stop = DeadlineStop(state.dir / "STOP", args.max_minutes)
    attack_cfg = smoke_attack_config(ma.load_config(Path(root) / cfg["siblings"]["attacks"]["path"]))
    cache = {}

    def loader(partitions):
        if "arrays" not in cache:
            cache["arrays"] = _smoke_arrays(root, cfg, partitions)
        return cache["arrays"]
    env = ex.Env(root, run_dir, smoke_cfg, phase="validation", state=state, stop_flag=stop, fingerprints=current, epochs=2,
                 arrays_loader=loader, attack_cfg=attack_cfg, extra={"plumbing_only": True, "qat_epochs": 1})
    stages = mp.BASE_STAGES + (("qat",) if args.with_qat else ())
    units = filter_smoke_units(mp.build_plan(smoke_cfg, [seed], "validation", stages, pgd_budget="pgd20"))
    ctx = Ctx(root, run_dir, campaign_path, smoke_cfg, state, current, "validation")
    checks = [check("smoke.scope", PASS, f"{len(units)} units; train {SMOKE_PER_CLASS['train']}/class, validation {SMOKE_PER_CLASS['validation']}/class, "
                                          f"attack rows {SMOKE_PER_CLASS['attack']}/class; test partition not loaded")]
    result = run_units(ctx, units, executors=get_executors(), env=env, stop=stop, workers=1)
    counts = result["counts"]
    bad = {k: v for k, v in counts.items() if k not in ("completed",)}
    qat_only_blocked = set(result["documented_blocked"]) and all(u.stage == "qat" or any("qat" in d for d in u.deps)
                                                                for u in units if result["status"].get(u.id) == "blocked")
    if stop.expired():
        checks.append(check("smoke.time", FAIL, f"exceeded --max-minutes {args.max_minutes}"))
    elif not bad:
        checks.append(check("smoke.units", PASS, f"all {counts.get('completed', 0)} units completed"))
    elif set(bad) == {"blocked"} and qat_only_blocked:
        checks.append(check("smoke.units", WARN, f"QAT stage blocked (documented reason in the unit records); other units completed: {counts}"))
    else:
        checks.append(check("smoke.units", FAIL, f"unit statuses {counts}; see {run_dir / 'logs'}"))
    if "arrays" in cache and "test" in cache["arrays"]:
        checks.append(check("smoke.test_partition", FAIL, "the test partition was loaded"))
    else:
        checks.append(check("smoke.test_partition", PASS, "only train and validation subsets were loaded"))
    record = state.read_unit(f"s{seed}.val.train.V1")
    if record and record["status"] == "completed" and not stop.expired():
        checks += _restart_check(root, env, smoke_cfg, run_dir, record["extra"]["state_hash"], seed)
    else:
        checks.append(check("restart.resume", FAIL, "the smoke V1 training unit did not complete; the restart check was skipped"))
    checks.append(_fingerprint_refusal_check(state, current, run_dir))
    try:
        summary = m2b_report.build_report(run_dir, "validation", include_qat=args.with_qat)
        checks.append(check("report.build", PASS, f"report generated from smoke predictions ({summary['units']} rows recomputed and cross-checked)"))
    except Exception as error:  # noqa: BLE001
        checks.append(check("report.build", FAIL, f"{type(error).__name__}: {error}"))
    elapsed = time.monotonic() - started
    status = summarize(checks)
    durable_write_json(run_dir / "smoke_report.json", {"schema": 1, "created_utc": utc_now(), "status": status, "elapsed_seconds": elapsed,
                                                       "max_minutes": args.max_minutes, "counts": counts, "checks": checks,
                                                       "note": "smoke subsets only; no result of this run is scientific evidence"})
    for c in checks:
        print(f"{c['status']:<5} {c['name']:<24} {c['detail']}")
    print(f"smoke {status} in {elapsed / 60:.1f} min; written {run_dir / 'smoke_report.json'}")
    return EXIT_FAILED if status == FAIL else EXIT_OK
