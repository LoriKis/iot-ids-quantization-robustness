"""
discover_artifacts.py — Automated discovery of PAVE artifact components.

Walks upstream/PAVE_artifact/ recursively (read-only) and writes a structured
report to notes/pave_reproduction_notes.md.

Discovers:
  1. Processed CICIoT2023 data splits (.csv, .npy, .pt, .parquet)
  2. Standard Transformer checkpoints (*.pt / *.pth with keyword matches)
  3. Primary evaluation script (parsed from upstream README.md)
  4. FGSM attack implementation (source files containing FGSM logic)
  5. PGD attack implementation (source files containing PGD logic)
  6. Default attack parameters (extracted from source)

Usage:
    python experiments/milestone1/discover_artifacts.py
    python experiments/milestone1/discover_artifacts.py --upstream path/to/PAVE_artifact
"""

import argparse
import os
import pathlib
import re
import sys
import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_EXTENSIONS = {".csv", ".npy", ".pt", ".parquet"}

CHECKPOINT_EXTENSIONS = {".pt", ".pth"}
CHECKPOINT_KEYWORDS = ["transformer", "standard", "ciciot"]

ATTACK_FGSM_KEYWORDS = ["FGSM", "fgsm", "fast_gradient", "FastGradient"]
ATTACK_PGD_KEYWORDS  = ["PGD", "pgd", "projected_gradient", "ProjectedGradient"]

# Patterns for extracting default parameter values from source
PARAM_PATTERNS = {
    "fgsm_epsilon": re.compile(r"epsilon\s*[:=]\s*([0-9.]+)"),
    "pgd_epsilon":  re.compile(r"epsilon\s*[:=]\s*([0-9.]+)"),
    "pgd_alpha":    re.compile(r"alpha\s*[:=]\s*([0-9.]+)"),
    "pgd_steps":    re.compile(r"(?:n_iter|iters|steps|iterations)\s*[:=]\s*(\d+)"),
    "eps_const":    re.compile(r"^EPS\s*=\s*([0-9.]+)", re.MULTILINE),
    "pgd_const":    re.compile(r"^PGD\s*=\s*dict\(([^)]+)\)", re.MULTILINE),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_match(path: pathlib.Path, keywords: list) -> bool:
    """Return True if any keyword appears in the file's name (case-insensitive)."""
    name = path.name.lower()
    return any(kw.lower() in name for kw in keywords)


def _content_match(path: pathlib.Path, keywords: list) -> list:
    """
    Return list of (line_number, line_text) tuples where any keyword appears.
    Returns [] on read errors (binary files, permission issues).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(kw in line for kw in keywords):
                hits.append((lineno, line.strip()))
        return hits
    except Exception:
        return []


def _extract_params_from_file(path: pathlib.Path) -> dict:
    """Extract default attack param values by scanning source text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    params = {}
    for name, pat in PARAM_PATTERNS.items():
        m = pat.search(text)
        if m:
            params[name] = m.group(1)
    return params


def _parse_readme_eval_script(readme: pathlib.Path) -> list:
    """
    Parse the upstream README.md and return a list of python invocation lines
    found in fenced code blocks.
    """
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8", errors="replace")
    # Extract all lines inside ``` fences that reference python
    in_block = False
    python_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_block = not in_block
            continue
        if in_block and ("python" in stripped or stripped.startswith("py ")):
            python_lines.append(stripped)
    return python_lines


# ---------------------------------------------------------------------------
# Discovery routines
# ---------------------------------------------------------------------------

def discover_data_splits(upstream: pathlib.Path) -> list:
    """Find processed data split files by extension."""
    found = []
    for ext in DATA_EXTENSIONS:
        for p in upstream.rglob(f"*{ext}"):
            size_mb = p.stat().st_size / (1024 * 1024)
            found.append((str(p.relative_to(upstream.parent)), f"{size_mb:.2f} MB"))
    return found


def discover_checkpoints(upstream: pathlib.Path) -> list:
    """Find transformer checkpoint files by extension + filename keywords."""
    found = []
    for ext in CHECKPOINT_EXTENSIONS:
        for p in upstream.rglob(f"*{ext}"):
            if _keyword_match(p, CHECKPOINT_KEYWORDS):
                size_mb = p.stat().st_size / (1024 * 1024)
                found.append((str(p.relative_to(upstream.parent)), f"{size_mb:.2f} MB"))
    return found


def discover_attack_impl(upstream: pathlib.Path, keywords: list, attack_name: str) -> list:
    """Find Python source files containing attack keywords."""
    found = []
    for p in upstream.rglob("*.py"):
        hits = _content_match(p, keywords)
        if hits:
            # Report first two matching lines only to keep report readable
            excerpt = hits[:2]
            found.append((str(p.relative_to(upstream.parent)), excerpt))
    return found


def discover_eval_script(upstream: pathlib.Path) -> tuple:
    """Return (readme_path, python_invocation_lines) from upstream README."""
    readme = upstream / "README.md"
    lines = _parse_readme_eval_script(readme)
    return str(readme.relative_to(upstream.parent)) if readme.exists() else "NOT FOUND", lines


def extract_default_params(upstream: pathlib.Path) -> dict:
    """
    Scan the PAVE source tree for canonical default attack parameters.
    Priority: module-level constants (EPS=, PGD=dict(...)) in sota_compare.py.
    Falls back to function-default args in adversarial.py.
    """
    params = {}

    # sota_compare.py has module-level EPS and PGD constants
    sota = upstream / "code" / "sota_compare.py"
    if sota.exists():
        text = sota.read_text(encoding="utf-8", errors="replace")
        m_eps = PARAM_PATTERNS["eps_const"].search(text)
        if m_eps:
            params["FGSM_epsilon"] = m_eps.group(1)
        m_pgd = PARAM_PATTERNS["pgd_const"].search(text)
        if m_pgd:
            raw = m_pgd.group(1)  # e.g. "eps=0.1, alpha=0.01, iters=10"
            for kv in raw.split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[f"PGD_{k.strip()}"] = v.strip()

    # Fall back to adversarial.py function signatures
    adv = upstream / "code" / "attacks" / "adversarial.py"
    if adv.exists() and not params:
        params.update(_extract_params_from_file(adv))

    return params


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    output_path: pathlib.Path,
    data_splits: list,
    checkpoints: list,
    fgsm_impl: list,
    pgd_impl: list,
    eval_readme: str,
    eval_lines: list,
    params: dict,
    upstream_path: pathlib.Path,
) -> None:
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    lines = [
        "# PAVE Artifact — Discovery Report",
        f"## Generated: {timestamp}",
        f"## Upstream: {upstream_path}",
        "",
    ]

    # ── Data Splits ──────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 1. Processed Data Splits")
    lines.append("")
    if data_splits:
        for path, size in data_splits:
            lines.append(f"- `{path}` ({size})")
    else:
        lines.append("- **NONE FOUND** — search pattern: .csv, .npy, .pt, .parquet")
    lines.append("")

    # ── Checkpoints ──────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 2. Standard Transformer Checkpoints")
    lines.append("")
    if checkpoints:
        for path, size in checkpoints:
            lines.append(f"- `{path}` ({size})")
    else:
        lines.append(
            "- **NONE FOUND** — searched *.pt / *.pth matching keywords: "
            "transformer, standard, ciciot"
        )
        lines.append(
            "  > NOTE: PAVE trains models from scratch at evaluation time. "
            "No pre-trained checkpoint files are shipped in the repository."
        )
    lines.append("")

    # ── Evaluation Script ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 3. Evaluation Script (from README)")
    lines.append("")
    lines.append(f"- README parsed: `{eval_readme}`")
    if eval_lines:
        lines.append("- Python invocations found in README code blocks:")
        for ln in eval_lines:
            lines.append(f"  ```")
            lines.append(f"  {ln}")
            lines.append(f"  ```")
    else:
        lines.append("- **NONE FOUND** — no python invocations in README fenced blocks.")
    lines.append("")

    # ── FGSM Implementation ──────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 4. FGSM Attack Implementation")
    lines.append("")
    if fgsm_impl:
        for path, hits in fgsm_impl:
            lines.append(f"- `{path}`")
            for lineno, text in hits:
                lines.append(f"  - line {lineno}: `{text[:120]}`")
    else:
        lines.append("- **NONE FOUND** — searched for: FGSM, fgsm, fast_gradient")
    lines.append("")

    # ── PGD Implementation ───────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 5. PGD Attack Implementation")
    lines.append("")
    if pgd_impl:
        for path, hits in pgd_impl:
            lines.append(f"- `{path}`")
            for lineno, text in hits:
                lines.append(f"  - line {lineno}: `{text[:120]}`")
    else:
        lines.append("- **NONE FOUND** — searched for: PGD, pgd, projected_gradient")
    lines.append("")

    # ── Default Parameters ───────────────────────────────────────────────────
    lines.append("---")
    lines.append("## 6. Default Attack Parameters (Extracted from Source)")
    lines.append("")
    if params:
        for k, v in sorted(params.items()):
            lines.append(f"- **{k}** = `{v}`")
    else:
        lines.append("- **NONE EXTRACTED** — manual inspection required.")
    lines.append("")

    # ── Protocol Validity Caveat ──────────────────────────────────────────────
    lines.append("---")
    lines.append("## 7. Protocol Validity Caveat")
    lines.append("")
    lines.append(
        "Unconstrained feature-space adversarial perturbations (FGSM, PGD) applied to "
        "network traffic features may produce feature vectors that do not correspond to "
        "any physically realizable network packet sequence. Results should be interpreted "
        "in the feature-space threat model only."
    )
    lines.append("")
    lines.append(
        "The PAVE framework explicitly evaluates both **unconstrained** and "
        "**constraint-respecting (projected)** variants of PGD to quantify how much "
        "robustness is overstated when protocol validity is ignored."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--upstream",
        default=None,
        help="Path to upstream/PAVE_artifact. Defaults to ../../upstream/PAVE_artifact "
             "relative to this script.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to write the discovery report. Defaults to ../../notes/pave_reproduction_notes.md.",
    )
    args = ap.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    upstream = pathlib.Path(args.upstream).resolve() if args.upstream else \
               project_root / "upstream" / "PAVE_artifact"
    output = pathlib.Path(args.output).resolve() if args.output else \
             project_root / "notes" / "pave_reproduction_notes.md"

    if not upstream.exists():
        print(f"ERROR: upstream directory not found: {upstream}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {upstream}")
    print(f"Output  : {output}")
    print()

    print("1/6  Discovering data splits ...")
    data_splits = discover_data_splits(upstream)
    print(f"     Found {len(data_splits)} file(s).")

    print("2/6  Discovering transformer checkpoints ...")
    checkpoints = discover_checkpoints(upstream)
    print(f"     Found {len(checkpoints)} file(s).")

    print("3/6  Discovering evaluation script from README ...")
    eval_readme, eval_lines = discover_eval_script(upstream)
    print(f"     Found {len(eval_lines)} invocation line(s).")

    print("4/6  Discovering FGSM implementation ...")
    fgsm_impl = discover_attack_impl(upstream, ATTACK_FGSM_KEYWORDS, "FGSM")
    print(f"     Found {len(fgsm_impl)} file(s).")

    print("5/6  Discovering PGD implementation ...")
    pgd_impl = discover_attack_impl(upstream, ATTACK_PGD_KEYWORDS, "PGD")
    print(f"     Found {len(pgd_impl)} file(s).")

    print("6/6  Extracting default attack parameters ...")
    params = extract_default_params(upstream)
    print(f"     Extracted {len(params)} parameter(s).")

    print()
    write_report(
        output_path=output,
        data_splits=data_splits,
        checkpoints=checkpoints,
        fgsm_impl=fgsm_impl,
        pgd_impl=pgd_impl,
        eval_readme=eval_readme,
        eval_lines=eval_lines,
        params=params,
        upstream_path=upstream,
    )

    # Print summary table
    print()
    print("=" * 60)
    print(" Discovery Summary")
    print("=" * 60)
    categories = [
        ("Data splits",           data_splits),
        ("Checkpoints",           checkpoints),
        ("Eval script lines",     [(l, "") for l in eval_lines]),
        ("FGSM impl files",       fgsm_impl),
        ("PGD impl files",        pgd_impl),
    ]
    for label, items in categories:
        status = f"{len(items)} found" if items else "NONE FOUND"
        print(f"  {label:<30} {status}")
    if params:
        print()
        print("  Default parameters:")
        for k, v in sorted(params.items()):
            print(f"    {k:<25} = {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
