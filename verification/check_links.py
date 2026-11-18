#!/usr/bin/env python3
"""POST-RUN check: relative Markdown links resolve, and REPORT.md table numbers equal verify_m2.json.

Scope: the new reader-facing Markdown (reports/milestone2/*.md (including HANDOFF.md), original/README.md, evidence/README.md, source/README.md,
post-run-adaptation/README.md, verification/README.md, verification/results/*.md). Copied VM files are not checked.
Reads files only; writes nothing. Exit code 0 = all good.
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
M2 = ROOT / "reports" / "milestone2"
FILES = [M2 / "README.md", M2 / "REPORT.md", M2 / "CHECKLIST.md", M2 / "HANDOFF.md", M2 / "original" / "README.md", M2 / "evidence" / "README.md",
         M2 / "source" / "README.md", M2 / "post-run-adaptation" / "README.md", ROOT / "verification" / "README.md"]
FILES += sorted((ROOT / "verification" / "results").glob("*.md"))
LINK = re.compile(r"\]\(([^)\s]+)\)")
problems = []


def slug(h):
    h = h.strip().lower()
    h = re.sub(r"[`*_]", "", h)
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(p):
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#{1,6}\s+(.*)", line)
        if m:
            out.add(slug(m.group(1)))
    return out


n_links = 0
for f in FILES:
    if not f.is_file():
        problems.append("missing file: %s" % f)
        continue
    in_code = False
    for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for t in LINK.findall(line):
            if re.match(r"^[a-z]+:", t):
                continue
            n_links += 1
            path, _, frag = t.partition("#")
            target = (f.parent / unquote(path)).resolve() if path else f
            if not target.exists():
                problems.append("%s:%d broken link %s" % (f.relative_to(ROOT), ln, t))
                continue
            if frag and target.suffix == ".md" and slug(frag) not in anchors(target):
                problems.append("%s:%d missing anchor %s" % (f.relative_to(ROOT), ln, t))

# numbers in REPORT tables
J = json.loads((ROOT / "verification" / "results" / "verify_m2.json").read_text(encoding="utf-8"))
rep = (M2 / "REPORT.md").read_text(encoding="utf-8")
NAMES = {"V1": "V1 FP32 ordinary", "V2": "V2 INT8 PTQ of V1", "V3": "V3 FP32 adversarially trained", "V4": "V4 INT8 PTQ of V3"}
rows = {}
for line in rep.splitlines():
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.setdefault(cells[0], []).append(cells[1:])
n_num = 0
for m, name in NAMES.items():
    c, h, e = J["c_clean"][m], J["h_efficiency"][m], J["e_robustness"][m]
    four = ["%.2f%%" % (100 * c["accuracy"]), "%.2f%%" % (100 * c["macro_f1"]), format(h["state_dict_bytes"], ","), "%.3f" % h["batch1_median_ms_median_of_3"],
            "%.3f" % h["batch1_p95_ms_median_of_3"], "%.4f" % h["batch256_per_example_ms"], "%.2f" % (h["peak_vmhwm_bytes_median"] / 1e6),
            "%.2f" % (h["peak_minus_baseline_bytes_median"] / 1e6)]
    acc = ["%.2f" % (100 * e[u]["accuracy"]) for u in ("clean_eps0.0", "fgsm_eps0.05", "fgsm_eps0.1", "pgd5_eps0.05", "pgd5_eps0.1")]
    f1 = ["%.2f" % (100 * e[u]["macro_f1"]) for u in ("clean_eps0.0", "fgsm_eps0.05", "fgsm_eps0.1", "pgd5_eps0.05", "pgd5_eps0.1")]
    got = rows.get(name, [])
    for want in (four, acc, f1):
        n_num += len(want)
        if want not in got:
            problems.append("REPORT.md table row for %s does not match verify_m2.json: expected %s" % (name, want))
print("links checked: %d; table numbers checked: %d; problems: %d" % (n_links, n_num, len(problems)))
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
