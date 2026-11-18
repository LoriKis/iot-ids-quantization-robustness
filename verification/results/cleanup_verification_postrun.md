# Pre-commit cleanup verification (POST-RUN)

Nothing here was part of the VM campaign. No model was loaded; nothing was trained, run, attacked or benchmarked. The VM folders were only read and hashed. `$c` is the clone, `$vm4` and `$vmd` are the two VM folders, `$tmp` is the temporary folder. A temporary index and two temporary checkouts outside the repository were used and deleted afterwards.

## Commands run (in this order)

```powershell
# Task 1: README bytes (raw bytes, no text pipeline)
git -C $c show 466faf724aaab5507d5cf12f74281c5f2a62ae28:README.md   # redirected to a temp file with Start-Process -RedirectStandardOutput
# byte edit with [IO.File]::ReadAllBytes / WriteAllBytes (removed one LF after the end marker), then a byte-range removal test and SHA-256 compare
git -C $c diff --name-status HEAD ; git -C $c diff --stat HEAD

# Task 4: read-only reads of the campaign logs and stage markers in $vmd\artifacts\m2-deadline-v1, then
Get-FileHash -Algorithm SHA256 <each file recorded in stage_markers/*.ok and ckpt/latest.json>   # result: verification/results/vm_hash_check_postrun.tsv

# Task 5: temporary index and checkouts
$env:GIT_INDEX_FILE = "$tmp\m2-tmp-2.index"
git -C $c read-tree HEAD
git -C $c add -- README.md reports/milestone2 verification
git -C $c diff --cached --name-status HEAD
git -C $c checkout-index -a --prefix="$tmp/m2-checkout-2/"
git -C $c -c core.autocrlf=true checkout-index -a --prefix="$tmp/m2-checkout-2-autocrlf/"   # config not saved
Remove-Item Env:GIT_INDEX_FILE ; Remove-Item "$tmp\m2-tmp-2.index"
$env:PYTHONDONTWRITEBYTECODE = '1'
python "$tmp\m2-checkout-2\verification\verify_m2.py" --evidence-root "$tmp\m2-checkout-2\reports\milestone2\evidence" --out "$tmp\m2-verify-out-2" --figures "$tmp\m2-verify-out-2\figures"
python "$tmp\m2-checkout-2\verification\check_links.py"
# SHA-256 comparisons: working copy vs checkout; VM originals vs checkout (paths from verification/results/copy_map.tsv); real index and HEAD checks
git -C $c diff --cached ; git -C $c status --short ; git -C $c rev-parse HEAD
```

## Output of the final run

```
add (no -f) output: []
temp-index name-status counts: M=1 A=337
temp-index non-A entries: M	README.md
entries outside publication paths: 0
temp-index entries matching forbidden patterns (.agents, .pt/.pth/.ckpt, __pycache__, .venv, .mplconfig, zip, dataset): 0
ignored files among publication paths (git ls-files -o -i --exclude-standard): 0
checkout-index (default config) exit=0
checkout-index (-c core.autocrlf=true, not saved) exit=0
checkout files: 564
absent check: .agents dirs=0 .venv dirs=0 __pycache__ dirs=0 .mplconfig dirs=0; .pt/.pth/.ckpt files=0; .pyc=0; balanced_CICIoT*=0
any path containing '.agents': 0
required paths missing in checkout: 0
source files present in checkout: 123; evidence files: 188
verify_m2.py (from checkout) exit=0
verify_m2 counts (summary line of verify_m2.txt): PASS=174 FAIL=0 NOTE=25
saved verify_m2.json equals regenerated-from-checkout output: True
saved verify_m2.txt equals regenerated-from-checkout output: True
check_links.py (from checkout) exit=0 : links checked: 203; table numbers checked: 72; problems: 0
publication files compared (working copy vs default checkout): 338; mismatches: 0
same comparison with -c core.autocrlf=true checkout: mismatches: 0
VM originals vs default checkout (copy_map.tsv): 312 files, mismatches: 0
VM originals vs autocrlf=true checkout: mismatches: 0
README in checkout: block-removed bytes equal original: True
checkout files under reports/milestone2 (text-like) that contain CR bytes (CRLF preserved): 17
git check-attr text for a copied CRLF file: reports/milestone2/source/v4-base/pytest.ini: text: unset
git check-attr text for README.md: README.md: text: auto
real index: diff --cached lines = 0
git status --short:  M README.md | ?? .agents/ | ?? reports/milestone2/ | ?? verification/
HEAD = 466faf724aaab5507d5cf12f74281c5f2a62ae28
git diff --name-status HEAD: M	README.md
```

Notes:
- The first run of the same steps found 5 mismatches between checkout and working copy, all in `verification/results/` (`check_links_output.txt`, `console_output.txt`, `copy_check.txt`, `copy_map.tsv`, `vm_hash_check_postrun.tsv`). Cause: those files had CRLF line endings and the root `.gitattributes` rule `* text=auto eol=lf` converts them to LF. Fix: new `verification/.gitattributes` (`* -text`). After the fix the final run shows 0 mismatches, also with `core.autocrlf=true`. No historical hash manifest was changed.
- `verify_m2.py` summary line: PASS=174 FAIL=0 NOTE=25.
- `git add` needed no `-f`; no ignored file is in the publication paths.