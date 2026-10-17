"""Safe legacy subprocess utility; corrected experiments use research.runner."""
from pathlib import Path
import subprocess
import sys


def tee_run(cmd, result_path, cwd):
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(f"Command: {list(map(str, cmd))!r}\nCWD: {cwd}\n")
        proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            stream.write(line)
            stream.flush()
        return proc.wait()


def resolve_eval_script(upstream_root, notes_dir=None):
    root = Path(upstream_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != "e73911bbdb241f3fb6c5de346139010c4aadefd4":
        raise RuntimeError("Unexpected upstream revision")
    script = root / "code/revision_experiments.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    return script
