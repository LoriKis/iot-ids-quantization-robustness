"""Clone only when absent; verify pin and clean state without resetting work."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import UPSTREAM, PIN, verify_upstream

if not UPSTREAM.exists():
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/robotumbel/PAVE_artifact.git", str(UPSTREAM)], check=True)
    subprocess.run(["git", "-C", str(UPSTREAM), "checkout", "--detach", PIN], check=True)
print("Verified pinned, clean model source:", verify_upstream())
