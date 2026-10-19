"""Local sequential baseline completion and review export, with resumable epochs.

--wait-for-first waits for the separately launched first seed to validate and
release its run lock. It does not retry a failed first seed or start concurrent
training. Without that flag, this is the exact normal continuation command.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-first", action="store_true")
    args = parser.parse_args()
    run = ROOT / "runs/milestone1-artifact-full"
    log = run / "controller.log"
    status = run / "controller.json"
    def write(state, **extra):
        value = {"state": state, "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temp.replace(status)
    if args.wait_for_first:
        write("waiting_for_first_seed_validation")
        while (run / "run.lock").exists():
            time.sleep(10)
        if not (run / "seed-1/manifest.json").is_file():
            write("stopped_first_seed_not_completed", reason="Inspect the first seed log/checkpoints before continuing; no further training launched")
            raise SystemExit(1)
    commands = [
        [sys.executable, "-B", "-m", "research.artifact_baseline", "--run-dir", str(run), "--seeds", "1", "--verify-only"],
        [sys.executable, "-B", "-m", "research.artifact_baseline", "--run-dir", str(run), "--resume"],
        [sys.executable, "-B", "-m", "research.artifact_baseline", "--run-dir", str(run), "--verify-only"],
        [sys.executable, "-B", "-m", "research.milestone1_report", "--run-dir", str(run), "--output", str(ROOT / "reports/milestone1-artifact")],
    ]
    # A normal continuation can resume an interrupted first seed before verifying it.
    if not args.wait_for_first:
        commands = commands[1:]
    with log.open("a", encoding="utf-8") as stream:
        for command in commands:
            write("executing", command=command)
            stream.write("\nCOMMAND " + json.dumps(command) + "\n")
            stream.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                write("failed", command=command, exit_code=result.returncode, log=str(log))
                raise SystemExit(result.returncode)
        # Preserve any earlier ZIP; generate a fresh timestamped package if needed.
        outputs = ROOT.parent / "outputs"
        archive = outputs / "milestone1-practical-review.zip"
        if archive.exists():
            archive = outputs / ("milestone1-practical-review-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".zip")
        command = [sys.executable, "-B", "-m", "research.milestone1_report", "--run-dir", str(run), "--output", str(ROOT / "reports/milestone1-artifact"), "--export", str(archive)]
        write("exporting", command=command)
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write("failed_export", exit_code=result.returncode, log=str(log))
            raise SystemExit(result.returncode)
        deliverable = outputs / "milestone1-practical"
        deliverable.mkdir(exist_ok=True)
        import shutil
        for path in (ROOT / "reports/milestone1-artifact").iterdir():
            if path.is_file():
                shutil.copy2(path, deliverable / path.name)
        write("complete", verified_seeds=[1, 7, 13, 29], review_zip=str(archive), report=str(deliverable / "REPORT.md"), claim="Released-artifact multiclass baseline and documented paper-reproduction attempt.")


if __name__ == "__main__":
    main()
