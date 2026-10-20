"""Queue one unchanged-manifest recovery pass after the active supervisor exits."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    queue_lock = (directory / "recovery.lock").open("a")
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(state, **extra):
        record = dict(status=state, time=datetime.now(timezone.utc).isoformat(),
                      manifest_id=manifest["manifest_id"], **extra)
        temporary = directory / "recovery-status.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(directory / "recovery-status.json")
        print(json.dumps(record), flush=True)

    command = [sys.executable, "-u", "fresh_trajectories.py"]
    for key, value in manifest["config"].items():
        command.append("--" + key.replace("_", "-"))
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(str(value))
    command.extend(["--run-dir", str(directory), "--workers", "3", "--resume"])
    status("waiting_for_active_supervisor", command=command,
           reason="Retry incomplete runs with original code and strict initialization guard; completed runs are skipped.")
    with (directory / "process.lock").open("a") as active_lock:
        fcntl.flock(active_lock, fcntl.LOCK_EX)
        previous = json.loads((directory / "status.json").read_text())
        if previous.get("status") == "complete":
            status("already_complete")
            return
    # The original runner owns its own exclusive lock and revalidates all hashes.
    status("resuming", command=command)
    with (directory / "recovery.log").open("a") as log:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent,
                                stdout=log, stderr=subprocess.STDOUT)
    status("complete" if result.returncode == 0 else "failed", returncode=result.returncode)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
