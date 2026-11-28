"""Stop an owned Team 21 notebook fit if an eligible ZIP reaches its verified CPU target."""
import os
from pathlib import Path
import signal
import time

from team21_reached_target import main as reached_target


def training_processes():
    root = Path.cwd()
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            if directory.stat().st_uid != os.getuid() or (directory / "cwd").resolve(strict=True) != root:
                continue
            argv = (directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            if len(argv) < 4 or Path(argv[1]).name != "run_notebook_pilot.py":
                if len(argv) < 5 or argv[1] != "-u" or Path(argv[2]).name != "run_notebook_pilot.py":
                    continue
            if "--run" not in argv or "--notebook" not in argv or argv[argv.index("--notebook")+1] != "train_team21.ipynb":
                continue
            run = Path(argv[argv.index("--run")+1]).resolve()
            if run.parent != root / "overnight" or not run.name.startswith("team21-") or (run / "resources.json").is_file():
                continue
            # A forked data loader inherits the runner's argv. Signal only the trainer.
            if (directory / "comm").read_text().strip().startswith("pt_data_worker"):
                continue
            yield int(directory.name), run.name
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


if __name__ == "__main__":
    while True:
        status = reached_target()
        if status == 0:
            for pid, run in training_processes():
                try:
                    os.kill(pid, signal.SIGINT)
                    print("Requested stop", run, pid, flush=True)
                except ProcessLookupError:
                    pass
            break
        if status != 3:
            raise RuntimeError(f"Target evidence check returned {status}")
        time.sleep(10)
