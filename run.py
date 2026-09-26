#!/usr/bin/env python3
"""Run a packaged research script with explicit, portable path parameters."""
import argparse
import os
from pathlib import Path
import runpy
import socket
import sys

from submission_paths import PACKAGE_ROOT, configure


def block_network(*args, **kwargs):
    raise RuntimeError("Network access disabled. Live inference requires --allow-network.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PACKAGE_ROOT,
                        help="Workspace with the documented code/data relative layout (default: package).")
    parser.add_argument("--data-root", type=Path, help="Raw-data root; default: PROJECT/data.")
    parser.add_argument("--models-root", type=Path, help="External model/cache root; default: PROJECT/external_models.")
    parser.add_argument("--allow-network", action="store_true", help="Explicit opt-in for API calls/downloads.")
    parser.add_argument("--script", required=True, help="Relative path to a Python script within this code package.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments after -- go to the selected script.")
    args = parser.parse_args()
    script = (PACKAGE_ROOT / args.script).resolve()
    if not script.is_relative_to(PACKAGE_ROOT) or script.suffix != ".py" or not script.is_file():
        parser.error("--script must identify an existing .py file inside the package")
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        parser.error("--project-root must exist; use a new workspace copy for experiments")
    configure(project=project, data=args.data_root, models=args.models_root)
    # Keep imports valid for subprocesses, including fresh BCR output directories.
    os.environ["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key, value in {"OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OMP_NUM_THREADS": "8"}.items():
        os.environ.setdefault(key, value)
    if not args.allow_network:
        socket.socket.connect = block_network
        socket.socket.connect_ex = block_network
        socket.create_connection = block_network
    os.chdir(project)
    sys.path.insert(0, str(script.parent))
    extra = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    sys.argv = [str(script), *extra]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
