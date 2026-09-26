"""Parameter-driven path configuration for the portable code distribution."""
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
_STATE = "IMMUNOAUDIT_SUBPROCESS_PATHS"


def configure(**paths):
    """Called by run.py; the private environment bridge also reaches subprocesses."""
    resolved = {k: str(Path(v).expanduser().resolve()) for k, v in paths.items() if v is not None}
    os.environ[_STATE] = json.dumps(resolved)


def path(kind, suffix=""):
    settings = json.loads(os.environ.get(_STATE, "{}"))
    project = Path(settings.get("project", PACKAGE_ROOT))
    defaults = {"project": project, "data": project / "data",
                "models": project / "external_models", "fonts": project / "fonts",
                "figure_qa": project / "external_tools" / "nature-figure"}
    if kind not in defaults:
        raise ValueError(f"Unknown path parameter: {kind}")
    relative = Path(suffix)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path suffix must remain relative to its configured root")
    return Path(settings.get(kind, defaults[kind])) / relative
