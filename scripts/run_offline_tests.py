"""Explicit synthetic test suite. Does not fit study data or call APIs."""
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis/bcr'))


def deny_network(*args, **kwargs):
    raise RuntimeError("Network prohibited in offline submission tests")


def main():
    os.chdir(ROOT)
    for key in list(os.environ):
        if any(word in key.upper() for word in ["API_KEY", "ACCESS_TOKEN", "SECRET_KEY"]):
            os.environ.pop(key)
    os.environ.pop("IMMUNOAUDIT_SUBPROCESS_PATHS", None)
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    import pytest
    suite = [
        "tests",
        "experiments/task2_cancer_context/tests",
        "analysis/cancer_context/tests",
        "analysis/grouped_tasks/tests/test_task135_core.py",
        "analysis/temporal_summary/tests/test_extension.py",
        "analysis/disagreement/tests/test_disagreement_case_analysis.py",
        "analysis/bcr/test_repair.py",
        "code_strategy3/tests/test_profiles_v1.py",
    ]
    # This one integration assertion needs the withheld canonical patient table.
    excluded = "experiments/task2_cancer_context/tests/test_run.py::FoldAssignmentTests::test_exact_legacy_fold_membership_and_events"
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--deselect", excluded, *suite]))


if __name__ == "__main__":
    main()
