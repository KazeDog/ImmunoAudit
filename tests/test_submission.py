import importlib
import json
from pathlib import Path
import pytest
import submission_paths


def test_path_parameters_and_subprocess_bridge(tmp_path, monkeypatch):
    monkeypatch.delenv("IMMUNOAUDIT_SUBPROCESS_PATHS", raising=False)
    submission_paths.configure(project=tmp_path / "work", data=tmp_path / "raw", models=tmp_path / "weights")
    assert submission_paths.path("data", "study/a.csv") == tmp_path / "raw/study/a.csv"
    assert submission_paths.path("models", "checkpoint.pt") == tmp_path / "weights/checkpoint.pt"
    importlib.reload(submission_paths)
    assert submission_paths.path("project") == tmp_path / "work"
    with pytest.raises(ValueError):
        submission_paths.path("project", "../escape")


def test_credentials_are_empty_without_environment(monkeypatch):
    for name in ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    from code_strategy3.python_settings import get_strategy3_profile
    assert get_strategy3_profile("qwen")["api_key"] is None
    assert get_strategy3_profile("gemini")["api_key"] is None


def test_archive_policy():
    root = submission_paths.PACKAGE_ROOT
    assert not (root / "code_strategy3/python_settings_local.py").exists()
    assert not list(root.rglob("*.csv"))
    assert not list(root.rglob("*.jsonl"))
