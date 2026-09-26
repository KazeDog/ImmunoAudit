"""Submission configuration: environment-only credentials, parameterized paths.

Private local settings are deliberately never imported by this distribution.
"""
import os
from submission_paths import path

WORKSPACE_DIR = path("project")
STRATEGY3_RESULTS_ROOT = WORKSPACE_DIR / "results" / "strategy3"
DEFAULT_STRATEGY3_PROFILE = "qwen"


def build_profile(provider, *, model, api_base=None, timeout_seconds=120, max_retries=3, temperature=0):
    return dict(provider=provider, model=model, api_key=None, api_base=api_base,
                timeout_seconds=timeout_seconds, max_retries=max_retries, temperature=temperature,
                cache_dir=STRATEGY3_RESULTS_ROOT / provider / "cache",
                results_dir=STRATEGY3_RESULTS_ROOT / provider)


STRATEGY3_PROFILES = {
    "qwen": build_profile("qwen", model="qwen3.5-plus", api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "gemini": build_profile("gemini", model="gemini-3.1-flash-preview"),
}


def get_strategy3_profile(profile_name=None):
    name = str(profile_name or DEFAULT_STRATEGY3_PROFILE).strip().lower()
    if name not in STRATEGY3_PROFILES:
        raise KeyError(f"Unknown Strategy 3 profile: {name}")
    profile = dict(STRATEGY3_PROFILES[name])
    if name == "qwen":
        profile["api_key"] = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    else:
        profile["api_key"] = os.environ.get("GEMINI_API_KEY")
    return profile


DEFAULT_STRATEGY3_CONFIG = get_strategy3_profile()
TASK3_QWEN_CONFIG = get_strategy3_profile("qwen")
