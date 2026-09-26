"""Versioned, non-secret Strategy 3 model profiles.

These profiles describe models that use the existing DashScope
OpenAI-compatible provider implementation. API keys intentionally do not
belong in this module; they continue to come from the existing local settings
or environment variables.
"""
from submission_paths import path as _submission_path
from pathlib import Path
PROFILE_SCHEMA_VERSION = 1
_WORKSPACE_DIR = _submission_path('project', '')
_RESULTS_ROOT = _WORKSPACE_DIR / 'results' / 'strategy3'
_DASHSCOPE_API_BASE = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

def _dashscope_profile(*, model: str, output_subdir: str) -> dict:
    results_dir = _RESULTS_ROOT / output_subdir
    return {'provider': 'qwen', 'model': model, 'api_base': _DASHSCOPE_API_BASE, 'cache_dir': results_dir / 'cache', 'results_dir': results_dir}
STRATEGY3_PROFILES_V1 = {'kimi-k2.6': _dashscope_profile(model='kimi/kimi-k2.6', output_subdir='kimi_k26'), 'minimax-m2.7': _dashscope_profile(model='MiniMax/MiniMax-M2.7', output_subdir='minimax_m27')}
STRATEGY3_PROFILE_ALIASES_V1 = {'kimi': 'kimi-k2.6', 'kimi-2.6': 'kimi-k2.6', 'kimi-k26': 'kimi-k2.6', 'minimax': 'minimax-m2.7', 'minimax-2.7': 'minimax-m2.7', 'minimax-m27': 'minimax-m2.7'}

def get_versioned_strategy3_profile(profile_name: str) -> dict | None:
    """Return a copy of a v1 profile, resolving supported aliases."""
    normalized_name = str(profile_name).strip().lower()
    canonical_name = STRATEGY3_PROFILE_ALIASES_V1.get(normalized_name, normalized_name)
    profile = STRATEGY3_PROFILES_V1.get(canonical_name)
    return None if profile is None else dict(profile)
