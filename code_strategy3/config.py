import os
from dataclasses import dataclass, field
from pathlib import Path
from .python_settings import DEFAULT_STRATEGY3_PROFILE
from .python_settings import WORKSPACE_DIR
from .python_settings import get_strategy3_profile
from .profiles_v1 import get_versioned_strategy3_profile
STRATEGY3_RESULTS_ROOT = WORKSPACE_DIR / 'results' / 'strategy3'

def default_results_dir(provider: str) -> Path:
    normalized_provider = str(provider or 'qwen').strip().lower() or 'qwen'
    return STRATEGY3_RESULTS_ROOT / normalized_provider

def default_cache_dir(provider: str) -> Path:
    return default_results_dir(provider) / 'cache'

def resolve_strategy3_profile(profile_name: str | None=None) -> dict:
    """Resolve server profiles plus non-secret, versioned model profiles.

    Versioned profiles inherit provider-level defaults and any locally supplied
    API key, while their model, endpoint, and output isolation remain fixed.
    """
    resolved_name = str(profile_name or DEFAULT_STRATEGY3_PROFILE).strip().lower()
    versioned_profile = get_versioned_strategy3_profile(resolved_name)
    if versioned_profile is None:
        return get_strategy3_profile(resolved_name)
    provider = str(versioned_profile['provider']).strip().lower()
    try:
        merged = get_strategy3_profile(provider)
    except KeyError:
        merged = {'provider': provider}
    merged.update(versioned_profile)
    return merged

@dataclass(frozen=True)
class Strategy3Config:
    provider: str
    model: str
    api_key: str | None = field(repr=False)
    api_base: str | None
    timeout_seconds: int
    max_retries: int
    temperature: float
    cache_dir: Path
    results_dir: Path

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> 'Strategy3Config':
        data = dict(mapping or {})
        provider = str(data.get('provider', DEFAULT_STRATEGY3_PROFILE)).strip().lower()
        try:
            merged = get_strategy3_profile(provider)
        except KeyError:
            merged = {'provider': provider}
        merged.update(data)
        model = str(merged.get('model', 'qwen-plus-latest'))
        api_key = data.get('api_key')
        if api_key is None:
            api_key = merged.get('api_key')
        api_base = merged.get('api_base')
        timeout_seconds = int(merged.get('timeout_seconds', 120))
        max_retries = int(merged.get('max_retries', 3))
        temperature = float(merged.get('temperature', 0))
        cache_dir = Path(merged.get('cache_dir', default_cache_dir(provider)))
        results_dir = Path(merged.get('results_dir', default_results_dir(provider)))
        return cls(provider=provider, model=model, api_key=api_key, api_base=api_base, timeout_seconds=timeout_seconds, max_retries=max_retries, temperature=temperature, cache_dir=cache_dir, results_dir=results_dir)

    @classmethod
    def from_env(cls, profile_name: str | None=None) -> 'Strategy3Config':
        selected_profile = str(profile_name or os.environ.get('STRATEGY3_PROFILE') or os.environ.get('STRATEGY3_PROVIDER') or DEFAULT_STRATEGY3_PROFILE).strip().lower()
        try:
            base = resolve_strategy3_profile(selected_profile)
        except KeyError:
            base = {'provider': selected_profile}
        provider = str(base.get('provider', selected_profile)).strip().lower()
        model = os.environ.get('STRATEGY3_MODEL', str(base.get('model', 'qwen-plus-latest')))
        api_key = cls.resolve_api_key(provider) or base.get('api_key')
        api_base = cls.resolve_api_base(provider)
        if api_base is None:
            api_base = base.get('api_base')
        timeout_seconds = int(os.environ.get('STRATEGY3_TIMEOUT_SECONDS', str(base.get('timeout_seconds', 120))))
        max_retries = int(os.environ.get('STRATEGY3_MAX_RETRIES', str(base.get('max_retries', 3))))
        temperature = float(os.environ.get('STRATEGY3_TEMPERATURE', str(base.get('temperature', 0))))
        cache_dir = Path(os.environ.get('STRATEGY3_CACHE_DIR', str(base.get('cache_dir', default_cache_dir(provider)))))
        results_dir = Path(os.environ.get('STRATEGY3_RESULTS_DIR', str(base.get('results_dir', default_results_dir(provider)))))
        return cls(provider=provider, model=model, api_key=api_key, api_base=api_base, timeout_seconds=timeout_seconds, max_retries=max_retries, temperature=temperature, cache_dir=cache_dir, results_dir=results_dir)

    @classmethod
    def from_profile(cls, profile_name: str | None=None) -> 'Strategy3Config':
        return cls.from_mapping(resolve_strategy3_profile(profile_name))

    def require_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        raise ValueError(f'Missing API key for provider: {self.provider}')

    def require_api_base(self) -> str:
        if self.api_base:
            return self.api_base
        raise ValueError(f'Missing API base for provider: {self.provider}')

    @staticmethod
    def resolve_api_key(provider: str) -> str | None:
        if provider == 'gemini':
            return os.environ.get('GEMINI_API_KEY')
        if provider == 'qwen':
            return os.environ.get('QWEN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
        return None

    @staticmethod
    def resolve_api_base(provider: str) -> str | None:
        if provider == 'gemini':
            return os.environ.get('STRATEGY3_API_BASE')
        if provider == 'qwen':
            return os.environ.get('STRATEGY3_API_BASE')
        return os.environ.get('STRATEGY3_API_BASE')

def load_strategy3_runtime_config(*, use_env: bool=False, profile_name: str | None=None) -> Strategy3Config:
    if use_env:
        return Strategy3Config.from_env(profile_name=profile_name)
    return Strategy3Config.from_profile(profile_name)
