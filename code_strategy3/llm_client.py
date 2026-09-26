import json
from typing import Any
from .cache import Strategy3Cache
from .config import Strategy3Config
from .providers import GeminiProvider
from .providers import QwenProvider
from .providers.base import BaseLLMProvider

def build_provider(config: Strategy3Config) -> BaseLLMProvider:
    if config.provider == 'gemini':
        return GeminiProvider(api_key=config.require_api_key(), timeout_seconds=config.timeout_seconds, max_retries=config.max_retries)
    if config.provider == 'qwen':
        return QwenProvider(api_key=config.require_api_key(), api_base=config.require_api_base(), timeout_seconds=config.timeout_seconds, max_retries=config.max_retries)
    raise ValueError(f'Unsupported provider: {config.provider}')

class Strategy3LLMClient:

    def __init__(self, config: Strategy3Config):
        self.config = config
        self.provider = build_provider(config)
        self.cache = Strategy3Cache(config.cache_dir)

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None, use_cache: bool=True) -> dict[str, Any]:
        payload = {'provider': self.config.provider, 'model': self.config.model, 'temperature': self.config.temperature, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema}
        cache_key = self.cache.build_key(payload)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        result = self.provider.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, model=self.config.model, temperature=self.config.temperature)
        result['cache_key'] = cache_key
        result['request'] = json.loads(json.dumps(payload, ensure_ascii=False))
        if use_cache:
            self.cache.set(cache_key, result)
        return result

    def batch_generate_json(self, *, requests: list[dict[str, Any]], use_cache: bool=True, max_workers: int=4, batch_mode: str='concurrent', poll_interval_seconds: int=15, completion_window: str='24h') -> list[dict[str, Any]]:
        if not requests:
            return []
        self.provider.ensure_batch_mode_supported(batch_mode)
        results: list[dict[str, Any] | None] = [None] * len(requests)
        uncached_requests: list[dict[str, Any]] = []
        uncached_indices: list[int] = []
        uncached_cache_keys: list[str] = []
        for idx, request in enumerate(requests):
            payload = {'provider': self.config.provider, 'model': self.config.model, 'temperature': self.config.temperature, 'system_prompt': request['system_prompt'], 'user_prompt': request['user_prompt'], 'schema': request.get('schema')}
            cache_key = self.cache.build_key(payload)
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    results[idx] = cached
                    continue
            uncached_requests.append(request)
            uncached_indices.append(idx)
            uncached_cache_keys.append(cache_key)
        uncached_results = self.provider.generate_json_batch(requests=uncached_requests, model=self.config.model, temperature=self.config.temperature, max_workers=max_workers, batch_mode=batch_mode, poll_interval_seconds=poll_interval_seconds, completion_window=completion_window)
        for local_idx, result in enumerate(uncached_results):
            global_idx = uncached_indices[local_idx]
            cache_key = uncached_cache_keys[local_idx]
            payload = {'provider': self.config.provider, 'model': self.config.model, 'temperature': self.config.temperature, 'system_prompt': uncached_requests[local_idx]['system_prompt'], 'user_prompt': uncached_requests[local_idx]['user_prompt'], 'schema': uncached_requests[local_idx].get('schema')}
            result['cache_key'] = cache_key
            result['request'] = json.loads(json.dumps(payload, ensure_ascii=False))
            results[global_idx] = result
            if use_cache:
                self.cache.set(cache_key, result)
        return [result for result in results if result is not None]
