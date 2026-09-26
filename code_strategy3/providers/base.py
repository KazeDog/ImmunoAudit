from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ProviderCapabilities:
    supported_batch_modes: tuple[str, ...] = ('sync', 'concurrent')

class BaseLLMProvider(ABC):
    capabilities = ProviderCapabilities()

    @abstractmethod
    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None, model: str, temperature: float) -> dict[str, Any]:
        raise NotImplementedError

    def generate_json_batch(self, *, requests: list[dict[str, Any]], model: str, temperature: float, max_workers: int, batch_mode: str='concurrent', poll_interval_seconds: int=15, completion_window: str='24h') -> list[dict[str, Any]]:
        self.ensure_batch_mode_supported(batch_mode)
        if batch_mode == 'sync':
            return [self.generate_json(system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature) for request in requests]
        worker_count = max(1, min(int(max_workers), len(requests)))
        ordered_results: list[dict[str, Any] | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {executor.submit(self.generate_json, system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature): idx for idx, request in enumerate(requests)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                ordered_results[idx] = future.result()
        return [result for result in ordered_results if result is not None]

    def ensure_batch_mode_supported(self, batch_mode: str) -> None:
        if batch_mode not in self.capabilities.supported_batch_modes:
            raise ValueError(f"{self.__class__.__name__} does not support batch_mode={batch_mode}. Supported modes: {', '.join(self.capabilities.supported_batch_modes)}")

    def supports_batch_mode(self, batch_mode: str) -> bool:
        return batch_mode in self.capabilities.supported_batch_modes

    @property
    def supported_batch_modes(self) -> tuple[str, ...]:
        return self.capabilities.supported_batch_modes

    def batch_mode_note(self) -> str:
        return ','.join(self.capabilities.supported_batch_modes)

    def _legacy_generate_json_batch(self, *, requests: list[dict[str, Any]], model: str, temperature: float) -> list[dict[str, Any]]:
        results = []
        for request in requests:
            result = self.generate_json(system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature)
            results.append(result)
        return results
