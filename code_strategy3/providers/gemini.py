import json
import time
from typing import Any
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from .base import BaseLLMProvider
from .base import ProviderCapabilities

class GeminiProvider(BaseLLMProvider):
    capabilities = ProviderCapabilities(supported_batch_modes=('sync', 'concurrent'))

    def __init__(self, *, api_key: str, timeout_seconds: int, max_retries: int):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None, model: str, temperature: float) -> dict[str, Any]:
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature, response_mime_type='application/json')
        if schema is not None:
            config.response_json_schema = schema
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(model=model, contents=user_prompt, config=config)
                parsed = self._extract_parsed_response(response)
                return {'provider': 'gemini', 'model': model, 'raw_response': response.to_json_dict(), 'parsed_response': parsed}
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                if self._is_non_retryable_error(exc):
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f'Gemini request failed after {self.max_retries} attempts: {last_error}')

    @staticmethod
    def _extract_parsed_response(response: types.GenerateContentResponse) -> dict[str, Any]:
        parsed = getattr(response, 'parsed', None)
        if isinstance(parsed, dict):
            return parsed
        if hasattr(parsed, 'model_dump'):
            return parsed.model_dump()
        text = getattr(response, 'text', '') or ''
        if text.strip():
            return json.loads(text)
        response_json = response.to_json_dict()
        candidates = response_json.get('candidates', [])
        if not candidates:
            raise ValueError('Gemini response contains no candidates.')
        parts = candidates[0].get('content', {}).get('parts', [])
        texts = [part.get('text', '') for part in parts if 'text' in part]
        merged = ''.join(texts).strip()
        if not merged:
            raise ValueError('Gemini response contains no text payload.')
        return json.loads(merged)

    @staticmethod
    def _is_non_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, genai_errors.ClientError):
            status_code = getattr(exc, 'status_code', None)
            return status_code is not None and 400 <= int(status_code) < 500 and (int(status_code) != 429)
        return False
