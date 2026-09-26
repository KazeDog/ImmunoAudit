from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import requests
from .base import BaseLLMProvider
from .base import ProviderCapabilities

class QwenProvider(BaseLLMProvider):
    capabilities = ProviderCapabilities(supported_batch_modes=('sync', 'concurrent', 'bailian_async'))

    def __init__(self, *, api_key: str, api_base: str, timeout_seconds: int, max_retries: int):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None, model: str, temperature: float) -> dict[str, Any]:
        url = f'{self.api_base}/chat/completions'
        payload = self._build_chat_payload(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, model=model, temperature=temperature)
        headers = self._json_headers()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                response_json = response.json()
                text = self._extract_text(response_json)
                parsed = self._parse_json_text(text)
                return {'provider': 'qwen', 'model': model, 'raw_response': response_json, 'parsed_response': parsed}
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f'Qwen request failed after {self.max_retries} attempts: {last_error}')

    def generate_json_batch(self, *, requests: list[dict[str, Any]], model: str, temperature: float, max_workers: int, batch_mode: str='concurrent', poll_interval_seconds: int=15, completion_window: str='24h') -> list[dict[str, Any]]:
        if not requests:
            return []
        self.ensure_batch_mode_supported(batch_mode)
        if batch_mode == 'bailian_async':
            return self._generate_json_batch_async(requests=requests, model=model, temperature=temperature, poll_interval_seconds=poll_interval_seconds, completion_window=completion_window)
        ordered_results: list[dict[str, Any] | None] = [None] * len(requests)
        failed_indices: list[int] = []
        worker_count = max(1, min(int(max_workers), len(requests)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {executor.submit(self.generate_json, system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature): idx for idx, request in enumerate(requests)}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception:
                    failed_indices.append(idx)
        for idx in failed_indices:
            request = requests[idx]
            time.sleep(1.0)
            ordered_results[idx] = self.generate_json(system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature)
        return [result for result in ordered_results if result is not None]

    def _generate_json_batch_async(self, *, requests: list[dict[str, Any]], model: str, temperature: float, poll_interval_seconds: int, completion_window: str) -> list[dict[str, Any]]:
        jsonl_lines = []
        request_order = []
        for idx, request in enumerate(requests):
            custom_id = str(request.get('custom_id') or request.get('sample_id') or f'request-{idx}')
            request_order.append(custom_id)
            payload = self._build_chat_payload(system_prompt=str(request['system_prompt']), user_prompt=str(request['user_prompt']), schema=request.get('schema'), model=model, temperature=temperature)
            jsonl_lines.append(json.dumps({'custom_id': custom_id, 'method': 'POST', 'url': '/v1/chat/completions', 'body': payload}, ensure_ascii=False))
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.jsonl', delete=False) as handle:
            handle.write('\n'.join(jsonl_lines) + '\n')
            temp_path = Path(handle.name)
        try:
            input_file = self._upload_batch_file(temp_path)
            batch_object = self._create_batch_job(input_file_id=str(input_file['id']), completion_window=completion_window)
            batch_id = str(batch_object['id'])
            batch_object = self._poll_batch_until_terminal(batch_id=batch_id, poll_interval_seconds=poll_interval_seconds)
            status = str(batch_object.get('status', ''))
            if status != 'completed':
                raise RuntimeError(f'Bailian batch job did not complete successfully: {status}')
            output_file_id = batch_object.get('output_file_id')
            if not output_file_id:
                raise RuntimeError('Bailian batch job completed without output_file_id.')
            output_content = self._download_file_content(str(output_file_id))
            return self._parse_batch_output_lines(output_content, request_order)
        finally:
            temp_path.unlink(missing_ok=True)

    def _build_chat_payload(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None, model: str, temperature: float) -> dict[str, Any]:
        schema_text = build_compact_schema_instruction(schema)
        user_content = user_prompt if schema_text is None else '\n\n'.join([user_prompt, 'Output format requirements:', schema_text])
        return {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}], 'temperature': temperature, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}

    def _upload_batch_file(self, file_path: Path) -> dict[str, Any]:
        url = f'{self.api_base}/files'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        with file_path.open('rb') as file_handle:
            response = requests.post(url, headers=headers, data={'purpose': 'batch'}, files={'file': (file_path.name, file_handle, 'application/jsonl')}, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _create_batch_job(self, *, input_file_id: str, completion_window: str) -> dict[str, Any]:
        url = f'{self.api_base}/batches'
        payload = {'input_file_id': input_file_id, 'endpoint': '/v1/chat/completions', 'completion_window': completion_window}
        response = requests.post(url, headers=self._json_headers(), json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _retrieve_batch_job(self, batch_id: str) -> dict[str, Any]:
        url = f'{self.api_base}/batches/{batch_id}'
        response = requests.get(url, headers=self._json_headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _poll_batch_until_terminal(self, *, batch_id: str, poll_interval_seconds: int) -> dict[str, Any]:
        terminal_statuses = {'completed', 'failed', 'expired', 'cancelled'}
        while True:
            batch_object = self._retrieve_batch_job(batch_id)
            status = str(batch_object.get('status', ''))
            if status in terminal_statuses:
                return batch_object
            time.sleep(max(1, int(poll_interval_seconds)))

    def _download_file_content(self, file_id: str) -> str:
        url = f'{self.api_base}/files/{file_id}/content'
        response = requests.get(url, headers=self._json_headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _parse_batch_output_lines(self, output_content: str, request_order: list[str]) -> list[dict[str, Any]]:
        parsed_by_custom_id: dict[str, dict[str, Any]] = {}
        for raw_line in output_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_json = json.loads(line)
            custom_id = str(line_json.get('custom_id', ''))
            response_body = line_json.get('response', {}).get('body', {})
            text = self._extract_text(response_body)
            parsed_by_custom_id[custom_id] = {'provider': 'qwen', 'model': response_body.get('model'), 'raw_response': response_body, 'parsed_response': json.loads(text)}
        results = []
        missing = []
        for custom_id in request_order:
            result = parsed_by_custom_id.get(custom_id)
            if result is None:
                missing.append(custom_id)
                continue
            results.append(result)
        if missing:
            raise RuntimeError(f'Missing batch outputs for custom_ids: {missing[:5]}')
        return results

    def _json_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        choices = response_json.get('choices', [])
        if not choices:
            raise ValueError('Qwen response contains no choices.')
        message = choices[0].get('message', {})
        content = message.get('content', '')
        if isinstance(content, list):
            content = ''.join((part.get('text', '') for part in content if isinstance(part, dict)))
        content = str(content).strip()
        if not content:
            raise ValueError('Qwen response contains no message content.')
        return content

    @classmethod
    def _parse_json_text(cls, text: str) -> dict[str, Any]:
        cleaned = cls._strip_code_fences(str(text).strip())
        if not cleaned:
            raise ValueError('Qwen response contains empty JSON text.')
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = cls._extract_first_json_object(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError(f'Expected top-level JSON object, got: {type(parsed).__name__}')
        return parsed

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub('^```[a-zA-Z0-9_-]*\\s*', '', stripped, count=1)
            stripped = re.sub('\\s*```$', '', stripped, count=1)
        return stripped.strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Any:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != '{':
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise ValueError('No valid JSON object found in provider response text.')

def build_compact_schema_instruction(schema: dict[str, Any] | None) -> str | None:
    if not schema:
        return None
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    field_specs = []
    for field_name in required:
        field_schema = properties.get(field_name, {})
        field_type = field_schema.get('type', 'value')
        if 'enum' in field_schema:
            field_specs.append(f"{field_name}: one of {field_schema['enum']}")
        elif field_type == 'array':
            item_type = field_schema.get('items', {}).get('type', 'value')
            field_specs.append(f'{field_name}: array of {item_type}')
        else:
            field_specs.append(f'{field_name}: {field_type}')
    return 'Return a JSON object with fields: ' + '; '.join(field_specs) + '.'
