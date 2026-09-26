import hashlib
import json
from pathlib import Path
from typing import Any

class Strategy3Cache:

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f'{key}.json'

    def build_key(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path_for_key(key)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
