"""Content hashes, a resumable call checkpoint, and a small thread pool."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so JSON stays parseable."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Dataset line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("Dataset contains no examples")
    return rows


class CallCache:
    """Append-only JSONL checkpoint so an interrupted run resumes without repeat calls."""

    def __init__(self, path: Path, *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if enabled and path.exists():
            for row in read_jsonl(path):
                self._entries[str(row["key"])] = row["value"]

    @staticmethod
    def key(*parts: Any) -> str:
        return digest(json.dumps([str(part) for part in parts]))

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps({"key": key, "value": json_safe(value)}, sort_keys=True)
        with self._lock:
            self._entries[key] = value
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def run_parallel(function: Any, items: list[Any], workers: int) -> list[Any]:
    """Apply ``function`` to ``items`` in order, using threads when workers > 1."""
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))
