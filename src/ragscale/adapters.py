"""Adapter contract: how ragscale calls a compressor, reader, or judge."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragscale.cache import digest
from ragscale.scoring import load_callable


@dataclass
class AdapterOutput:
    text: str
    tokens: int | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] | None = None


def adapter_output(value: Any) -> AdapterOutput:
    """Accept a string, a dict with text/answer/output, or an AdapterOutput."""
    if isinstance(value, AdapterOutput):
        return value
    if isinstance(value, str):
        return AdapterOutput(text=value)
    if isinstance(value, dict):
        text = value.get("text", value.get("answer", value.get("output")))
        if text is None:
            raise ValueError("Adapter dictionaries must contain text, answer, or output")
        return AdapterOutput(
            text=str(text),
            tokens=int(value["tokens"]) if value.get("tokens") is not None else None,
            cost_usd=float(value["cost_usd"]) if value.get("cost_usd") is not None else None,
            metadata=dict(value.get("metadata") or {}),
        )
    raise ValueError(f"Adapter returned unsupported type: {type(value).__name__}")


def invoke_adapter(function: Any, **context: Any) -> AdapterOutput:
    """Call an adapter with only the arguments it declares, timing the call."""
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = context if accepts_kwargs else {
        name: context[name] for name in signature.parameters if name in context
    }
    required = [
        name for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required:
        raise ValueError(f"Adapter requires unsupported arguments: {', '.join(required)}")
    started = time.perf_counter()
    output = adapter_output(function(**kwargs))
    metadata = dict(output.metadata or {})
    metadata.setdefault("measured_latency_ms", 1000 * (time.perf_counter() - started))
    output.metadata = metadata
    return output


def resolve_adapter(
    config: str | dict[str, Any], *, role: str, base_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return (callable, provenance) for a ``callable:`` or ``type: openai_compatible`` block."""
    value = {"callable": config} if isinstance(config, str) else dict(config)
    if value.get("callable"):
        spec = str(value["callable"])
        return load_callable(spec, base_dir), {"type": "python_callable", "callable": spec}
    adapter_type = str(value.get("type", ""))
    if adapter_type == "openai_compatible":
        from ragscale.openai_compatible import build_openai_compatible_adapter

        adapter = build_openai_compatible_adapter(value, role=role)
        return adapter, adapter.provenance
    raise ValueError(
        f"{role.capitalize()} adapter requires callable or type: openai_compatible"
    )


def provenance_hash(value: dict[str, Any]) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")))
