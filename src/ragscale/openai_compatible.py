"""Built-in adapters for OpenAI-compatible Chat Completions endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_BASE_URL = "https://api.openai.com/v1"

_DEFAULT_SYSTEM_PROMPTS = {
    "compressor": (
        "Compress the supplied evidence for answering the question. Preserve the facts "
        "needed to answer it. Return only the compressed evidence."
    ),
    "reader": (
        "Answer the question using the supplied evidence. Reply with only the answer "
        "span: a name, number, date, yes or no, or a short phrase. No sentence, no "
        "explanation, no punctuation after the answer."
    ),
    "judge": (
        "You grade short answers. Decide whether the predicted answer conveys the same "
        "fact as at least one reference answer, allowing different wording. Reply with "
        "exactly one character: 1 if it matches, 0 if it does not."
    ),
}

_DEFAULT_USER_PROMPT = "Question:\n{question}\n\nEvidence:\n{evidence}"
_DEFAULT_JUDGE_PROMPT = (
    "Question:\n{question}\n\nReference answers:\n{references}\n\n"
    "Predicted answer:\n{prediction}"
)


def _new_client(**kwargs: Any) -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - guarded by package dependency
        raise ImportError(
            "The OpenAI-compatible adapter requires the 'openai' package. "
            "Install ragscale again with its declared dependencies."
        ) from error
    return OpenAI(**kwargs)


_BACKOFF_SCALE = 1.0


def _is_rate_limit(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    return "429" in str(error) or "rate-limited" in str(error).lower()


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_text(response: Any) -> str:
    """Return the visible text of the first choice, or an empty string."""
    choices = _read_attr(response, "choices", [])
    if not choices:
        raise ValueError("OpenAI-compatible endpoint returned no choices")
    message = _read_attr(choices[0], "message")
    content = _read_attr(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                text = _read_attr(item, "text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return ""


@dataclass
class OpenAICompatibleAdapter:
    """Callable compressor or reader backed by a Chat Completions endpoint."""

    role: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str | None = None
    user_prompt_template: str = _DEFAULT_USER_PROMPT
    parameters: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 2
    rate_limit_retries: int = 6
    empty_retries: int = 2
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.role not in _DEFAULT_SYSTEM_PROMPTS:
            raise ValueError("OpenAI-compatible adapter role must be compressor, reader, or judge")
        if not self.model:
            raise ValueError("OpenAI-compatible adapters require model")
        if not self.base_url:
            raise ValueError("OpenAI-compatible adapter base_url cannot be empty")
        if not self.api_key_env:
            raise ValueError("OpenAI-compatible adapters require api_key_env")
        if self.timeout <= 0:
            raise ValueError("OpenAI-compatible adapter timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("OpenAI-compatible adapter max_retries cannot be negative")
        if self.rate_limit_retries < 0:
            raise ValueError("OpenAI-compatible adapter rate_limit_retries cannot be negative")
        forbidden = {"model", "messages", "stream"} & set(self.parameters)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"OpenAI-compatible parameters cannot override: {names}")
        if self.system_prompt is None:
            self.system_prompt = _DEFAULT_SYSTEM_PROMPTS[self.role]
        if self.role == "judge" and self.user_prompt_template == _DEFAULT_USER_PROMPT:
            self.user_prompt_template = _DEFAULT_JUDGE_PROMPT

    @classmethod
    def from_config(cls, config: dict[str, Any], *, role: str) -> "OpenAICompatibleAdapter":
        default_prompt = _DEFAULT_JUDGE_PROMPT if role == "judge" else _DEFAULT_USER_PROMPT
        return cls(
            role=role,
            model=str(config.get("model", "")),
            base_url=str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            system_prompt=config.get("system_prompt"),
            user_prompt_template=str(config.get("user_prompt_template", default_prompt)),
            parameters=dict(config.get("parameters") or {}),
            headers={str(key): str(value) for key, value in (config.get("headers") or {}).items()},
            timeout=float(config.get("timeout", 60.0)),
            max_retries=int(config.get("max_retries", 2)),
            rate_limit_retries=int(config.get("rate_limit_retries", 6)),
            empty_retries=int(config.get("empty_retries", 2)),
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "type": "openai_compatible",
            "endpoint": "chat_completions",
            "role": self.role,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "parameters": self.parameters,
            "header_names": sorted(self.headers),
            "system_prompt_hash": _sha256(str(self.system_prompt)),
            "user_prompt_template_hash": _sha256(self.user_prompt_template),
        }

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"OpenAI-compatible adapter needs environment variable {self.api_key_env}"
                )
            self._client = _new_client(
                api_key=api_key,
                base_url=self.base_url,
                default_headers=self.headers or None,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _complete(self, values: dict[str, str]) -> dict[str, Any]:
        try:
            user_prompt = self.user_prompt_template.format_map(values)
        except KeyError as error:
            raise ValueError(
                f"Unknown OpenAI-compatible prompt placeholder: {error.args[0]}"
            ) from error
        request = dict(self.parameters)
        request.update({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        })
        client = self._get_client()
        response = self._create_with_backoff(client, request)
        text = _response_text(response)
        empty_attempts = 0
        while not text and empty_attempts < self.empty_retries:
            empty_attempts += 1
            response = self._create_with_backoff(client, request)
            text = _response_text(response)
        if not text and self.role != "reader":
            finish = _read_attr(_read_attr(response, "choices", [None])[0], "finish_reason")
            raise ValueError(
                f"OpenAI-compatible {self.role} returned no visible text for model {self.model} "
                f"(finish_reason={finish}). Hidden reasoning may have used the token budget."
            )
        usage = _read_attr(response, "usage")
        prompt_tokens = _read_attr(usage, "prompt_tokens")
        completion_tokens = _read_attr(usage, "completion_tokens")
        total_tokens = _read_attr(usage, "total_tokens")
        reported_cost = _read_attr(usage, "cost")
        if reported_cost is None and not isinstance(usage, dict):
            extra = getattr(usage, "model_extra", None) or {}
            reported_cost = extra.get("cost") if isinstance(extra, dict) else None
        response_id = _read_attr(response, "id") or _read_attr(response, "_request_id")
        metadata = {
            "adapter_type": "openai_compatible",
            "endpoint": "chat_completions",
            "base_url": self.base_url,
            "model": _read_attr(response, "model", self.model),
            "configured_model": self.model,
            "provider": _read_attr(response, "provider"),
            "response_id": response_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": _read_attr(_read_attr(response, "choices", [None])[0], "finish_reason"),
            "empty_response": not text,
            "empty_retries_used": empty_attempts,
            "prompt_hash": _sha256(json.dumps(request["messages"], sort_keys=True)),
        }
        return {
            "text": text,
            "tokens": int(total_tokens) if total_tokens is not None else None,
            "cost_usd": float(reported_cost) if reported_cost is not None else None,
            "metadata": metadata,
        }

    def _create_with_backoff(self, client: Any, request: dict[str, Any]) -> Any:
        """Retry upstream rate limits with exponential backoff; the SDK handles transport retries."""
        attempt = 0
        while True:
            try:
                return client.chat.completions.create(**request)
            except Exception as error:
                if _is_rate_limit(error) and attempt < self.rate_limit_retries:
                    delay = min(2 ** attempt, 30) * _BACKOFF_SCALE
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible {self.role} request failed for model {self.model}: {error}"
                ) from error

    def __call__(
        self,
        question: str,
        evidence: str,
        condition: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if self.role == "judge":
            raise ValueError("Judge adapters score answers; configure them under scoring.metrics")
        default_condition = "compressed" if self.role == "reader" else "compile"
        return self._complete({
            "question": question,
            "evidence": evidence,
            "condition": condition or default_condition,
        })

    def judge(self, prediction: str, references: list[str], case: dict[str, Any]) -> dict[str, Any]:
        """Grade one prediction against the references; the returned score is 0 or 1."""
        if self.role != "judge":
            raise ValueError("Only judge adapters can grade answers")
        result = self._complete({
            "question": str(case.get("question", "")),
            "references": "\n".join(str(value) for value in references),
            "prediction": str(prediction),
        })
        result["score"] = parse_judge_verdict(result["text"])
        return result


def parse_judge_verdict(text: str) -> float:
    """Map a judge reply to 0 or 1. Anything else raises rather than scoring silently."""
    token = str(text).strip().strip(".").strip().lower()
    if token in {"1", "yes", "true", "correct", "match"}:
        return 1.0
    if token in {"0", "no", "false", "incorrect", "mismatch"}:
        return 0.0
    raise ValueError(f"Judge reply is not a 0/1 verdict: {text[:80]!r}")


def build_openai_compatible_adapter(
    config: dict[str, Any], *, role: str
) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter.from_config(config, role=role)
