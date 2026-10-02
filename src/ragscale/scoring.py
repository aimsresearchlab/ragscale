"""Built-in answer scorers and custom-scorer loading for ragscale."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import string
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


BUILTIN_METRICS = (
    "exact_match",
    "token_f1",
    "token_precision",
    "token_recall",
    "rouge_l",
    "answer_contains",
)


def normalize_answer(value: Any) -> str:
    """Apply the standard lowercase, punctuation, article, and space normalization."""
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _references(values: str | Iterable[str]) -> list[str]:
    if isinstance(values, str):
        result = [values]
    else:
        result = [str(value) for value in values]
    if not result:
        raise ValueError("At least one reference answer is required")
    if not any(normalize_answer(value) for value in result):
        raise ValueError("Reference answers normalize to empty strings")
    return result


def exact_match(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(reference) for reference in _references(references)))


def _token_overlap(prediction: str, reference: str) -> tuple[float, float, float]:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        exact = float(predicted == expected)
        return exact, exact, exact
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall), precision, recall


def token_f1(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    return max(_token_overlap(prediction, reference)[0] for reference in _references(references))


def token_precision(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    return max(_token_overlap(prediction, reference)[1] for reference in _references(references))


def token_recall(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    return max(_token_overlap(prediction, reference)[2] for reference in _references(references))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    predicted = normalize_answer(prediction).split()
    values = []
    for reference in _references(references):
        expected = normalize_answer(reference).split()
        if not predicted or not expected:
            values.append(float(predicted == expected))
            continue
        overlap = _lcs_length(predicted, expected)
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        values.append(2 * precision * recall / (precision + recall) if overlap else 0.0)
    return max(values)


def answer_contains(prediction: str, references: str | Iterable[str], **_: Any) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalize_answer(reference) in normalized for reference in _references(references)))


_SCORERS: dict[str, Callable[..., float]] = {
    "exact_match": exact_match,
    "em": exact_match,
    "token_f1": token_f1,
    "f1": token_f1,
    "token_precision": token_precision,
    "token_recall": token_recall,
    "rouge_l": rouge_l,
    "answer_contains": answer_contains,
}


def _load_module_file(module_name: str, path: Path) -> Any:
    """Execute one adapter file under a path-specific name.

    Two projects commonly both call their file ``adapters.py``. Importing by
    bare name would return whichever one entered ``sys.modules`` first, so a
    file next to the configuration is loaded under a name derived from its
    location and executed afresh on every load.
    """
    unique = "ragscale_user_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module


def load_callable(spec: str, base_dir: str | Path | None = None) -> Callable[..., Any]:
    """Load ``module:function`` or ``module.function`` without framework coupling.

    When ``base_dir`` holds ``<module>.py``, that file is used directly; other
    names fall back to a normal import.
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        module_name, separator, attribute = spec.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Callable must use module:function syntax: {spec}")
    module = None
    if base_dir is not None:
        local_file = (Path(base_dir) / module_name.replace(".", "/")).with_suffix(".py")
        if local_file.is_file():
            module = _load_module_file(module_name, local_file.resolve())
    if module is None:
        module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if not callable(value):
        raise ValueError(f"Configured object is not callable: {spec}")
    return value


class JudgeScorer:
    """Scorer backed by an OpenAI-compatible judge; returns 0 or 1 per answer."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.provenance = dict(adapter.provenance)

    @property
    def provenance_hash(self) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(self.provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def __call__(self, prediction: str, references: list[str], case: dict[str, Any]) -> float:
        return float(self.adapter.judge(prediction, references, case)["score"])


def resolve_scorer(
    spec: str | dict[str, Any], base_dir: str | Path | None = None,
) -> tuple[str, Callable[..., float]]:
    """Resolve a built-in metric name, a custom callable, or an LLM judge."""
    if isinstance(spec, str):
        name = spec
        callable_spec = None
    else:
        name = str(spec.get("name") or spec.get("callable") or "custom")
        callable_spec = spec.get("callable")
        if not callable_spec and spec.get("type") == "openai_compatible":
            from ragscale.openai_compatible import build_openai_compatible_adapter

            return name, JudgeScorer(build_openai_compatible_adapter(spec, role="judge"))
    if callable_spec:
        scorer = load_callable(str(callable_spec), base_dir)
    else:
        scorer = _SCORERS.get(name)
        if scorer is None:
            choices = ", ".join(BUILTIN_METRICS)
            raise ValueError(f"Unknown metric '{name}'. Built-ins: {choices}; or provide callable")
    return name, scorer


def score_answer(
    scorer: Callable[..., float],
    prediction: str,
    references: list[str],
    case: dict[str, Any],
) -> float:
    context = {"prediction": prediction, "references": references, "case": case}
    signature = inspect.signature(scorer)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = context if accepts_kwargs else {
        name: context[name] for name in signature.parameters if name in context
    }
    value = float(scorer(**kwargs))
    if not 0 <= value <= 1:
        raise ValueError(f"Scorer returned {value}; scores must lie in [0, 1]")
    return value
