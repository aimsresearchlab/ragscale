"""Load the paper's released 176,864-row interaction matrix as a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_INSTALLED_DATA_DIR = Path(__file__).resolve().parent / "data"
_SOURCE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

MAIN_PAPER_READERS = frozenset({
    "google/gemma-3-12b-it",
    "microsoft/phi-4",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen3-32b",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-r1-distill-llama-70b",
    "x-ai/grok-4.1-fast",
    "qwen/qwen3-8b",
    "bytedance-seed/seed-2.0-mini",
})


def load_interaction_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Return the bundled matrix. See data/README_DATA.md for the column schema.

    The matrix is a broad interaction artifact. A shared ``method`` label does
    not prove two readers saw byte-identical compressed evidence; join the
    paper's artifact hashes before any fixed-artifact claim.
    """
    if path is None:
        for directory in (_INSTALLED_DATA_DIR, _SOURCE_DATA_DIR):
            for name in ("interaction_matrix.csv.gz", "interaction_matrix.csv"):
                candidate = directory / name
                if candidate.exists():
                    path = candidate
                    break
            if path is not None:
                break
    if path is None:
        raise FileNotFoundError("interaction_matrix.csv.gz is not bundled in this checkout")
    return pd.read_csv(path)
