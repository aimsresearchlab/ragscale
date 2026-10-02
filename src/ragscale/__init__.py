"""ragscale: audit whether a fixed RAG compression layer preserves a reader comparison."""

from ragscale.matrix import MAIN_PAPER_READERS, load_interaction_matrix
from ragscale.presets import write_starter_project
from ragscale.replay import ReplayAuditResult, replay_audit, write_replay_audit
from ragscale.runner import run_config
from ragscale.scoring import BUILTIN_METRICS, exact_match, normalize_answer, token_f1

__version__ = "0.5.1"

__all__ = [
    "run_config", "write_starter_project",
    "ReplayAuditResult", "replay_audit", "write_replay_audit",
    "BUILTIN_METRICS", "exact_match", "token_f1", "normalize_answer",
    "load_interaction_matrix", "MAIN_PAPER_READERS",
]
