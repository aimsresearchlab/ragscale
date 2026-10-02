import pytest

from ragscale.scoring import (
    answer_contains,
    exact_match,
    normalize_answer,
    rouge_l,
    token_f1,
    token_precision,
    token_recall,
)


def test_normalization_and_alias_exact_match():
    assert normalize_answer("The, Paris!") == "paris"
    assert exact_match("Paris.", ["City of Paris", "the Paris"]) == 1


def test_token_scores_use_best_reference():
    references = ["Jane Austen", "Austen"]
    assert token_f1("Jane Austen wrote it", references) == pytest.approx(2 / 3)
    assert token_precision("Jane Austen wrote it", references) == pytest.approx(0.5)
    assert token_recall("Jane Austen wrote it", references) == 1
    assert rouge_l("Jane Austen wrote it", references) == pytest.approx(2 / 3)
    assert answer_contains("The author was Jane Austen.", references) == 1


def test_empty_normalized_reference_is_rejected():
    with pytest.raises(ValueError, match="normalize to empty"):
        exact_match("", [")"])
