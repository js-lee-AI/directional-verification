"""Directional label distillation: score candidate answers in the direction a teacher knows.

    import ddv

    label = ddv.select_label(scores, candidates, parent)   # scores is (teachers, candidates)

Frozen teachers score the query parent after each candidate child, the
candidate with the highest mean score becomes the pseudo-label, and a student
learns the reverse query from those labels. Selection needs only numpy. Teacher
scoring and student training need the train extra, and torch is imported only
when one of those names is first used.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data import (FORMS, POOLS, TEACHERS, context_sums, forward_pairs, load_inventory, load_pool,
                   load_scores, parent_children, read_json, validate_dataset, write_json)
from .labels import (RULES, combine, select_index, select_label, select_labels, self_mask,
                     soft_agreement)
from .metrics import (InventoryMatcher, answer_flags, evaluate_predictions, holm,
                      label_accuracy, label_correct, mcnemar_p, open_match, paired_counts,
                      whole_match)
from .scores import (CONTEXT, KNOWN, REVERSE, context_pairs, domain_context, known_pairs,
                     mean_token, reverse_pairs, summed)

# Heavy names, imported on first attribute access (PEP 562).
_LAZY = {
    "load_teacher": "teacher",
    "score_pairs": "teacher",
    "score_candidates": "teacher",
    "continuation_lengths": "teacher",
    "build_student": "student",
    "train_phase": "student",
    "decode_name": "student",
    "train_student": "train",
    "generate_predictions": "train",
    "load_run": "train",
}

__all__ = [
    "__version__",
    # scores
    "KNOWN", "REVERSE", "CONTEXT", "known_pairs", "reverse_pairs", "context_pairs",
    "mean_token", "summed", "domain_context",
    # label selection
    "RULES", "combine", "select_index", "select_label", "select_labels", "self_mask",
    "soft_agreement",
    # data
    "TEACHERS", "POOLS", "FORMS", "load_inventory", "load_pool", "load_scores", "context_sums",
    "parent_children", "forward_pairs", "validate_dataset", "read_json", "write_json",
    # metrics
    "label_accuracy", "label_correct", "paired_counts", "mcnemar_p", "holm",
    "answer_flags", "evaluate_predictions", "InventoryMatcher", "open_match", "whole_match",
    # teachers and students, train extra
    "load_teacher", "score_pairs", "score_candidates", "continuation_lengths",
    "build_student", "train_phase", "decode_name", "train_student", "generate_predictions", "load_run",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
