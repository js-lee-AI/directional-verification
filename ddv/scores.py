"""Prompts and scores for the two sentence directions.

The known direction scores the query parent after each candidate child
(eq. 1). The reverse direction scores each candidate after the parent, and the
domain-context (DC) correction subtracts the candidate's context-only score
(eq. 3).
"""

from __future__ import annotations

import numpy as np

KNOWN = "{child}'s parent is"
REVERSE = "{parent}'s child is"
CONTEXT = "The child is"


def known_pairs(parent, candidates):
    """(prompt, continuation) pairs for the known direction."""
    return [(KNOWN.format(child=child), parent) for child in candidates]


def reverse_pairs(parent, candidates):
    """(prompt, continuation) pairs for the requested (reverse) direction."""
    return [(REVERSE.format(parent=parent), child) for child in candidates]


def context_pairs(candidates):
    """(prompt, continuation) pairs for the context-only name score."""
    return [(CONTEXT, child) for child in candidates]


def mean_token(logprobs):
    """Mean log probability of the continuation tokens (eq. 1)."""
    logprobs = np.asarray(logprobs, dtype=np.float64)
    if logprobs.size == 0:
        raise ValueError("The continuation has no tokens.")
    return float(logprobs.sum() / logprobs.size)


def summed(mean, tokens):
    """Turn mean-token scores back into summed-token scores."""
    return np.asarray(mean, dtype=np.float64) * np.asarray(tokens, dtype=np.float64)


def domain_context(reverse, context, lam=1.0):
    """Reverse score minus lam times the context-only score (eq. 3).

    lam=1 is the DC reverse score. Pass mean-token or summed-token arrays, the
    same form for both arguments.
    """
    return np.asarray(reverse, dtype=np.float64) - lam * np.asarray(context, dtype=np.float64)
