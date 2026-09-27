"""Pseudo-label selection from teacher score channels.

A channel is one teacher (or teacher and template) scoring every candidate of a
query. The primary rule averages the channels and takes the best candidate
other than the queried parent (eq. 2). The other rules are the aggregation
controls of Section 2.1 and Appendix A.
"""

from __future__ import annotations

import numpy as np

RULES = ("mean", "zscore", "probability", "borda", "rrf", "agreement")
SELF = -1e9


def self_mask(candidates, parent):
    """0 for eligible candidates, -1e9 for a candidate equal to the queried parent."""
    return np.array([SELF if name == parent else 0.0 for name in candidates])


def _softmax_rows(x):
    z = x - x.max(axis=1, keepdims=True)
    a = np.exp(z)
    return a / a.sum(axis=1, keepdims=True)


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def combine(scores, mask, rule="mean"):
    """Combine a (channels, candidates) array into one score per candidate."""
    a = np.asarray(scores, dtype=np.float64)
    if rule == "mean":
        return a.mean(0)
    if rule == "zscore":
        return ((a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)).mean(0)
    if rule == "probability":
        # The self slot is masked before each channel's softmax.
        return _softmax_rows(a + mask).mean(0)
    if rule in ("borda", "rrf"):
        eligible = np.flatnonzero(mask == 0)
        order = np.argsort(-a[:, eligible], axis=1, kind="stable")
        rank = np.argsort(order, axis=1, kind="stable") + 1
        values = len(eligible) - rank if rule == "borda" else 1 / (60 + rank)
        out = np.full(a.shape[1], SELF)
        out[eligible] = values.sum(0)
        return out
    raise ValueError(f"Unknown rule {rule!r}; 'agreement' needs select_labels.")


def select_index(scores, candidates, parent, rule="mean"):
    """Index of the selected candidate for one query. Ties keep candidate order."""
    mask = self_mask(candidates, parent)
    return int(np.argmax(combine(scores, mask, rule) + mask))


def select_label(scores, candidates, parent, rule="mean"):
    """The selected candidate name for one query."""
    return candidates[select_index(scores, candidates, parent, rule)]


def soft_agreement(distributions, iters=200, tol=1e-9, start=0.7):
    """Channel weights and per-query posteriors from eqs. 4 and 5.

    distributions[i] is a (channels, candidates) array of per-channel softmax
    distributions for query i.
    """
    n = len(distributions)
    channels = distributions[0].shape[0]
    weights = np.full(channels, start)
    posteriors = None
    for _ in range(iters):
        posteriors = []
        for i in range(n):
            m = distributions[i].shape[1]
            lp = np.zeros(m)
            for r in range(channels):
                q = (1 - weights[r]) / m + weights[r] * distributions[i][r]
                lp += np.log(q + 1e-12)
            lp -= lp.max()
            e = np.exp(lp)
            posteriors.append(e / e.sum())
        new = np.array([sum(float((posteriors[i] * distributions[i][r]).sum()) for i in range(n)) / n
                        for r in range(channels)])
        new = np.clip(new, 1e-3, 1 - 1e-3)
        if np.abs(new - weights).max() < tol:
            weights = new
            break
        weights = new
    return posteriors, weights


def _check(queries, channels):
    if not channels:
        raise ValueError("At least one teacher channel is required.")
    if any(len(channel) != len(queries) for channel in channels):
        raise ValueError("Each teacher must score every query.")
    arrays = []
    for index, row in enumerate(queries):
        values = np.asarray([channel[index] for channel in channels], dtype=np.float64)
        if values.shape != (len(channels), len(row["candidates"])):
            raise ValueError(f"Score shape mismatch: {row['id']}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite score: {row['id']}")
        arrays.append(values)
    return arrays


def select_labels(queries, channels, rule="mean"):
    """Select one label per query.

    queries are dicts with "id", "parent" and "candidates". channels[r][i] holds
    channel r's scores for the candidates of query i, in candidate order.
    """
    arrays = _check(queries, channels)
    if rule == "mean":
        labels = []
        for row, values in zip(queries, arrays):
            scores = values.mean(axis=0)
            eligible = [i for i, name in enumerate(row["candidates"]) if name != row["parent"]]
            if not eligible:
                raise ValueError(f"No eligible candidate: {row['id']}")
            best = max(eligible, key=lambda i: scores[i])
            labels.append(row["candidates"][best])
        return labels
    masks = [self_mask(row["candidates"], row["parent"]) for row in queries]
    if rule == "agreement":
        distributions = [np.array([_softmax(v + h) for v in values]) for values, h in zip(arrays, masks)]
        posteriors, _ = soft_agreement(distributions)
        return [row["candidates"][int(np.argmax(p + h))]
                for row, p, h in zip(queries, posteriors, masks)]
    return [row["candidates"][int(np.argmax(combine(values, h, rule) + h))]
            for row, values, h in zip(queries, arrays, masks)]
