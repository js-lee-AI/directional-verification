"""Answer matching, student metrics and paired label comparisons."""

import difflib
import unicodedata
from math import comb

from .data import parent_children


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(c.lower() for c in text if c.isalnum() or c == " ").strip()


def open_match(text, names):
    text = normalize(text)
    return any(normalize(name) and normalize(name) in text for name in names)


def whole_match(text, names):
    text = " ".join(normalize(text).split())
    return any(text == " ".join(normalize(name).split()) and normalize(name)
               for name in names)


class InventoryMatcher:
    def __init__(self, inventory):
        self.inventory = sorted(inventory)
        self.tokens = {}
        self.cache = {}
        for index, name in enumerate(self.inventory):
            for token in name.lower().split():
                self.tokens.setdefault(token, set()).add(index)

    def match(self, text):
        if not text:
            return ""
        if text in self.cache:
            return self.cache[text]
        candidates = set()
        for token in text.lower().split():
            candidates.update(self.tokens.get(token, ()))
        # Ties go to whichever tied index comes first when iterating the Python set of
        # sorted-inventory indices. That order is deterministic, and it is the one the
        # paper's runs used, so every stored mapping is reproduced.
        indices = candidates if candidates else range(len(self.inventory))
        best = max(indices, key=lambda i: difflib.SequenceMatcher(
            None, text.lower(), self.inventory[i].lower()).ratio())
        self.cache[text] = self.inventory[best]
        return self.cache[text]


FLAGS = ("label_accuracy", "open_accuracy", "whole_accuracy", "inventory_accuracy",
         "whole_selected_label", "inventory_selected_label", "open_selected_label")


def answer_flags(dataset, child_parents, predictions, matcher=None):
    """Per-query flags for the evaluation split, in dataset order.

    The *_accuracy flags credit any true child of the query parent. The
    *_selected_label flags check the answer against the query's training label.
    """
    gold = parent_children(child_parents)
    matcher = matcher or InventoryMatcher(child_parents)
    by_id = {}
    for row in predictions:
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        by_id[row["id"]] = row["text"]
    evaluation = set(dataset["evaluation_ids"])
    if set(by_id) != evaluation:
        raise ValueError("Prediction IDs must exactly match the evaluation split.")
    if not evaluation:
        raise ValueError("The evaluation split is empty.")
    rows = []
    for row in dataset["queries"]:
        if row["id"] not in evaluation:
            continue
        text, label = by_id[row["id"]], row["label"]
        names = gold[row["parent"]]
        matched = matcher.match(text)
        rows.append({"id": row["id"],
                     "label_accuracy": label in names,
                     "open_accuracy": bool(open_match(text, names)),
                     "whole_accuracy": bool(whole_match(text, names)),
                     "inventory_accuracy": matched in names,
                     "whole_selected_label": bool(whole_match(text, [label])),
                     "inventory_selected_label": matched == label,
                     "open_selected_label": bool(open_match(text, [label]))})
    return rows


def evaluate_predictions(dataset, child_parents, predictions, matcher=None):
    """Counts and percentages of the answer_flags over the evaluation split."""
    rows = answer_flags(dataset, child_parents, predictions, matcher)
    counts = {key: sum(r[key] for r in rows) for key in FLAGS}
    n = len(rows)
    return {"n": n, "counts": counts,
            "percent": {key: 100 * value / n for key, value in counts.items()}}


def label_correct(queries, labels, child_parents):
    """One flag per query, True when the label is a true child of the parent."""
    if len(labels) != len(queries):
        raise ValueError("Supply one label per query.")
    gold = parent_children(child_parents)
    return [label in gold[row["parent"]] for row, label in zip(queries, labels)]


def label_accuracy(queries, labels, child_parents):
    """Percent of labels that name a true child of the query parent."""
    flags = label_correct(queries, labels, child_parents)
    return 100 * sum(flags) / len(flags)


def paired_counts(left, right):
    """Wins and losses of left over right on paired correctness flags."""
    wins = sum(bool(a) and not bool(b) for a, b in zip(left, right))
    losses = sum(bool(b) and not bool(a) for a, b in zip(left, right))
    return wins, losses


def mcnemar_p(wins, losses):
    """Exact two-sided McNemar p value from the discordant counts."""
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(p_values):
    """Holm-adjusted p values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda k: p_values[k])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[j]))
        adjusted[j] = running
    return adjusted
