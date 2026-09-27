"""Reading, writing and checking the corpus, the candidate pools and the scores."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .scores import domain_context, summed

TEACHERS = ("qwen", "olmo", "mistral", "llama")
POOLS = ("screened", "uniform64", "lexical64")
FORMS = ("known_mean", "known_sum", "reverse_mean", "reverse_sum", "dc_mean", "dc_sum")


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        path.write_bytes(gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0))
        return
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def candidate_fingerprint(queries):
    return fingerprint([[r["id"], r["parent"], r["candidates"]] for r in queries])


def parent_children(child_parents):
    result = {}
    for child, parents in child_parents.items():
        for parent in parents:
            result.setdefault(parent, set()).add(child)
    return result


def forward_pairs(child_parents, withheld_children):
    withheld = set(withheld_children)
    return [(f"{child}'s parent is", parent)
            for child, parents in child_parents.items() if child not in withheld
            for parent in parents]


def validate_dataset(data, child_parents):
    queries = data["queries"]
    ids = [r["id"] for r in queries]
    if not queries or len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be nonempty and unique.")
    inventory = set(child_parents)
    gold = parent_children(child_parents)
    for row in queries:
        candidates = row["candidates"]
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"Invalid candidate pool: {row['id']}")
        if not set(candidates) <= inventory or row["parent"] not in gold:
            raise ValueError(f"Unknown name: {row['id']}")
        if row["label"] not in candidates or row["label"] == row["parent"]:
            raise ValueError(f"Invalid selected label: {row['id']}")
    withheld = set(data["withheld_children"])
    if not withheld <= inventory:
        raise ValueError("Unknown withheld child.")
    retained = forward_pairs(child_parents, withheld)
    exposed = {p for _, p in retained}
    expected = [r["id"] for r in queries if r["parent"] not in exposed]
    if data["evaluation_ids"] != expected:
        raise ValueError("Evaluation IDs do not match the forward-exposure filter.")
    return {"queries": len(queries), "evaluated_queries": len(expected),
            "candidate_slots": sum(len(r["candidates"]) for r in queries),
            "withheld_children": len(withheld), "forward_pairs": len(retained)}


def load_inventory(root="data"):
    """The corpus as {child: [parents]}."""
    return read_json(Path(root) / "inventory.json")["child_parents"]


def load_pool(name, root="data"):
    """A candidate pool with its queries, withheld children and evaluation IDs."""
    return read_json(Path(root) / f"{name}.json")


def load_scores(name, root="data", teachers=TEACHERS, forms=("known_mean",), lam=1.0):
    """Teacher channels for a bundled pool, as {form: [channel per teacher]}.

    A channel holds one score array per query, in candidate order. known_mean is
    the known-direction score of eq. 1. The other forms come from the comparator
    files and are built as in Section 2.2, with lam the coefficient of eq. 3 in
    the dc forms.
    """
    root = Path(root)
    queries = load_pool(name, root)["queries"]
    expected = candidate_fingerprint(queries)
    out = {form: [] for form in forms}
    for teacher in teachers:
        known = read_json(root / "scores" / f"{name}_{teacher}.json.gz")
        if known["candidate_fingerprint"] != expected:
            raise ValueError(f"Scores do not match the {name} candidates: {teacher}")
        comparator = None
        if any(form != "known_mean" for form in forms):
            comparator = read_json(root / "scores" / f"{name}_{teacher}_reverse.json.gz")
            if comparator["candidate_fingerprint"] != expected:
                raise ValueError(f"Comparator scores do not match the {name} candidates: {teacher}")
        for form in forms:
            out[form].append(_form(form, queries, known, comparator, lam))
    return out


def _form(form, queries, known, comparator, lam=1.0):
    if form == "known_mean":
        return [np.asarray(v, dtype=np.float64) for v in known["scores"]]
    if form == "known_sum":
        return [summed(v, n) for v, n in zip(known["scores"], comparator["parent_tokens"])]
    mean, tokens = comparator["reverse_mean"], comparator["reverse_tokens"]
    if form == "reverse_mean":
        return [np.asarray(v, dtype=np.float64) for v in mean]
    if form == "reverse_sum":
        return [summed(v, n) for v, n in zip(mean, tokens)]
    cm, ct = comparator["context_mean"], comparator["context_tokens"]
    if form == "dc_mean":
        return [domain_context(v, [cm[c] for c in row["candidates"]], lam)
                for v, row in zip(mean, queries)]
    if form == "dc_sum":
        return [domain_context(summed(v, n), summed([cm[c] for c in row["candidates"]],
                                                    [ct[c] for c in row["candidates"]]), lam)
                for v, n, row in zip(mean, tokens, queries)]
    raise ValueError(f"Unknown score form {form!r}, choose from {FORMS}.")


def context_sums(name, root="data", teachers=TEACHERS):
    """Summed context-only name scores as {name: [score per teacher]}."""
    out = {}
    for teacher in teachers:
        comparator = read_json(Path(root) / "scores" / f"{name}_{teacher}_reverse.json.gz")
        for child, mean in comparator["context_mean"].items():
            out.setdefault(child, []).append(float(summed(mean, comparator["context_tokens"][child])))
    return out
