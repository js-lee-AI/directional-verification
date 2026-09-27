"""Train and evaluate one masked-diffusion student run. Needs the train extra."""

from __future__ import annotations

import random
from pathlib import Path

from .data import fingerprint, forward_pairs, read_json, validate_dataset, write_json
from .metrics import evaluate_predictions


def generate_predictions(model, tokenizer, mask_id, dtype, dataset, slots, device):
    """Decode an answer for every query in the evaluation split."""
    from .student import decode_name

    evaluation = set(dataset["evaluation_ids"])
    rows = []
    for query in dataset["queries"]:
        if query["id"] in evaluation:
            text = decode_name(model, tokenizer, mask_id, dtype,
                               f"{query['parent']}'s child is", slots, device)
            rows.append({"id": query["id"], "text": text})
    return rows


def train_student(dataset, child_parents, config, out, seed=0, device="cuda"):
    """Forward warm stage, then mixed reverse and forward SFT, then evaluation.

    Writes dataset.json, run.json, the LoRA adapter, the tokenizer,
    predictions.json and metrics.json to out, and returns the metrics.
    """
    import torch

    from .student import build_student, check_bidirectionality, train_phase

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory.")
    dimensions = validate_dataset(dataset, child_parents)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model, tokenizer, mask_id, dtype = build_student(config, device)
    check_bidirectionality(model, tokenizer, mask_id, dtype, device)
    forward = forward_pairs(child_parents, dataset["withheld_children"])
    rng.shuffle(forward)
    reverse = [(f"{row['parent']}'s child is", row["label"]) for row in dataset["queries"]]
    repeats = max(1, round(len(forward) / len(reverse)))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset.json", dataset)
    write_json(out / "run.json", {"config": config, "seed": seed, "dataset": dataset["name"],
                                  "dataset_fingerprint": fingerprint(dataset),
                                  "inventory_fingerprint": fingerprint(child_parents),
                                  "dimensions": dimensions, "reverse_repeats": repeats})
    print("Forward adaptation", flush=True)
    train_phase(model, tokenizer, mask_id, dtype, forward, config["warm_steps"], config, device)
    print("Mixed supervision", flush=True)
    train_phase(model, tokenizer, mask_id, dtype, reverse * repeats + forward,
                config["sft_steps"], config, device)
    # The frozen embedding rows are rebuilt when the adapter is loaded.
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    tokenizer.save_pretrained(out / "tokenizer")
    predictions = generate_predictions(model, tokenizer, mask_id, dtype, dataset,
                                       config["answer_slots"], device)
    write_json(out / "predictions.json", predictions)
    metrics = evaluate_predictions(dataset, child_parents, predictions)
    write_json(out / "metrics.json", metrics)
    return metrics


def load_run(run, device="cuda"):
    """Rebuild a trained student from a run directory."""
    from .student import build_student

    config = read_json(Path(run) / "run.json")["config"]
    return build_student(config, device, Path(run) / "adapter")
