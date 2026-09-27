"""Scoring candidates with a frozen causal language model. Needs the train extra."""

from __future__ import annotations

import numpy as np

from .scores import context_pairs, known_pairs, reverse_pairs


def load_teacher(spec, device="cuda"):
    """Load a teacher from {"model": ..., "revision": ...}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec.get("revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=spec.get("revision"), dtype=dtype).to(device).eval()
    return model, tokenizer


def score_pairs(model, tokenizer, pairs, batch_size=16):
    """Mean-token log probability of each continuation after its prompt.

    Raw prompts, no chat template. The continuation is scored after a leading
    space, and the prompt tokens must be a prefix of the full sequence.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    device = model.get_input_embeddings().weight.device
    means = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            prefixes = tokenizer([p for p, _ in batch], add_special_tokens=True).input_ids
            sequences = tokenizer([p + " " + target for p, target in batch],
                                  add_special_tokens=True).input_ids
            for prefix, sequence in zip(prefixes, sequences):
                if sequence[:len(prefix)] != prefix or len(sequence) <= len(prefix):
                    raise ValueError("Tokenizer changed the prompt/continuation boundary.")
            ids = torch.full((len(batch), max(map(len, sequences))),
                             tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for i, sequence in enumerate(sequences):
                ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
                attention[i, :len(sequence)] = 1
            logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            observed = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            for i, (prefix, sequence) in enumerate(zip(prefixes, sequences)):
                span = observed[i, len(prefix) - 1:len(sequence) - 1]
                means.append(span.sum().item() / len(span))
    return means


def continuation_lengths(tokenizer, pairs):
    """Number of continuation tokens that score_pairs averages over, per pair."""
    prefixes = tokenizer([p for p, _ in pairs], add_special_tokens=True).input_ids
    sequences = tokenizer([p + " " + target for p, target in pairs], add_special_tokens=True).input_ids
    return [len(s) - len(p) for p, s in zip(prefixes, sequences)]


def score_candidates(model, tokenizer, parent, candidates, direction="known", batch_size=16):
    """Mean-token scores of the candidates for one parent query.

    direction="known" scores the parent after "{child}'s parent is" (eq. 1).
    "reverse" scores each candidate after "{parent}'s child is", and "context"
    scores each candidate after "The child is".
    """
    if direction == "known":
        pairs = known_pairs(parent, candidates)
    elif direction == "reverse":
        pairs = reverse_pairs(parent, candidates)
    elif direction == "context":
        pairs = context_pairs(candidates)
    else:
        raise ValueError("direction must be 'known', 'reverse' or 'context'.")
    return np.asarray(score_pairs(model, tokenizer, pairs, batch_size))
