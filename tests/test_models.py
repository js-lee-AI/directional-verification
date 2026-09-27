"""CPU tests of teacher scoring and the student on a tiny random model. Need the train extra."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import ddv  # noqa: E402
from ddv.student import (bidirectional_mask, build_student, collate_answers,  # noqa: E402
                         decode_name, train_phase)

torch.set_num_threads(1)


def test_answer_slots_and_padding_targets(tiny_config):
    _, tokenizer, mask_id, _ = build_student(tiny_config, "cpu")
    slots = 4
    batch = [("First Child's parent is", "Parent")]
    prefix = tokenizer(batch[0][0], add_special_tokens=True).input_ids
    saw_pad_target = False
    for seed in range(10):
        ids, attention, labels = collate_answers(tokenizer, mask_id, batch, slots, random.Random(seed))
        assert torch.all(labels[0, :len(prefix)] == -100)
        assert labels[0, -1].item() == -100
        assert torch.all(attention == 1)
        chosen = labels != -100
        assert torch.all(ids[chosen] == mask_id)
        assert 1 <= int(chosen.sum()) <= slots
        saw_pad_target |= bool(torch.any(labels == tokenizer.pad_token_id))
    assert saw_pad_target


def test_mask_exposes_right_context(tiny_config):
    model, tokenizer, mask_id, dtype = build_student(tiny_config, "cpu")
    model.eval()
    a = tokenizer.convert_tokens_to_ids("A")
    b = tokenizer.convert_tokens_to_ids("B")
    ids = torch.tensor([[3, mask_id, a], [3, mask_id, b]])
    with torch.no_grad():
        causal = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        bidirectional = model(input_ids=ids, attention_mask=bidirectional_mask(
            torch.ones_like(ids), dtype)).logits
    assert float((causal[0, 1] - causal[1, 1]).abs().max()) < 1e-6
    assert float((bidirectional[0, 1] - bidirectional[1, 1]).abs().max()) > 1e-4
    mask = bidirectional_mask(torch.tensor([[1, 1, 0]]), dtype)
    assert mask[0, 0, 0, 0].item() == 0
    assert mask[0, 0, 0, 2].item() == torch.finfo(dtype).min


def test_teacher_score_alignment(tiny_config):
    model, tokenizer, _, _ = build_student(tiny_config, "cpu")
    pairs = [("First Child's parent is", "Parent Name"), ("Other parent is", "A")]
    observed = ddv.score_pairs(model, tokenizer, pairs, batch_size=2)
    expected = []
    model.eval()
    for prefix, target in pairs:
        context = tokenizer(prefix, add_special_tokens=True).input_ids
        sequence = tokenizer(prefix + " " + target, add_special_tokens=True).input_ids
        with torch.no_grad():
            logp = model(input_ids=torch.tensor([sequence])).logits[0].float().log_softmax(-1)
        values = [logp[j - 1, sequence[j]].item() for j in range(len(context), len(sequence))]
        expected.append(sum(values) / len(values))
    np.testing.assert_allclose(observed, expected, atol=1e-6)
    assert ddv.continuation_lengths(tokenizer, pairs) == [2, 1]


def test_score_rejects_changed_token_boundary(tiny_config):
    model, _, _, _ = build_student(tiny_config, "cpu")

    class BrokenTokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return SimpleNamespace(input_ids=[[3] if t == "prefix" else [4, 5] for t in texts])

    with pytest.raises(ValueError):
        ddv.score_pairs(model, BrokenTokenizer(), [("prefix", "target")])


def test_known_direction_scores_the_same_parent_after_each_child(tiny_model):
    model, tokenizer = ddv.load_teacher({"model": str(tiny_model)}, "cpu")
    candidates = ["First Child", "Other Child"]
    known = ddv.score_candidates(model, tokenizer, "Parent Name", candidates)
    pairs = [("First Child's parent is", "Parent Name"), ("Other Child's parent is", "Parent Name")]
    np.testing.assert_allclose(known, ddv.score_pairs(model, tokenizer, pairs))
    reverse = ddv.score_candidates(model, tokenizer, "Parent Name", candidates, direction="reverse")
    assert reverse.shape == (2,) and not np.allclose(known, reverse)


def test_training_and_adapter_roundtrip(tiny_config, tmp_path):
    torch.manual_seed(11)
    model, tokenizer, mask_id, dtype = build_student(tiny_config, "cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    samples = [("First Child's parent is", "Parent Name"), ("Parent Name's child is", "First Child")]
    train_phase(model, tokenizer, mask_id, dtype, samples, 2, tiny_config, "cpu")
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    prediction = decode_name(model, tokenizer, mask_id, dtype, "Parent Name's child is", 4, "cpu")
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    loaded, tok2, mask2, dtype2 = build_student(tiny_config, "cpu", tmp_path)
    assert decode_name(loaded, tok2, mask2, dtype2, "Parent Name's child is", 4, "cpu") == prediction
    loaded.eval()
    ids = torch.tensor([[3, mask_id, 4]])
    attention = bidirectional_mask(torch.ones_like(ids), dtype)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids, attention_mask=attention).logits,
                                   loaded(input_ids=ids, attention_mask=attention).logits)


def test_train_student_writes_a_reloadable_run(tiny_config, tmp_path):
    inventory = {"First Child": ["Parent Name"], "Other Child": ["Other Parent"]}
    dataset = {"name": "tiny", "withheld_children": ["First Child"], "evaluation_ids": ["q0"],
               "queries": [{"id": "q0", "parent": "Parent Name", "child": "First Child",
                            "candidates": ["First Child", "Other Child"], "label": "First Child"}]}
    run = tmp_path / "run"
    metrics = ddv.train_student(dataset, inventory, tiny_config, run, seed=0, device="cpu")
    for name in ("dataset.json", "run.json", "predictions.json", "metrics.json", "adapter", "tokenizer"):
        assert (run / name).exists()
    assert metrics["n"] == 1 and metrics["percent"]["label_accuracy"] == 100  # the label is a true child
    model, tokenizer, mask_id, dtype = ddv.load_run(run, "cpu")
    again = ddv.generate_predictions(model, tokenizer, mask_id, dtype, dataset, 4, "cpu")
    assert again == ddv.read_json(run / "predictions.json")
    with pytest.raises(ValueError):
        ddv.train_student(dataset, inventory, tiny_config, run, seed=0, device="cpu")
