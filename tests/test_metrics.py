import pytest

import ddv


def test_metric_boundaries():
    assert ddv.open_match("The child is Jose Garcia.", ["Jose Garcia"])
    assert not ddv.whole_match("The child is Jose Garcia.", ["Jose Garcia"])
    assert ddv.whole_match("  JOSE   GARCIA. ", ["Jose Garcia"])
    assert ddv.whole_match("José García", ["Jose Garcia"])
    assert not ddv.open_match("Garcia", ["Jose Garcia"])
    assert not ddv.whole_match("", ["Jose Garcia"])
    matcher = ddv.InventoryMatcher(["Jose Garcia", "Jane Jones"])
    assert matcher.match("") == ""
    assert matcher.match("Jose Gacia") == "Jose Garcia"
    assert matcher.match("Jnae Jnoes") == "Jane Jones"  # no shared token, full scan


def test_prediction_denominator_and_any_child():
    dataset = {"queries": [{"id": "a", "parent": "Parent Name", "label": "First Child"}],
               "evaluation_ids": ["a"]}
    corpus = {"First Child": ["Parent Name"], "Second Child": ["Parent Name"]}
    result = ddv.evaluate_predictions(dataset, corpus, [{"id": "a", "text": "Second Child"}])
    assert result["percent"]["whole_accuracy"] == 100
    assert result["percent"]["whole_selected_label"] == 0
    with pytest.raises(ValueError):
        ddv.evaluate_predictions(dataset, corpus, [])
    with pytest.raises(ValueError):
        ddv.evaluate_predictions(dataset, corpus, [{"id": "a", "text": ""}] * 2)


def test_flags_add_up_to_the_counts():
    corpus = {"A": ["P"], "B": ["P"], "C": ["Q"]}
    dataset = {"queries": [{"id": "1", "parent": "P", "label": "A"},
                           {"id": "2", "parent": "Q", "label": "A"},
                           {"id": "3", "parent": "P", "label": "B"}],
               "evaluation_ids": ["1", "2"]}
    predictions = [{"id": "1", "text": "The child is A"}, {"id": "2", "text": "C"}]
    flags = ddv.answer_flags(dataset, corpus, predictions)
    assert [f["id"] for f in flags] == ["1", "2"]
    result = ddv.evaluate_predictions(dataset, corpus, predictions)
    assert result["n"] == 2
    for key, count in result["counts"].items():
        assert sum(f[key] for f in flags) == count
    assert result["counts"]["label_accuracy"] == 1
    assert result["counts"]["open_selected_label"] == 1


def test_label_accuracy():
    corpus = {"A": ["P"], "B": ["Q"]}
    queries = [{"parent": "P"}, {"parent": "P"}, {"parent": "Q"}, {"parent": "Q"}]
    assert ddv.label_correct(queries, ["A", "B", "B", "A"], corpus) == [True, False, True, False]
    assert ddv.label_accuracy(queries, ["A", "B", "B", "B"], corpus) == 75
    with pytest.raises(ValueError):
        ddv.label_accuracy(queries, ["A"], corpus)


def test_mcnemar_and_holm_by_hand():
    assert ddv.paired_counts([1, 1, 0, 0, 1], [0, 1, 1, 0, 0]) == (2, 1)
    assert ddv.mcnemar_p(5, 0) == pytest.approx(2 / 32)
    assert ddv.mcnemar_p(3, 3) == 1.0
    assert ddv.mcnemar_p(0, 0) == 1.0
    assert ddv.mcnemar_p(1, 9) == pytest.approx(2 * 11 / 1024)
    assert ddv.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ddv.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
