import copy

import numpy as np
import pytest

import ddv

RULES = ("mean", "zscore", "probability", "borda", "rrf")


def test_selection_ignores_gold_and_self():
    queries = [{"id": "q", "parent": "Parent Name",
                "candidates": ["Parent Name", "First Child", "Other Child"]}]
    channels = [[[100.0, 1.0, 0.0]], [[100.0, 0.0, 1.0]]]
    assert ddv.select_labels(queries, channels) == ["First Child"]
    contaminated = copy.deepcopy(queries)
    contaminated[0]["gold"] = "Other Child"
    assert ddv.select_labels(contaminated, channels) == ["First Child"]
    channels[0][0][0] = float("nan")
    with pytest.raises(ValueError):
        ddv.select_labels(queries, channels)


def test_ties_keep_candidate_order():
    queries = [{"id": "q", "parent": "P", "candidates": ["B", "A", "C"]}]
    for rule in RULES:
        assert ddv.select_labels(queries, [[[0.0, 0.0, -1.0]]], rule) == ["B"]
        assert ddv.select_label([[0.0, 0.0, -1.0]], ["B", "A", "C"], "P", rule) == "B"


@pytest.mark.parametrize("rule", RULES + ("agreement",))
def test_the_queried_parent_is_never_selected(rule):
    queries = [{"id": f"q{i}", "parent": "P", "candidates": ["P", "A", "B"]} for i in range(3)]
    channels = [[[50.0, -1.0, -2.0]] * 3, [[40.0, -2.0, -1.5]] * 3]
    assert "P" not in ddv.select_labels(queries, channels, rule)


def test_zscore_removes_channel_scale():
    # The second channel is ten times larger but ranks the other way round.
    scores = [[1.0, 2.0, 3.0], [30.0, 20.0, 10.0], [0.0, 0.1, 0.3]]
    assert ddv.select_label(scores, ["A", "B", "C"], "P", "mean") == "A"
    assert ddv.select_label(scores, ["A", "B", "C"], "P", "zscore") == "C"


def test_rank_rules_on_a_hand_checked_case():
    scores = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    mask = ddv.self_mask(["A", "B", "C"], "P")
    # Borda gives A 2 + 0, B 1 + 2 and C 0 + 1.
    assert ddv.combine(scores, mask, "borda").tolist() == [2.0, 3.0, 1.0]
    np.testing.assert_allclose(ddv.combine(scores, mask, "rrf"),
                               [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])
    # With a self slot the other two candidates are ranked among themselves.
    with_self = np.array([[9.0, 3.0, 2.0], [9.0, 1.0, 3.0]])
    mask = ddv.self_mask(["P", "A", "B"], "P")
    assert ddv.combine(with_self, mask, "borda")[1:].tolist() == [1.0, 1.0]


def test_probability_masks_self_before_the_softmax():
    scores = np.array([[5.0, 0.0, np.log(3.0)]])
    mask = ddv.self_mask(["P", "A", "B"], "P")
    np.testing.assert_allclose(ddv.combine(scores, mask, "probability")[1:], [0.25, 0.75])


def test_batch_and_single_query_calls_agree():
    rng = np.random.default_rng(0)
    queries, channels = [], [[] for _ in range(4)]
    for i in range(30):
        names = [f"n{j}" for j in range(int(rng.integers(2, 9)))]
        queries.append({"id": str(i), "parent": names[int(rng.integers(len(names)))], "candidates": names})
        for channel in channels:
            channel.append(rng.normal(size=len(names)))
    for rule in RULES:
        batch = ddv.select_labels(queries, channels, rule)
        single = [ddv.select_label([c[i] for c in channels], q["candidates"], q["parent"], rule)
                  for i, q in enumerate(queries)]
        assert batch == single


def test_agreement_downweights_a_channel_that_disagrees():
    rng = np.random.default_rng(1)
    queries, good, noise = [], [], []
    for i in range(40):
        queries.append({"id": str(i), "parent": "P", "candidates": ["A", "B", "C", "D"]})
        good.append(np.array([3.0, 0.0, 0.0, 0.0]))
        noise.append(rng.normal(scale=3.0, size=4))
    distributions = [np.array([ddv.labels._softmax(v) for v in values])
                     for values in zip(good, good, noise)]
    _, weights = ddv.soft_agreement(distributions)
    assert weights[0] == pytest.approx(weights[1]) and weights[2] < weights[0]
    assert set(ddv.select_labels(queries, [good, good, noise], "agreement")) == {"A"}


def test_bad_inputs_raise():
    queries = [{"id": "q", "parent": "P", "candidates": ["A", "B"]}]
    with pytest.raises(ValueError):
        ddv.select_labels(queries, [])
    with pytest.raises(ValueError):
        ddv.select_labels(queries, [[[0.0, 1.0, 2.0]]])
    with pytest.raises(ValueError):
        ddv.select_labels(queries, [[[0.0, 1.0]], [[]]])
    with pytest.raises(ValueError):
        ddv.select_labels([{"id": "q", "parent": "A", "candidates": ["A"]}], [[[0.0]]])
    with pytest.raises(ValueError):
        ddv.combine([[0.0, 1.0]], np.zeros(2), "agreement")
