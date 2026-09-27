from pathlib import Path

import numpy as np
import pytest

import ddv
from ddv.data import candidate_fingerprint

DATA = Path(__file__).resolve().parents[1] / "data"


def test_withholding_all_edges_of_a_child():
    corpus = {"First Child": ["First Parent", "Second Parent"],
              "Sibling Child": ["First Parent"], "Other Child": ["Other Parent"]}
    assert len(ddv.forward_pairs(corpus, ["First Child"])) == 2
    dataset = {"queries": [{"id": "a", "parent": "First Parent",
                            "candidates": ["First Child"], "label": "First Child"},
                           {"id": "b", "parent": "Second Parent",
                            "candidates": ["First Child"], "label": "First Child"}],
               "withheld_children": ["First Child"], "evaluation_ids": ["b"]}
    assert ddv.validate_dataset(dataset, corpus)["evaluated_queries"] == 1
    dataset["evaluation_ids"].append("a")
    with pytest.raises(ValueError):
        ddv.validate_dataset(dataset, corpus)


def test_a_label_equal_to_the_parent_is_rejected():
    corpus = {"A": ["P"], "P": ["Q"]}
    dataset = {"queries": [{"id": "a", "parent": "P", "candidates": ["A", "P"], "label": "P"}],
               "withheld_children": [], "evaluation_ids": []}
    with pytest.raises(ValueError):
        ddv.validate_dataset(dataset, corpus)


def test_gzip_json_is_byte_stable(tmp_path):
    value = {"b": [1.5, -2.0], "a": "José"}
    ddv.write_json(tmp_path / "x.json.gz", value)
    first = (tmp_path / "x.json.gz").read_bytes()
    ddv.write_json(tmp_path / "x.json.gz", value)
    assert (tmp_path / "x.json.gz").read_bytes() == first
    assert ddv.read_json(tmp_path / "x.json.gz") == value


def _tiny_root(tmp_path):
    queries = [{"id": "q", "parent": "P", "candidates": ["A", "B"], "label": "A"}]
    ddv.write_json(tmp_path / "tiny.json", {"queries": queries})
    signature = candidate_fingerprint(queries)
    for teacher in ddv.TEACHERS:
        ddv.write_json(tmp_path / "scores" / f"tiny_{teacher}.json.gz",
                       {"teacher": teacher, "candidate_fingerprint": signature, "scores": [[-1.0, -2.0]]})
        ddv.write_json(tmp_path / "scores" / f"tiny_{teacher}_reverse.json.gz",
                       {"teacher": teacher, "candidate_fingerprint": signature,
                        "reverse_mean": [[-3.0, -2.0]], "reverse_tokens": [[2, 4]],
                        "parent_tokens": [[1, 1]], "context_mean": {"A": -4.0, "B": -1.0},
                        "context_tokens": {"A": 2, "B": 4}})
    return tmp_path


def test_score_forms_follow_eq3(tmp_path):
    root = _tiny_root(tmp_path)
    forms = ddv.load_scores("tiny", root, forms=ddv.FORMS)
    first = {form: forms[form][0][0].tolist() for form in ddv.FORMS}
    assert first["known_mean"] == [-1.0, -2.0]
    assert first["reverse_sum"] == [-6.0, -8.0]
    assert first["dc_mean"] == [1.0, -1.0]
    assert first["dc_sum"] == [2.0, -4.0]
    assert ddv.context_sums("tiny", root)["B"] == [-4.0] * 4


def test_scores_for_other_candidates_are_refused(tmp_path):
    root = _tiny_root(tmp_path)
    pool = ddv.read_json(root / "tiny.json")
    pool["queries"][0]["candidates"] = ["B", "A"]
    ddv.write_json(root / "tiny.json", pool)
    with pytest.raises(ValueError):
        ddv.load_scores("tiny", root)


@pytest.mark.parametrize("pool,queries,slots", [("screened", 1500, 28463),
                                                ("uniform64", 512, 32768),
                                                ("lexical64", 512, 32768)])
def test_bundled_pools(pool, queries, slots):
    inventory = ddv.load_inventory(DATA)
    data = ddv.load_pool(pool, DATA)
    dims = ddv.validate_dataset(data, inventory)
    assert (dims["queries"], dims["candidate_slots"]) == (queries, slots)
    channels = ddv.load_scores(pool, DATA)["known_mean"]
    assert ddv.select_labels(data["queries"], channels) == [r["label"] for r in data["queries"]]


def test_bundled_forms_are_consistent():
    forms = ddv.load_scores("screened", DATA, teachers=("qwen",), forms=ddv.FORMS)
    context = ddv.read_json(DATA / "scores" / "screened_qwen_reverse.json.gz")
    queries = ddv.load_pool("screened", DATA)["queries"]
    for i in (0, 700, 1499):
        names = queries[i]["candidates"]
        expected = forms["reverse_mean"][0][i] - np.array([context["context_mean"][c] for c in names])
        np.testing.assert_allclose(forms["dc_mean"][0][i], expected)
        assert np.isfinite(forms["dc_sum"][0][i]).all()


def test_quickstart_numbers_are_the_stored_scores():
    pool = ddv.load_pool("screened", DATA)
    i = next(k for k, r in enumerate(pool["queries"]) if r["parent"] == "Susana Dosamantes")
    names = pool["queries"][i]["candidates"]
    pick = [names.index(n) for n in ("Paulina Rubio", "Diego Luna", "Odiseo Bichir", "Selena Gomez")]
    forms = ddv.load_scores("screened", DATA, forms=("known_mean", "reverse_sum", "dc_sum"))
    mean = {form: np.mean([channel[i] for channel in forms[form]], axis=0)[pick] for form in forms}
    np.testing.assert_allclose(mean["known_mean"], [-2.46, -2.93, -2.87, -4.23], atol=0.005)
    np.testing.assert_allclose(mean["reverse_sum"], [-15.61, -13.60, -23.71, -14.30], atol=0.005)
    context = mean["reverse_sum"] - mean["dc_sum"]
    np.testing.assert_allclose(context, [-24.78, -21.12, -35.68, -18.51], atol=0.005)
    # Over all 17 candidates the picks are those of Section 4.2.
    full = {form: np.mean([channel[i] for channel in forms[form]], axis=0) for form in forms}
    assert names[int(np.argmax(full["known_mean"]))] == pool["queries"][i]["child"] == "Paulina Rubio"
    assert names[int(np.argmax(full["reverse_sum"]))] == "Diego Luna"
    assert names[int(np.argmax(full["dc_sum"]))] == "Odiseo Bichir"
    assert names[int(np.argmax(full["reverse_sum"] - full["dc_sum"]))] == "Selena Gomez"
