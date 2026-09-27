"""Helpers shared by the table scripts. Run the scripts from the repository root."""

from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path

import ddv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ("A", "B", "C")  # the paper's names for training seeds 0, 1 and 2


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=str(DATA), help="the bundled data folder")
    p.add_argument("--out", help="also write the printed rows to this JSON file")
    return p


def f2(value):
    return f"{value:.2f}"


def f1(value):
    return f"{value:.1f}"


def pm(values):
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def pct(flags):
    return 100 * sum(flags) / len(flags)


def surname_mismatch(query):
    return query["parent"].lower().split()[-1] != query["child"].lower().split()[-1]


class Pool:
    """A bundled pool with cached label selections and correctness flags."""

    def __init__(self, name, root=DATA):
        self.name, self.root = name, Path(root)
        self.data = ddv.load_pool(name, self.root)
        self.queries = self.data["queries"]
        self.inventory = ddv.load_inventory(self.root)
        evaluated = set(self.data["evaluation_ids"])
        self.evaluated = [i for i, q in enumerate(self.queries) if q["id"] in evaluated]
        self._scores, self._labels = {}, {}

    def scores(self, form):
        if form not in self._scores:
            self._scores[form] = ddv.load_scores(self.name, self.root, forms=(form,))[form]
        return self._scores[form]

    def labels(self, form, rule="mean", teachers=None):
        key = (form, rule, teachers)
        if key not in self._labels:
            channels = self.scores(form)
            if teachers is not None:
                channels = [channels[ddv.TEACHERS.index(t)] for t in teachers]
            self._labels[key] = ddv.select_labels(self.queries, channels, rule)
        return self._labels[key]

    def correct(self, form, rule="mean", teachers=None):
        return ddv.label_correct(self.queries, self.labels(form, rule, teachers), self.inventory)

    def accuracy(self, form, rule="mean", teachers=None, subset=None):
        flags = self.correct(form, rule, teachers)
        subset = range(len(flags)) if subset is None else subset
        return pct([flags[i] for i in subset])

    def relabeled(self, form):
        data = copy.deepcopy(self.data)
        for row, label in zip(data["queries"], self.labels(form)):
            row["label"] = label
        return data

    def student_flags(self, form, matcher=None):
        """Per-query answer flags of the three stored student runs trained on form labels."""
        data = self.relabeled(form)
        matcher = matcher or ddv.InventoryMatcher(self.inventory)
        runs = []
        for seed in range(3):
            stored = ddv.read_json(self.root / "predictions" / f"{self.name}_{form}_seed{seed}.json.gz")
            runs.append(ddv.answer_flags(data, self.inventory, stored["predictions"], matcher))
        return runs


class Report:
    """Prints script values next to the values printed in the paper."""

    def __init__(self):
        self.tables, self.compared, self.mismatches = [], 0, []

    def table(self, title, columns, rows, paper):
        """rows and paper map a row name to its cells, formatted as the paper prints them."""
        width = max(len(name) for name in rows) + 2
        cells = [max([len(c)] + [len(r[k]) for r in rows.values()]) + 2 for k, c in enumerate(columns)]
        print(f"\n{title}")
        print(" " * width + "".join(c.rjust(w) for c, w in zip(columns, cells)))
        for name, row in rows.items():
            expected = paper.get(name)
            marks = []
            for k, got in enumerate(row):
                want = None if expected is None else expected[k]
                if want is None:
                    marks.append(got)
                    continue
                self.compared += 1
                if got != want:
                    self.mismatches.append(f"{title} / {name} / {columns[k]}: script {got}, paper {want}")
                    got += "*"
                marks.append(got)
            print(name.ljust(width) + "".join(m.rjust(w) for m, w in zip(marks, cells)))
        self.tables.append({"title": title, "columns": list(columns), "rows": rows, "paper": paper})

    def finish(self, out=None):
        print(f"\n{self.compared - len(self.mismatches)} of {self.compared} values match the paper.")
        for line in self.mismatches:
            print("  differs (marked *):", line)
        if out:
            ddv.write_json(out, {"tables": self.tables, "mismatches": self.mismatches})
        return 1 if self.mismatches else 0
