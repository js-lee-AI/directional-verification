"""Tables 9 and 10. Primary student results by training run, acquired pools.

Scores the stored outputs of the three runs (A, B, C are seeds 0, 1, 2) on the
1,390 exposure-free queries. Table 9 gives open and matched percentages and
exact counts of whole-answer correctness and label equality. Table 10 compares
known-direction with summed-DC students.
"""

import statistics

import ddv
from common import RUNS, Pool, Report, f2, parser, pct, pm

TABLE9 = [("known_mean", "Known mean"), ("reverse_mean", "Reverse mean"), ("dc_mean", "DC reverse mean")]
PAPER9 = {
    "Known mean A": ["80.29", "1112", "89.28", "1211"],
    "Known mean B": ["78.99", "1095", "89.14", "1186"],
    "Known mean C": ["74.82", "1035", "88.42", "1129"],
    "Reverse mean A": ["48.63", "673", "53.60", "1132"],
    "Reverse mean B": ["46.69", "648", "53.02", "1103"],
    "Reverse mean C": ["46.19", "639", "52.81", "1084"],
    "DC reverse mean A": ["49.21", "683", "53.53", "1256"],
    "DC reverse mean B": ["49.35", "682", "53.81", "1252"],
    "DC reverse mean C": ["49.21", "681", "54.46", "1260"],
}
PAPER10 = {
    "Known-direction mean A": ["80.29", "80.00", "89.28"],
    "Known-direction mean B": ["78.99", "78.78", "89.14"],
    "Known-direction mean C": ["74.82", "74.46", "88.42"],
    "Known-direction mean Mean": ["78.03 ± 2.86", "77.75 ± 2.91", "88.94 ± 0.46"],
    "DC reverse sum A": ["63.17", "63.09", "73.81"],
    "DC reverse sum B": ["64.60", "64.53", "73.88"],
    "DC reverse sum C": ["62.01", "61.80", "74.24"],
    "DC reverse sum Mean": ["63.26 ± 1.30", "63.14 ± 1.37", "73.98 ± 0.23"],
    "Known minus DC Paired": ["14.77 ± 2.18", "14.60 ± 2.15", "14.96 ± 0.69"],
}
# Appendix C prose: correct labels out of 1,390.
PAPER_LABELS = {"Known-direction mean": ["1,261", "90.72"], "DC reverse sum": ["1,045", "75.18"]}
KEYS10 = ("open_accuracy", "whole_accuracy", "inventory_accuracy")


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pool = Pool("screened", args.data)
    matcher = ddv.InventoryMatcher(pool.inventory)
    report, rows9 = Report(), {}
    for form, name in TABLE9:
        for run, flags in zip(RUNS, pool.student_flags(form, matcher)):
            rows9[f"{name} {run}"] = [f2(pct([r["open_accuracy"] for r in flags])),
                                      str(sum(r["whole_accuracy"] for r in flags)),
                                      f2(pct([r["inventory_accuracy"] for r in flags])),
                                      str(sum(r["whole_selected_label"] for r in flags))]
    report.table("Table 9. Primary student results by run (percent, or counts out of 1,390)",
                 ("Open (%)", "Whole count", "Matched (%)", "Label count"), rows9, PAPER9)

    rows10, values, labels = {}, {}, {}
    for form, name in (("known_mean", "Known-direction mean"), ("dc_sum", "DC reverse sum")):
        runs = pool.student_flags(form, matcher)
        values[form] = [[pct([r[key] for r in flags]) for flags in runs] for key in KEYS10]
        for k, run in enumerate(RUNS):
            rows10[f"{name} {run}"] = [f2(v[k]) for v in values[form]]
        rows10[f"{name} Mean"] = [pm(v) for v in values[form]]
        correct = sum(runs[0][i]["label_accuracy"] for i in range(len(runs[0])))
        labels[name] = [f"{correct:,}", f2(100 * correct / len(runs[0]))]
    rows10["Known minus DC Paired"] = [
        pm([a - b for a, b in zip(k, d)]) for k, d in zip(values["known_mean"], values["dc_sum"])]
    report.table("Table 10. Summed-token DC reverse control on the 1,390 exposure-free queries (%)",
                 ("Open", "Whole answer", "Matched"), rows10, PAPER10)
    report.table("Appendix C. Correct labels out of 1,390", ("correct", "%"), labels, PAPER_LABELS)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
