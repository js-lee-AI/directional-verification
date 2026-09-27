"""Tables 11 and 12. Selected-label accuracy and how students reproduce their labels.

Inventory identity I requires the inventory-matched answer to equal the
selected label. Open inclusion O credits the label's full name inside the
answer. F counts wrong labels with correct answers and L correct labels with
wrong answers, under matched (m) and open (o) scoring. Stored outputs of the
three runs on the acquired pools, A, B, C being seeds 0, 1, 2.
"""

import ddv
from common import RUNS, Pool, Report, f2, parser, pct, pm

SOURCES = [("known_mean", "Known mean"), ("reverse_mean", "Reverse mean"), ("dc_sum", "DC reverse sum")]
PAPER11 = {
    "n = 1,390  Known mean": ["90.72", "97.70 ± 0.61", "84.82 ± 2.97"],
    "n = 1,390  Reverse mean": ["53.60", "97.67 ± 0.30", "80.12 ± 1.63"],
    "n = 1,390  DC reverse sum": ["75.18", "97.39 ± 0.51", "80.24 ± 1.30"],
    "n = 666  Known mean": ["82.73", "97.25 ± 0.76", "84.33 ± 2.23"],
    "n = 666  Reverse mean": ["5.26", "96.60 ± 0.35", "72.12 ± 1.00"],
}
PAPER12 = {
    "Known mean A": ["1366", "1215", "0", "20", "1", "146"],
    "Known mean B": ["1359", "1188", "1", "23", "1", "164"],
    "Known mean C": ["1349", "1134", "1", "33", "0", "221"],
    "Reverse mean A": ["1361", "1138", "5", "5", "0", "69"],
    "Reverse mean B": ["1359", "1110", "4", "12", "0", "96"],
    "Reverse mean C": ["1353", "1093", "3", "14", "0", "103"],
    "DC reverse sum A": ["1353", "1098", "3", "22", "1", "168"],
    "DC reverse sum B": ["1347", "1134", "5", "23", "1", "148"],
    "DC reverse sum C": ["1361", "1114", "4", "17", "0", "183"],
}
# Appendix D prose: correct labels N+ and wrong labels turned into whole-answer matches.
PAPER_N = {"Known mean": ["1261", "0"], "Reverse mean": ["745", "0"], "DC reverse sum": ["1045", "0"]}


def fixed_lost(flags, key):
    fixed = sum(not r["label_accuracy"] and r[key] for r in flags)
    lost = sum(r["label_accuracy"] and not r[key] for r in flags)
    return fixed, lost


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pool = Pool("screened", args.data)
    matcher = ddv.InventoryMatcher(pool.inventory)
    runs = {form: pool.student_flags(form, matcher) for form, _ in SOURCES}
    known, reverse = pool.labels("known_mean"), pool.labels("reverse_mean")
    disagree = {pool.queries[i]["id"] for i in pool.evaluated if known[i] != reverse[i]}

    report, rows11 = Report(), {}
    for subset, tag, sources in ((None, "n = 1,390", SOURCES), (disagree, f"n = {len(disagree)}", SOURCES[:2])):
        for form, name in sources:
            kept = [[r for r in flags if subset is None or r["id"] in subset] for flags in runs[form]]
            rows11[f"{tag}  {name}"] = [f2(pct([r["label_accuracy"] for r in kept[0]])),
                                        pm([pct([r["inventory_selected_label"] for r in f]) for f in kept]),
                                        pm([pct([r["open_selected_label"] for r in f]) for f in kept])]
    report.table("Table 11. Selected-label accuracy, inventory identity I and open inclusion O (%)",
                 ("Label accuracy", "I", "O"), rows11, PAPER11)

    rows12, totals = {}, {}
    for form, name in SOURCES:
        whole_fixed = 0
        for run, flags in zip(RUNS, runs[form]):
            fm, lm = fixed_lost(flags, "inventory_accuracy")
            fo, lo = fixed_lost(flags, "open_accuracy")
            whole_fixed += fixed_lost(flags, "whole_accuracy")[0]
            rows12[f"{name} {run}"] = [str(sum(r["inventory_selected_label"] for r in flags)),
                                       str(sum(r["open_selected_label"] for r in flags)),
                                       str(fm), str(lm), str(fo), str(lo)]
        totals[name] = [str(sum(r["label_accuracy"] for r in runs[form][0])), str(whole_fixed)]
    report.table("Table 12. Selected-label transfer counts by run", ("I", "O", "Fm", "Lm", "Fo", "Lo"),
                 rows12, PAPER12)
    report.table("Appendix D. Correct labels N+, and wrong labels turned into whole-answer matches "
                 "over the three runs", ("N+", "whole fixed"), totals, PAPER_N)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
