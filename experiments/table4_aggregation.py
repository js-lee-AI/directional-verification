"""Table 4 and Figure 2a. Scoring direction under six aggregation rules, acquired pools.

Four teachers, main template, all 1,500 trained queries. Known uses mean-token
scores, Mean and Summed are the reverse scores, and DC, sum subtracts the summed
context-only score. The last columns compare known with summed DC, with exact
McNemar p values Holm-corrected over the six rules. Figure 2a plots the Known,
Summed and DC, sum columns. The paper's bootstrap intervals are not recomputed.
"""

import ddv
from common import Pool, Report, f2, parser

RULES = [("mean", "Raw-score mean"), ("zscore", "z-score mean"), ("probability", "Probability mean"),
         ("borda", "Borda"), ("rrf", "RRF"), ("agreement", "Soft agreement")]
FORMS = ("known_mean", "reverse_mean", "reverse_sum", "dc_sum")
COLUMNS = ("Known", "Mean", "Summed", "DC, sum", "Points", "W/L", "Holm p")

PAPER = {
    "Raw-score mean": ["90.73", "54.60", "72.40", "75.60", "15.13", "259/32", "1.0e-44"],
    "z-score mean": ["90.27", "54.27", "72.40", "76.20", "14.07", "248/37", "4.8e-39"],
    "Probability mean": ["90.87", "54.27", "73.00", "74.13", "16.73", "280/29", "6.0e-52"],
    "Borda": ["88.93", "55.20", "72.07", "76.47", "12.47", "226/39", "3.4e-33"],
    "RRF": ["89.13", "54.53", "71.60", "76.00", "13.13", "236/39", "3.1e-35"],
    "Soft agreement": ["90.87", "54.33", "73.27", "75.00", "15.87", "266/28", "4.0e-49"],
}
# Appendix B prose: known minus summed DC across the six rules, and the span of each column.
PAPER_SPANS = {
    "1,390 exposure-free": ["12.95", "16.83", "1.87", "2.01"],
    "1,500 trained": ["12.47", "16.73", "1.93", "2.33"],  # the gap range is from Section 4.2
}
# Section 3 and Appendix A counts for the acquired pools.
PAPER_POOL = {"acquired pools": ["1500", "28463", "97.13", "86"]}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pool = Pool("screened", args.data)
    report, rows, flags = Report(), {}, {}
    for rule, name in RULES:
        acc = [pool.accuracy(form, rule) for form in FORMS]
        known, dc = pool.correct("known_mean", rule), pool.correct("dc_sum", rule)
        wins, losses = ddv.paired_counts(known, dc)
        flags[name] = (wins, losses)
        rows[name] = [f2(a) for a in acc] + [f2(acc[0] - acc[3]), f"{wins}/{losses}", None]
    adjusted = ddv.holm([ddv.mcnemar_p(*flags[name]) for _, name in RULES])
    for (_, name), p in zip(RULES, adjusted):
        rows[name][-1] = f"{p:.1e}"
    report.table("Table 4. Label accuracy (%) on the 1,500 acquired pools", COLUMNS, rows, PAPER)

    spans = {}
    for subset, label in ((pool.evaluated, "1,390 exposure-free"), (None, "1,500 trained")):
        known = [pool.accuracy("known_mean", rule, subset=subset) for rule, _ in RULES]
        dc = [pool.accuracy("dc_sum", rule, subset=subset) for rule, _ in RULES]
        gaps = [k - d for k, d in zip(known, dc)]
        spans[label] = [f2(min(gaps)), f2(max(gaps)), f2(max(known) - min(known)), f2(max(dc) - min(dc))]
    report.table("Appendix B. Known minus summed DC across rules, and span of each column (points)",
                 ("gap min", "gap max", "Known span", "DC span"), spans, PAPER_SPANS)

    queries, gold = pool.queries, ddv.parent_children(pool.inventory)
    covered = sum(bool(set(q["candidates"]) & gold[q["parent"]]) for q in queries)
    counts = {"acquired pools": [str(len(queries)), str(sum(len(q["candidates"]) for q in queries)),
                                 f2(100 * covered / len(queries)),
                                 str(sum(q["parent"] in q["candidates"] for q in queries))]}
    report.table("Section 3 and Appendix A. Pool counts",
                 ("queries", "slots", "has true child %", "self slots"), counts, PAPER_POOL)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
