"""Table 5. Mean-token domain-context correction on the acquired pools.

Known, reverse and DC reverse labels from mean-token scores, on all 1,500
trained queries and on the 305 whose parent and recorded child have different
last names. The second table checks the win and loss counts of Appendix B for
known against mean-token DC under the raw-score mean.
"""

import ddv
from common import Pool, Report, f2, parser, surname_mismatch

RULES = [("mean", "Raw"), ("zscore", "z-score"), ("probability", "Probability")]
FORMS = ("known_mean", "reverse_mean", "dc_mean")
COLUMNS = ("Known", "Reverse", "DC", "Known 305", "Reverse 305", "DC 305")

PAPER = {
    "Raw": ["90.73", "54.60", "55.60", "71.80", "40.66", "25.25"],
    "z-score": ["90.27", "54.27", "56.40", "70.82", "40.33", "25.90"],
    "Probability": ["90.87", "54.27", "53.87", "72.79", "40.98", "23.61"],
}
PAPER_WL = {"Known vs DC, mean token": ["557", "30"]}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pool = Pool("screened", args.data)
    mismatch = [i for i, q in enumerate(pool.queries) if surname_mismatch(q)]
    report, rows = Report(), {}
    for rule, name in RULES:
        rows[name] = ([f2(pool.accuracy(form, rule)) for form in FORMS]
                      + [f2(pool.accuracy(form, rule, subset=mismatch)) for form in FORMS])
    report.table(f"Table 5. Label accuracy (%), all 1,500 and {len(mismatch)} surname-mismatched",
                 COLUMNS, rows, PAPER)
    wins, losses = ddv.paired_counts(pool.correct("known_mean"), pool.correct("dc_mean"))
    report.table("Appendix B. Paired counts on all 1,500", ("wins", "losses"),
                 {"Known vs DC, mean token": [str(wins), str(losses)]}, PAPER_WL)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
