"""Tables 27a and 29a. Labels and students on surname-mismatched trained queries.

The subsets keep evaluated queries whose parent and recorded child have
different last names, 294 on the acquired pools and 228 on each unscreened
list. DC is the summed-token DC reverse label. Student entries score the
stored outputs of the three runs (three-seed mean and sample SD). The Llama
two-template and eight-channel rows of Table 29a need score files that are
not bundled here.
"""

import ddv
from common import Pool, Report, f2, parser, pct, pm, surname_mismatch

POOLS = [("screened", "Acquired, 294"), ("uniform64", "Uniform, 228"), ("lexical64", "Lexical, 228")]
PAPER27 = {
    "Acquired, 294": ["71.43", "44.22", "53.17 ± 3.76", "30.95 ± 1.48"],
    "Uniform, 228": ["73.25", "44.30", "70.32 ± 0.91", "41.81 ± 0.51"],
    "Lexical, 228": ["47.81", "25.44", "46.05 ± 0.44", "23.83 ± 0.67"],
}
PAPER29 = {
    "Known mean": ["53.17 ± 3.76", "68.82 ± 0.39"],
    "Reverse mean": ["33.22 ± 1.09", "39.68 ± 1.09"],
    "DC reverse mean": ["21.20 ± 0.20", "24.04 ± 0.52"],
}


def subset_open(runs, keep, key="open_accuracy"):
    return pm([pct([flags[j][key] for j in keep]) for flags in runs])


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    report, rows27, rows29 = Report(), {}, {}
    for name, label in POOLS:
        pool = Pool(name, args.data)
        matcher = ddv.InventoryMatcher(pool.inventory)
        subset = [i for i in pool.evaluated if surname_mismatch(pool.queries[i])]
        keep = [j for j, i in enumerate(pool.evaluated) if surname_mismatch(pool.queries[i])]
        known, dc = pool.student_flags("known_mean", matcher), pool.student_flags("dc_sum", matcher)
        rows27[label] = [f2(pool.accuracy("known_mean", subset=subset)),
                         f2(pool.accuracy("dc_sum", subset=subset)),
                         subset_open(known, keep), subset_open(dc, keep)]
        if name == "screened":
            for form, row in (("known_mean", "Known mean"), ("reverse_mean", "Reverse mean"),
                              ("dc_mean", "DC reverse mean")):
                runs = known if form == "known_mean" else pool.student_flags(form, matcher)
                rows29[row] = [subset_open(runs, keep), subset_open(runs, keep, "inventory_accuracy")]
    report.table("Table 27a. Surname-mismatched queries, label and student open accuracy (%)",
                 ("Known label", "DC label", "Known students", "DC students"), rows27, PAPER27)
    report.table("Table 29a. Primary MDM-0.6B students on the 294 acquired queries (%)",
                 ("Open", "Matched"), rows29, PAPER29)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
