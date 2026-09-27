"""Table 14 and the corpus parent rows of Table 24. Known-direction gains, unscreened cohort.

Four-model means on the main template, with the known direction on mean tokens
throughout. Table 14 prints the gain in points and the wins and losses. The
paper's bootstrap intervals are not recomputed, and the alternate-template rows
need score files that are not bundled here. Table 24 compares the C->P sentence
(the known direction) with the P->C sentence under summed DC on the 512 corpus
parent queries, with exact McNemar p values. Its child-query and notable-parent
rows need score files that are not bundled here.
"""

import ddv
from common import Pool, Report, f2, parser

ROWS = [("reverse_mean", "P->C, mean token", "uniform64"), ("dc_mean", "DC, mean token", "uniform64"),
        ("reverse_mean", "P->C, mean token", "lexical64"), ("dc_mean", "DC, mean token", "lexical64"),
        ("reverse_sum", "P->C, summed token", "uniform64"), ("dc_sum", "DC, summed token", "uniform64"),
        ("reverse_sum", "P->C, summed token", "lexical64"), ("dc_sum", "DC, summed token", "lexical64")]
NAMES = {"uniform64": "Uniform", "lexical64": "Lexical"}

PAPER = {
    "P->C, mean token  Uniform": ["55.66", "285/0"],
    "DC, mean token  Uniform": ["26.56", "145/9"],
    "P->C, mean token  Lexical": ["41.21", "214/3"],
    "DC, mean token  Lexical": ["33.98", "193/19"],
    "P->C, summed token  Uniform": ["27.73", "145/3"],
    "DC, summed token  Uniform": ["14.45", "82/8"],
    "P->C, summed token  Lexical": ["24.41", "142/17"],
    "DC, summed token  Lexical": ["15.43", "101/22"],
}
PAPER24 = {
    "Corpus, 512  Parent  Uniform": ["87.89", "73.44", "82/8", "1.4e-16"],
    "Corpus, 512  Parent  Lexical": ["71.29", "55.86", "101/22", "2.7e-13"],
}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pools = {name: Pool(name, args.data) for name in NAMES}
    report, rows = Report(), {}
    for form, comparator, name in ROWS:
        pool = pools[name]
        wins, losses = ddv.paired_counts(pool.correct("known_mean"), pool.correct(form))
        gain = pool.accuracy("known_mean") - pool.accuracy(form)
        rows[f"{comparator}  {NAMES[name]}"] = [f2(gain), f"{wins}/{losses}"]
    report.table("Table 14. Known-direction gain in points over each comparator",
                 ("Gain", "Wins/losses"), rows, PAPER)

    rows24 = {}
    for name, pool in pools.items():
        wins, losses = ddv.paired_counts(pool.correct("known_mean"), pool.correct("dc_sum"))
        rows24[f"Corpus, 512  Parent  {NAMES[name]}"] = [
            f2(pool.accuracy("known_mean")), f2(pool.accuracy("dc_sum")), f"{wins}/{losses}",
            f"{ddv.mcnemar_p(wins, losses):.1e}"]
    report.table("Table 24. Sentence scoring on corpus facts, parent queries (%)",
                 ("C->P", "P->C", "W/L", "p"), rows24, PAPER24)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
