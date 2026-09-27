"""Table 15. Accuracy on the 228 surname-mismatched queries of the unscreened cohort.

Queries whose parent and recorded child have different last names, with the
candidate lists unchanged. Per-teacher rows use the main template and mean
tokens. The known direction (C->P) uses mean tokens in every row. The
alternate-template summed row needs score files that are not bundled here.
"""

import ddv
from common import Pool, Report, f2, parser, surname_mismatch

POOLS = [("uniform64", "Uniform"), ("lexical64", "Lexical")]
SCORERS = [(("qwen",), "Qwen3-8B"), (("olmo",), "OLMo-2-7B"), (("mistral",), "Mistral-7B"),
           (("llama",), "Llama-3.1-8B"), (ddv.TEACHERS, "Four-model mean")]
COLUMNS = ("Uniform C->P", "Uniform P->C", "Uniform DC", "Lexical C->P", "Lexical P->C", "Lexical DC")

PAPER = {
    "Qwen3-8B": ["41.67", "7.02", "17.11", "24.56", "6.14", "9.65"],
    "OLMo-2-7B": ["58.33", "15.35", "22.81", "35.96", "12.28", "8.33"],
    "Mistral-7B": ["65.79", "16.23", "35.09", "42.54", "12.28", "17.98"],
    "Llama-3.1-8B": ["75.44", "15.79", "23.25", "58.33", "14.47", "12.72"],
    "Four-model mean": ["73.25", "15.35", "32.02", "47.81", "12.28", "13.60"],
    "Summed, main template": ["73.25", "29.39", "44.30", "47.81", "18.42", "25.44"],
}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    rows = {name: [] for _, name in SCORERS}
    rows["Summed, main template"] = []
    sizes = []
    for name, _ in POOLS:
        pool = Pool(name, args.data)
        subset = [i for i, q in enumerate(pool.queries) if surname_mismatch(q)]
        sizes.append(len(subset))
        for teachers, scorer in SCORERS:
            rows[scorer] += [f2(pool.accuracy(form, teachers=teachers, subset=subset))
                             for form in ("known_mean", "reverse_mean", "dc_mean")]
        rows["Summed, main template"] += [f2(pool.accuracy(form, subset=subset))
                                          for form in ("known_mean", "reverse_sum", "dc_sum")]
    report = Report()
    report.table(f"Table 15. Accuracy (%) on {' and '.join(map(str, sizes))} surname-mismatched queries",
                 COLUMNS, rows, PAPER)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
