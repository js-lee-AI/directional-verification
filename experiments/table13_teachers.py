"""Table 13. Per-teacher candidate accuracy on the unscreened cohort, main template.

C->P is the known direction and P->C scores each candidate child after the
parent. DC subtracts the context-only score after "The child is". The
alternate-template rows need score files that are not bundled here.
"""

import ddv
from common import Pool, Report, f2, parser

POOLS = [("uniform64", "Uniform"), ("lexical64", "Lexical")]
FORMS = ("known_mean", "reverse_mean", "dc_mean", "known_sum", "reverse_sum", "dc_sum")
COLUMNS = ("C->P mean", "P->C mean", "DC mean", "C->P sum", "P->C sum", "DC sum")
SCORERS = [(("qwen",), "Qwen3-8B"), (("olmo",), "OLMo-2-7B"), (("mistral",), "Mistral-7B"),
           (("llama",), "Llama-3.1-8B"), (ddv.TEACHERS, "Four-model mean")]

PAPER = {
    "Uniform  Qwen3-8B": ["72.27", "23.44", "47.27", "72.27", "52.15", "57.03"],
    "Uniform  OLMo-2-7B": ["81.05", "32.81", "50.20", "81.05", "55.08", "64.06"],
    "Uniform  Mistral-7B": ["84.38", "31.84", "61.52", "84.38", "61.72", "70.31"],
    "Uniform  Llama-3.1-8B": ["88.48", "26.37", "41.60", "88.48", "55.66", "62.50"],
    "Uniform  Four-model mean": ["87.89", "32.23", "61.33", "87.50", "60.16", "73.44"],
    "Lexical  Qwen3-8B": ["56.45", "24.41", "31.45", "56.45", "39.26", "43.55"],
    "Lexical  OLMo-2-7B": ["63.48", "28.32", "29.10", "63.48", "41.60", "47.85"],
    "Lexical  Mistral-7B": ["65.62", "30.08", "41.60", "65.62", "45.51", "55.66"],
    "Lexical  Llama-3.1-8B": ["75.59", "26.17", "27.54", "75.59", "45.51", "51.76"],
    "Lexical  Four-model mean": ["71.29", "30.08", "37.30", "71.29", "46.88", "55.86"],
}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    report, rows = Report(), {}
    for name, block in POOLS:
        pool = Pool(name, args.data)
        for teachers, scorer in SCORERS:
            rows[f"{block}  {scorer}"] = [f2(pool.accuracy(form, teachers=teachers)) for form in FORMS]
    report.table("Table 13. Candidate accuracy (%) on 512 queries per pool, main template",
                 COLUMNS, rows, PAPER)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
