"""Table 18 and Figure 4a, the teacher-scored rows. Label accuracy on all three cohorts.

Acquired pools use the 1,390 exposure-free queries and the unscreened lists
all 512 queries. The known direction uses mean tokens, and the two reverse
rows sum tokens, four-teacher means throughout. The tuned-reverse, teacherless
and generation rows need outputs that are not bundled here.
"""

from common import Pool, Report, f2, parser

POOLS = [("screened", "Acquired, 1,390"), ("uniform64", "Uniform, 512"), ("lexical64", "Lexical, 512")]
ROWS = [("known_mean", "Known direction"), ("dc_sum", "DC reverse, summed"),
        ("reverse_sum", "Reverse, summed")]
PAPER = {
    "Known direction": ["90.72", "87.89", "71.29"],
    "DC reverse, summed": ["75.18", "73.44", "55.86"],
    "Reverse, summed": ["71.80", "60.16", "46.88"],
}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    pools = [Pool(name, args.data) for name, _ in POOLS]
    rows = {label: [f2(p.accuracy(form, subset=p.evaluated)) for p in pools] for form, label in ROWS}
    report = Report()
    report.table("Table 18. Label accuracy (%) of the teacher-scored label sources",
                 [label for _, label in POOLS], rows, PAPER)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
