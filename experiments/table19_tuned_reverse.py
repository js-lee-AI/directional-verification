"""Table 19, the background rows. Tuned reverse label accuracy with each cohort's own best lambda.

Background scores subtract lambda times the context-only score after "The
child is" (eq. 3), from summed or mean-token reverse scores, and average the
four teachers. Lambda runs over 0 to 1.5 in steps of 0.05, and ties prefer the
value closest to one, then the smaller value (Appendix F). The acquired pools
use the 1,390 exposure-free queries. The transferred acquired comparator is the
summed background score at 0.75, chosen on the other cohorts. The Monte Carlo
rows, and the transferred comparators of the unscreened lists, need score files
that are not bundled here.
"""

import numpy as np

import ddv
from common import Pool, Report, f2, parser, pct

POOLS = [("screened", "Acquired, 1,390"), ("uniform64", "Uniform, 512"), ("lexical64", "Lexical, 512")]
GRID = np.round(np.arange(0, 1.5001, 0.05), 2)
PAPER = {
    "Background, summed": ["81.58 (0.60)", "76.37 (0.80)", "58.79 (0.75)"],
    "Background, mean": ["80.94 (0.45)", "74.80 (0.70)", "56.84 (0.55)"],
    "Transferred": ["80.65 (0.75)", None, None],
    "Known direction": ["90.72", "87.89", "71.29"],
    "Known - transferred": ["10.07", None, None],
    "Wins/losses": ["170/30", None, None],
}


def parts(pool, form):
    """Per-teacher reverse and context-only scores, summed or mean-token."""
    reverse = ddv.load_scores(pool.name, pool.root, forms=(form,))[form]
    context = []
    for teacher in ddv.TEACHERS:
        c = ddv.read_json(pool.root / "scores" / f"{pool.name}_{teacher}_reverse.json.gz")
        names = [row["candidates"] for row in pool.queries]
        if form == "reverse_sum":
            context.append([ddv.summed([c["context_mean"][x] for x in n], [c["context_tokens"][x] for x in n])
                            for n in names])
        else:
            context.append([[c["context_mean"][x] for x in n] for n in names])
    return reverse, context


def correct(pool, reverse, context, lam):
    channels = [[ddv.domain_context(r, c, lam) for r, c in zip(rt, ct)] for rt, ct in zip(reverse, context)]
    return ddv.label_correct(pool.queries, ddv.select_labels(pool.queries, channels), pool.inventory)


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    rows = {name: [] for name in PAPER}
    for name, _ in POOLS:
        pool = Pool(name, args.data)
        subset = pool.evaluated
        flags = {}
        for form, row in (("reverse_sum", "Background, summed"), ("reverse_mean", "Background, mean")):
            reverse, context = parts(pool, form)
            flags[form] = {lam: correct(pool, reverse, context, lam) for lam in GRID}
            accuracy = {lam: pct([flags[form][lam][i] for i in subset]) for lam in GRID}
            best = max(accuracy.values())
            lam = min((lam for lam in GRID if accuracy[lam] == best), key=lambda x: (abs(x - 1), x))
            rows[row].append(f"{f2(best)} ({lam:.2f})")
        known = pool.accuracy("known_mean", subset=subset)
        rows["Known direction"].append(f2(known))
        if name == "screened":
            chosen = flags["reverse_sum"][0.75]
            transferred = pct([chosen[i] for i in subset])
            wins, losses = ddv.paired_counts([pool.correct("known_mean")[i] for i in subset],
                                             [chosen[i] for i in subset])
            rows["Transferred"].append(f"{f2(transferred)} (0.75)")
            rows["Known - transferred"].append(f2(known - transferred))
            rows["Wins/losses"].append(f"{wins}/{losses}")
        else:
            for row in ("Transferred", "Known - transferred", "Wins/losses"):
                rows[row].append("-")
    report = Report()
    report.table("Table 19. Tuned reverse label accuracy (%) with the best lambda in parentheses",
                 [label for _, label in POOLS], rows, PAPER)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
