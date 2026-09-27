"""Table 17. Unscreened-cohort students by training run, known direction against summed DC.

Scores the stored outputs of the three runs (A, B, C are seeds 0, 1, 2) on the
512 trained queries of each list. Count pairs give known-direction/summed-DC
students. Inv. counts outputs whose inventory-matched name equals the selected
label. The open gain, wins and losses and the exact McNemar p value compare
the two students of the same run. The paper's bootstrap intervals are not
recomputed.
"""

import ddv
from common import RUNS, Pool, Report, f2, parser

POOLS = [("uniform64", "Uniform"), ("lexical64", "Lexical")]
KEYS = ("open_accuracy", "whole_accuracy", "inventory_accuracy", "inventory_selected_label")
PAPER = {
    "Uniform A": ["440/365", "440/365", "448/375", "509/510", "14.65", "84/9", "2.2e-16"],
    "Uniform B": ["435/368", "435/368", "449/375", "510/511", "13.09", "85/18", "1.4e-11"],
    "Uniform C": ["435/371", "434/371", "447/375", "509/511", "12.50", "79/15", "1.2e-11"],
    "Lexical A": ["359/275", "359/275", "365/285", "512/509", "16.41", "110/26", "1.7e-13"],
    "Lexical B": ["359/282", "359/282", "364/286", "511/510", "15.04", "100/23", "1.2e-12"],
    "Lexical C": ["355/280", "354/280", "363/286", "510/512", "14.65", "103/28", "2.8e-11"],
}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    report, rows = Report(), {}
    for name, label in POOLS:
        pool = Pool(name, args.data)
        matcher = ddv.InventoryMatcher(pool.inventory)
        known = pool.student_flags("known_mean", matcher)
        dc = pool.student_flags("dc_sum", matcher)
        for run, k, d in zip(RUNS, known, dc):
            cells = [f"{sum(r[key] for r in k)}/{sum(r[key] for r in d)}" for key in KEYS]
            wins, losses = ddv.paired_counts([r["open_accuracy"] for r in k], [r["open_accuracy"] for r in d])
            cells += [f2(100 * (wins - losses) / len(k)), f"{wins}/{losses}",
                      f"{ddv.mcnemar_p(wins, losses):.1e}"]
            rows[f"{label} {run}"] = cells
    report.table("Table 17. Unscreened students by run (counts out of 512, known/summed DC)",
                 ("Open", "Whole", "Matched", "Inv.", "Open gain", "W/L", "p"), rows, PAPER)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
