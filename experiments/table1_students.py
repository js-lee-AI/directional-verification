"""Table 1. MDM-0.6B students trained on selected reverse labels.

x is label accuracy, rebuilt from the bundled teacher scores. The student
columns score the stored outputs of the three training runs behind each row
(three-seed mean and sample SD). To score your own runs instead, train them
with train_student.py and run `ddv summarize --runs <run dirs>`.
"""

import ddv
from common import Pool, Report, f2, parser, pct, pm

BLOCKS = [
    ("screened", "Acquired", [("reverse_mean", "Reverse, mean token"),
                              ("dc_mean", "DC reverse, mean token"),
                              ("dc_sum", "DC reverse, summed token"),
                              ("known_mean", "Known direction, mean token")]),
    ("uniform64", "Uniform", [("dc_sum", "DC reverse, summed token"),
                              ("known_mean", "Known direction, mean token")]),
    ("lexical64", "Lexical", [("dc_sum", "DC reverse, summed token"),
                              ("known_mean", "Known direction, mean token")]),
]
COLUMNS = ("x", "Open", "Whole", "Matched", "String", "Inventory")
KEYS = ("open_accuracy", "whole_accuracy", "inventory_accuracy",
        "whole_selected_label", "inventory_selected_label")

PAPER = {
    "Acquired  Reverse, mean token":
        ["53.60", "47.17 ± 1.29", "47.00 ± 1.27", "53.14 ± 0.41", "79.59 ± 1.74", "97.67 ± 0.30"],
    "Acquired  DC reverse, mean token":
        ["54.82", "49.26 ± 0.08", "49.06 ± 0.07", "53.93 ± 0.48", "90.36 ± 0.29", "97.65 ± 0.79"],
    "Acquired  DC reverse, summed token":
        ["75.18", "63.26 ± 1.30", "63.14 ± 1.37", "73.98 ± 0.23", "79.98 ± 1.35", "97.39 ± 0.51"],
    "Acquired  Known direction, mean token":
        ["90.72", "78.03 ± 2.86", "77.75 ± 2.91", "88.94 ± 0.46", "84.56 ± 3.02", "97.70 ± 0.61"],
    "Uniform  DC reverse, summed token":
        ["73.44", "71.88 ± 0.59", "71.88 ± 0.59", "73.24 ± 0.00", "97.33 ± 0.11", "99.74 ± 0.11"],
    "Uniform  Known direction, mean token":
        ["87.89", "85.29 ± 0.56", "85.22 ± 0.63", "87.50 ± 0.20", "96.81 ± 0.60", "99.48 ± 0.11"],
    "Lexical  DC reverse, summed token":
        ["55.86", "54.49 ± 0.70", "54.49 ± 0.70", "55.79 ± 0.11", "97.07 ± 0.68", "99.67 ± 0.30"],
    "Lexical  Known direction, mean token":
        ["71.29", "69.86 ± 0.45", "69.79 ± 0.56", "71.09 ± 0.20", "97.59 ± 0.69", "99.80 ± 0.20"],
}
# Open-accuracy gain of known-direction over summed-DC students (Section 4.1).
# The paper prints the acquired and lexical gains, so only those two are shown.
PAPER_GAIN = {"Acquired": ["14.77"], "Lexical": ["15.36"]}


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    report, rows, gains, matcher = Report(), {}, {}, None
    for name, block, sources in BLOCKS:
        pool = Pool(name, args.data)
        matcher = matcher or ddv.InventoryMatcher(pool.inventory)
        opens = {}
        for form, label in sources:
            runs = pool.student_flags(form, matcher)
            cells = [f2(pool.accuracy(form, subset=pool.evaluated))]
            for key in KEYS:
                cells.append(pm([pct([row[key] for row in run]) for run in runs]))
            rows[f"{block}  {label}"] = cells
            opens[form] = [pct([row["open_accuracy"] for row in run]) for run in runs]
        if block in PAPER_GAIN:
            gain = sum(k - d for k, d in zip(opens["known_mean"], opens["dc_sum"])) / 3
            gains[block] = [f2(gain)]
    report.table("Table 1. Students on the evaluated queries (%)", COLUMNS, rows, PAPER)
    report.table("Section 4.1. Open-accuracy gain, known minus summed DC (points)",
                 ("gain",), gains, PAPER_GAIN)
    print("\nNot rebuilt here, their score files are not bundled: the Llama two-template and "
          "eight-channel agreement rows, and the tuned reverse rows.")
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
