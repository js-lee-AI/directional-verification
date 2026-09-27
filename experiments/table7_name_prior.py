"""Table 7, Table 8 and Figure 3. Label accuracy by name prior and continuation length.

A candidate's name score is its summed context-only log probability after
"The child is", averaged over the four teachers. Its continuation length is the
mean token count of the child name across the teachers' tokenizers. Terciles
are cut within each pool on the evaluated queries whose recorded child is a
candidate. K is the mean-token known direction, R and D the summed reverse and
DC scores. Intervals resample parents 10,000 times with the seed below.
"""

from collections import defaultdict

import numpy as np

import ddv
from common import Pool, Report, f1, parser

SEED, DRAWS = 20260911, 10000
POOLS = [("screened", "Acquired"), ("uniform64", "Uniform"), ("lexical64", "Lexical")]
COLUMNS = ("n", "K", "R", "D", "K - D", "95% CI")

PAPER = {
    "Acquired  name  Low": ["449", "95.1", "61.7", "89.8", "5.3", "[2.7, 8.1]"],
    "Acquired  name  Mid": ["448", "92.2", "78.1", "85.7", "6.5", "[3.1, 10.0]"],
    "Acquired  name  High": ["449", "93.5", "82.4", "57.5", "36.1", "[31.4, 40.7]"],
    "Uniform  name  Low": ["171", "90.6", "49.7", "86.5", "4.1", "[0.6, 8.2]"],
    "Uniform  name  Mid": ["170", "82.4", "61.2", "70.0", "12.4", "[6.5, 18.8]"],
    "Uniform  name  High": ["171", "90.6", "69.6", "63.7", "26.9", "[19.9, 33.9]"],
    "Lexical  name  Low": ["171", "75.4", "40.4", "74.9", "0.6", "[-4.7, 5.8]"],
    "Lexical  name  Mid": ["170", "63.5", "46.5", "53.5", "10.0", "[3.5, 16.5]"],
    "Lexical  name  High": ["171", "74.9", "53.8", "39.2", "35.7", "[28.1, 43.3]"],
    "Acquired  length  Low": ["568", "93.0", "81.0", "76.2", "16.7", "[13.1, 20.4]"],
    "Acquired  length  Mid": ["363", "94.2", "76.0", "77.7", "16.5", "[12.1, 21.0]"],
    "Acquired  length  High": ["415", "94.0", "62.9", "79.5", "14.5", "[10.8, 18.3]"],
    "Uniform  length  Low": ["201", "84.1", "66.7", "64.7", "19.4", "[12.9, 25.9]"],
    "Uniform  length  Mid": ["143", "86.7", "61.5", "75.5", "11.2", "[6.3, 16.8]"],
    "Uniform  length  High": ["168", "93.5", "51.2", "82.1", "11.3", "[6.5, 16.7]"],
    "Lexical  length  Low": ["201", "65.2", "49.3", "43.8", "21.4", "[14.4, 28.4]"],
    "Lexical  length  Mid": ["143", "70.6", "48.3", "58.7", "11.9", "[4.9, 18.9]"],
    "Lexical  length  High": ["168", "79.2", "42.9", "67.9", "11.3", "[4.8, 17.9]"],
}
PAPER_WRONG = {
    "Acquired": ["1,346", "40/86", "315/349", "38/301"],
    "Uniform": ["512", "15/62", "199/204", "19/136"],
    "Lexical": ["512", "57/147", "245/272", "68/226"],
}


def cluster_interval(diff, parents, rng):
    groups = defaultdict(list)
    for d, p in zip(diff, parents):
        groups[p].append(d)
    sums = np.array([sum(v) for v in groups.values()], float)
    counts = np.array([len(v) for v in groups.values()], float)
    idx = rng.integers(0, len(sums), size=(DRAWS, len(sums)))
    return np.quantile(sums[idx].sum(1) / counts[idx].sum(1), [0.025, 0.975])


def terciles(values):
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.where(values <= q1, 0, np.where(values <= q2, 1, 2))


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    report, wrong, strata = Report(), {}, {}
    for name, block in POOLS:
        pool = Pool(name, args.data)
        gold = ddv.parent_children(pool.inventory)
        labels = {k: pool.labels(f) for k, f in (("K", "known_mean"), ("R", "reverse_sum"), ("D", "dc_sum"))}
        prior = {c: float(np.mean(v)) for c, v in ddv.context_sums(name, args.data).items()}
        lengths = [ddv.read_json(pool.root / "scores" / f"{name}_{t}_reverse.json.gz")["reverse_tokens"]
                   for t in ddv.TEACHERS]
        ok = {k: [] for k in "KRD"}
        misses = {k: [0, 0] for k in "KRD"}
        values = {"name": [], "length": []}
        parents = []
        for i in pool.evaluated:
            q = pool.queries[i]
            if q["child"] not in q["candidates"]:
                continue
            j = q["candidates"].index(q["child"])
            for k in "KRD":
                good = labels[k][i] in gold[q["parent"]]
                ok[k].append(good)
                if not good:
                    misses[k][0] += prior[labels[k][i]] > prior[q["child"]]
                    misses[k][1] += 1
            values["name"].append(prior[q["child"]])
            values["length"].append(float(np.mean([t[i][j] for t in lengths])))
            parents.append(q["parent"])
        wrong[block] = [f"{len(parents):,}"] + [f"{a}/{b}" for a, b in (misses[k] for k in "KRD")]
        strata[block] = (ok, values, parents)

    by_kind = {"name": {}, "length": {}}
    for block, (ok, values, parents) in strata.items():
        # One random stream per pool, drawn in the order of the original analysis.
        # Its first draw is the pooled interval, which the table does not print.
        rng = np.random.default_rng(SEED)
        K, R, D = (np.asarray(ok[k], bool) for k in "KRD")
        cluster_interval((K.astype(int) - D.astype(int)).tolist(), parents, rng)
        for kind in ("name", "length"):
            bins = terciles(np.asarray(values[kind], float))
            for b, level in enumerate(("Low", "Mid", "High")):
                m = bins == b
                diff = (K[m].astype(int) - D[m].astype(int)).tolist()
                lo, hi = cluster_interval(diff, [p for p, t in zip(parents, m) if t], rng)
                by_kind[kind][f"{block}  {kind}  {level}"] = [
                    str(int(m.sum())), f1(100 * K[m].mean()), f1(100 * R[m].mean()),
                    f1(100 * D[m].mean()), f1(100 * np.mean(diff)), f"[{100 * lo:.1f}, {100 * hi:.1f}]"]
    rows = {**by_kind["name"], **by_kind["length"]}
    report.table("Table 7 and Figure 3. Label accuracy (%) by tercile of the true child's name score "
                 "and continuation length", COLUMNS, rows, PAPER)
    report.table("Table 8. Wrong selections whose name score exceeds the true child's",
                 ("n", "K", "R", "D"), wrong, PAPER_WRONG)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
