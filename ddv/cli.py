"""The `ddv` command. `python -m ddv` runs the same thing."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from . import __version__

# Section 3 sizes of the bundled pools: queries, evaluated queries, retained
# forward pairs, withheld children and candidate slots.
DIMENSIONS = {"screened": (1500, 1390, 7888, 2000, 28463),
              "uniform64": (512, 512, 9757, 535, 32768),
              "lexical64": (512, 512, 9757, 535, 32768)}


def _demo(args):
    from . import domain_context, select_label

    parent = "Susana Dosamantes"
    candidates = ["Paulina Rubio", "Diego Luna", "Odiseo Bichir", "Selena Gomez"]
    known = [[-2.46, -2.93, -2.87, -4.23]]
    reverse = [[-15.61, -13.60, -23.71, -14.30]]
    context = [[-24.78, -21.12, -35.68, -18.51]]
    print("known direction:", select_label(known, candidates, parent))
    print("reverse:        ", select_label(reverse, candidates, parent))
    print("reverse with DC:", select_label(domain_context(reverse, context), candidates, parent))
    return 0


def _select(args):
    from .data import TEACHERS, candidate_fingerprint, read_json, write_json
    from .labels import select_labels

    data = read_json(args.data)
    records = [read_json(path) for path in args.scores]
    tags = [r["teacher"] for r in records]
    if len(set(tags)) != len(tags):
        raise SystemExit("Supply one score file per teacher.")
    expected = candidate_fingerprint(data["queries"])
    if any(r["candidate_fingerprint"] != expected for r in records):
        raise SystemExit("Score files belong to a different candidate pool.")
    if set(tags) <= set(TEACHERS):
        records.sort(key=lambda r: TEACHERS.index(r["teacher"]))
    labels = select_labels(data["queries"], [r["scores"] for r in records], rule=args.rule)
    for row, label in zip(data["queries"], labels):
        row["label"] = label
    write_json(args.out, data)
    print(f"Selected {len(labels)} labels with the {args.rule} rule from {len(records)} channels.")
    return 0


def _validate(args):
    from .data import load_inventory, load_pool, load_scores, validate_dataset
    from .labels import select_labels

    root = Path(args.root) / "data"
    if not (root / "inventory.json").exists():
        raise SystemExit(f"No bundled data under {root}. Run this from a clone of the repository.")
    inventory = load_inventory(root)
    for name, dimensions in DIMENSIONS.items():
        data = load_pool(name, root)
        result = validate_dataset(data, inventory)
        actual = tuple(result[k] for k in ["queries", "evaluated_queries", "forward_pairs",
                                           "withheld_children", "candidate_slots"])
        if actual != dimensions:
            raise SystemExit(f"Unexpected dataset dimensions: {name}")
        labels = select_labels(data["queries"], load_scores(name, root)["known_mean"])
        if labels != [r["label"] for r in data["queries"]]:
            raise SystemExit(f"Selected labels differ: {name}")
        print(f"{name}: {result['queries']} queries, {result['candidate_slots']} candidate slots, "
              f"{result['evaluated_queries']} evaluated, all {len(labels)} labels match")
    return 0


def _evaluate(args):
    from .data import read_json, validate_dataset, write_json
    from .metrics import evaluate_predictions

    data = read_json(args.data)
    inventory = read_json(args.inventory)["child_parents"]
    validate_dataset(data, inventory)
    predictions = read_json(args.predictions)
    if isinstance(predictions, dict):  # a bundled file keeps the rows under "predictions"
        predictions = predictions["predictions"]
    metrics = evaluate_predictions(data, inventory, predictions)
    write_json(args.out, metrics)
    for key, value in metrics["percent"].items():
        print(f"{key:26s} {value:6.2f}")
    return 0


def _summarize(args):
    from .data import read_json, write_json

    records = [read_json(Path(path) / "run.json") for path in args.runs]
    if len({r["seed"] for r in records}) != len(records):
        raise SystemExit("Supply runs with distinct training seeds.")
    for key in ["dataset_fingerprint", "inventory_fingerprint", "config"]:
        if any(r[key] != records[0][key] for r in records):
            raise SystemExit(f"Runs have different {key} values.")
    metrics = [read_json(Path(path) / "metrics.json") for path in args.runs]
    if len(metrics) < 2 or any(m["n"] != metrics[0]["n"] for m in metrics):
        raise SystemExit("At least two runs with the same evaluation size are needed.")
    summary = {}
    for key in metrics[0]["percent"]:
        values = [m["percent"][key] for m in metrics]
        summary[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
        print(f"{key:26s} {summary[key]['mean']:6.2f} +- {summary[key]['sample_sd']:.2f}")
    if args.out:
        write_json(args.out, {"n_runs": len(metrics), "n_queries": metrics[0]["n"],
                              "percent": summary})
    return 0


def main(argv=None):
    from .labels import RULES

    parser = argparse.ArgumentParser(
        prog="ddv", description="Directional label distillation: pseudo-labels from known-direction scores.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart")
    demo.set_defaults(func=_demo)

    select = sub.add_parser("select", help="select labels from teacher score files")
    select.add_argument("--data", required=True, help="pool JSON with queries and candidates")
    select.add_argument("--scores", nargs="+", required=True, help="one score file per teacher")
    select.add_argument("--rule", choices=RULES, default="mean")
    select.add_argument("--out", required=True)
    select.set_defaults(func=_select)

    validate = sub.add_parser("validate", help="rebuild every bundled label from the bundled scores")
    validate.add_argument("--root", default=".", help="the repository checkout")
    validate.set_defaults(func=_validate)

    evaluate = sub.add_parser("evaluate", help="score saved student predictions")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--inventory", default="data/inventory.json")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.set_defaults(func=_evaluate)

    summarize = sub.add_parser("summarize", help="mean and sample SD over seeded student runs")
    summarize.add_argument("--runs", nargs="+", required=True)
    summarize.add_argument("--out")
    summarize.set_defaults(func=_summarize)

    args = parser.parse_args(argv)
    return args.func(args)
