"""Acquire candidate pools by scanning the full inventory. Needs a GPU and the train extra.

Each acquisition teacher scores every inventory name in the known direction and
keeps its eight best. merge unites the three lists in Qwen, OLMo, Mistral order
(Section 2.1). Rescore the merged pool with all four teachers
(score_teachers.py) and select labels (`ddv select`) before training.

    python experiments/acquire_candidates.py scan --data data/screened.json --teacher qwen --out scans/qwen.json
    python experiments/acquire_candidates.py merge --data data/screened.json \\
        --scans scans/qwen.json scans/olmo.json scans/mistral.json --out pools/screened.json
"""

import argparse

import numpy as np

import ddv
from ddv.data import fingerprint

ORDER = ["qwen", "olmo", "mistral"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--data", required=True)
    scan.add_argument("--inventory", default="data/inventory.json")
    scan.add_argument("--teachers", default="configs/teachers.json")
    scan.add_argument("--teacher", required=True, choices=ORDER)
    scan.add_argument("--batch-size", type=int, default=16)
    scan.add_argument("--device", default="cuda")
    scan.add_argument("--out", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--data", required=True)
    merge.add_argument("--scans", nargs=3, required=True)
    merge.add_argument("--out", required=True)
    args = parser.parse_args()

    data = ddv.read_json(args.data)
    signature = fingerprint([[r["id"], r["parent"]] for r in data["queries"]])
    if args.command == "scan":
        inventory = sorted(ddv.read_json(args.inventory)["child_parents"])
        model, tokenizer = ddv.load_teacher(ddv.read_json(args.teachers)[args.teacher], args.device)
        top = []
        for i, row in enumerate(data["queries"]):
            scores = ddv.score_pairs(model, tokenizer, ddv.known_pairs(row["parent"], inventory),
                                     args.batch_size)
            indices = np.argsort(-np.asarray(scores), kind="stable")[:8]
            top.append([inventory[j] for j in indices])
            print(f"{i + 1}/{len(data['queries'])} queries", flush=True)
        ddv.write_json(args.out, {"teacher": args.teacher, "query_fingerprint": signature, "top8": top})
        return
    scans = [ddv.read_json(path) for path in args.scans]
    if {s["teacher"] for s in scans} != set(ORDER):
        raise SystemExit("Supply one scan from each acquisition teacher.")
    scans.sort(key=lambda s: ORDER.index(s["teacher"]))
    if any(s["query_fingerprint"] != signature or len(s["top8"]) != len(data["queries"]) for s in scans):
        raise SystemExit("Scans belong to different queries.")
    for i, row in enumerate(data["queries"]):
        row["candidates"] = list(dict.fromkeys(c for s in scans for c in s["top8"][i]))
        row.pop("label", None)
    ddv.write_json(args.out, data)


if __name__ == "__main__":
    main()
