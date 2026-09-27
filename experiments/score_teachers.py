"""Score a candidate pool with one frozen teacher. Needs a GPU and the train extra.

The default writes known-direction scores in the format of data/scores/<pool>_<teacher>.json.gz.
With --comparator it writes the reverse and context-only scores with their
token counts, the format of data/scores/<pool>_<teacher>_reverse.json.gz.
Scores use raw prompts without chat templates.

    python experiments/score_teachers.py --pool data/screened.json --teacher qwen \\
        --out scores/screened_qwen.json.gz
"""

import argparse

import ddv
from ddv.data import candidate_fingerprint

TEACHERS = "configs/teachers.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", required=True, help="a pool JSON such as data/screened.json")
    parser.add_argument("--teacher", required=True, choices=ddv.TEACHERS)
    parser.add_argument("--teachers", default=TEACHERS, help="model names and revisions")
    parser.add_argument("--comparator", action="store_true",
                        help="score the reverse direction and the context-only names instead")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = ddv.read_json(args.pool)
    queries = data["queries"]
    spec = ddv.read_json(args.teachers)[args.teacher]
    model, tokenizer = ddv.load_teacher(spec, args.device)
    record = {"teacher": args.teacher, "model": spec,
              "candidate_fingerprint": candidate_fingerprint(queries)}
    if not args.comparator:
        record["scores"] = []
        for i, row in enumerate(queries):
            pairs = ddv.known_pairs(row["parent"], row["candidates"])
            record["scores"].append(ddv.score_pairs(model, tokenizer, pairs, args.batch_size))
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(queries)} queries", flush=True)
    else:
        record.update(reverse_mean=[], reverse_tokens=[], parent_tokens=[])
        for i, row in enumerate(queries):
            pairs = ddv.reverse_pairs(row["parent"], row["candidates"])
            record["reverse_mean"].append(ddv.score_pairs(model, tokenizer, pairs, args.batch_size))
            record["reverse_tokens"].append(ddv.continuation_lengths(tokenizer, pairs))
            known = ddv.known_pairs(row["parent"], row["candidates"][:1])
            record["parent_tokens"].append(ddv.continuation_lengths(tokenizer, known)[0])
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(queries)} queries", flush=True)
        names = sorted({c for row in queries for c in row["candidates"]})
        pairs = ddv.context_pairs(names)
        record["context_mean"] = dict(zip(names, ddv.score_pairs(model, tokenizer, pairs, args.batch_size)))
        record["context_tokens"] = dict(zip(names, ddv.continuation_lengths(tokenizer, pairs)))
    ddv.write_json(args.out, record)


if __name__ == "__main__":
    main()
