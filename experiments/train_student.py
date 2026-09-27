"""Train one masked-diffusion student run (Table 1 rows). Needs a GPU with bf16 and the train extra.

Trains Qwen3-0.6B with the configuration of Section 3: 4,000 forward warm steps,
4,800 mixed SFT steps, batch 16, learning rate 1e-4, rank-64 LoRA. Forward
facts of withheld children are removed before both stages. --labels picks the
label source, reselected from the bundled scores, and three seeds give one row.

    for seed in 0 1 2; do
      python experiments/train_student.py --pool screened --labels known_mean --seed $seed \\
          --out runs/screened_known_mean_$seed
    done
    ddv summarize --runs runs/screened_known_mean_0 runs/screened_known_mean_1 runs/screened_known_mean_2
"""

import argparse
import copy

import ddv

LABELS = ("known_mean", "reverse_mean", "dc_mean", "dc_sum")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", choices=ddv.POOLS, help="a bundled pool")
    source.add_argument("--dataset", help="your own pool JSON with a label per query")
    parser.add_argument("--labels", choices=LABELS, default="known_mean",
                        help="label source for a bundled pool")
    parser.add_argument("--data", default="data")
    parser.add_argument("--config", default="configs/student.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    inventory = ddv.load_inventory(args.data)
    if args.dataset:
        dataset = ddv.read_json(args.dataset)
    else:
        dataset = copy.deepcopy(ddv.load_pool(args.pool, args.data))
        if args.labels != "known_mean":
            channels = ddv.load_scores(args.pool, args.data, forms=(args.labels,))[args.labels]
            for row, label in zip(dataset["queries"], ddv.select_labels(dataset["queries"], channels)):
                row["label"] = label
    metrics = ddv.train_student(dataset, inventory, ddv.read_json(args.config), args.out,
                                seed=args.seed, device=args.device)
    for key, value in metrics["percent"].items():
        print(f"{key:26s} {value:6.2f}")


if __name__ == "__main__":
    main()
