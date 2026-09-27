"""Reload a trained student from its run folder and answer every evaluated query again.

Needs the train extra. Writes the new predictions and their metrics next to
--out. To score saved predictions without a model, use `ddv evaluate`.

    python experiments/evaluate_student.py --run runs/screened_known_mean_0 --out runs/reloaded.json
"""

import argparse
from pathlib import Path

import ddv


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    dataset = ddv.read_json(run / "dataset.json")
    inventory = ddv.load_inventory(args.data)
    record = ddv.read_json(run / "run.json")
    if record["inventory_fingerprint"] != ddv.data.fingerprint(inventory):
        raise SystemExit("The inventory differs from the training run.")
    model, tokenizer, mask_id, dtype = ddv.load_run(run, args.device)
    predictions = ddv.generate_predictions(model, tokenizer, mask_id, dtype, dataset,
                                           record["config"]["answer_slots"], args.device)
    ddv.write_json(Path(args.out).with_suffix(".predictions.json"), predictions)
    metrics = ddv.evaluate_predictions(dataset, inventory, predictions)
    ddv.write_json(args.out, metrics)
    for key, value in metrics["percent"].items():
        print(f"{key:26s} {value:6.2f}")


if __name__ == "__main__":
    main()
