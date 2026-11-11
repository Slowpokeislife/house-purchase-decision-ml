"""CLI for Kaggle data, model training, tuning and single-row inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from .config import DEFAULT_DATA


def main(argv=None):
    parser = argparse.ArgumentParser(prog="house-purchase", description="House purchase decision classification")
    sub = parser.add_subparsers(dest="command", required=True)
    downloader = sub.add_parser("download", help="Download Kaggle dataset to ignored data/raw")
    downloader.add_argument("--destination", type=Path, default=DEFAULT_DATA)
    for name in ("baseline", "train", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--out", type=Path)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--tracking", action="store_true", help="Opt-in MLflow logging (local by default)")
        if name == "train":
            p.add_argument("--architecture", choices=("original", "optimized"), default="optimized")
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--batch-size", type=int, default=64)
        if name == "tune":
            p.add_argument("--trials", type=int, default=20)
            p.add_argument("--trial-epochs", type=int, default=25)
            p.add_argument("--final-epochs", type=int, default=100)
    pred = sub.add_parser("predict", help="Predict from a JSON with all 23 raw features")
    pred.add_argument("--artifacts", type=Path, default=Path("artifacts/optimized"))
    pred.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        from .data import download_dataset
        print(download_dataset(args.destination))
    elif args.command == "predict":
        from .predict import predict_one
        record = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(predict_one(args.artifacts, record), indent=2))
    else:
        from .experiments import run_baseline, run_training, run_tuning
        common = {"data": args.data, "seed": args.seed, "tracking": args.tracking}
        if args.out is not None:
            common["out"] = args.out
        if args.command == "baseline":
            metrics = run_baseline(**common)
        elif args.command == "train":
            metrics = run_training(architecture=args.architecture, epochs=args.epochs,
                                   batch_size=args.batch_size, **common)
        else:
            metrics = run_tuning(trials=args.trials, trial_epochs=args.trial_epochs,
                                 final_epochs=args.final_epochs, **common)
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
