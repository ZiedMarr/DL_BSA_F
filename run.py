# run.py
# Single-run CLI entry point. Useful for quick sanity checks and ad-hoc runs.
# Example: python run.py --epochs 1 --learning_rate 1e-4 --model_name resnet

import argparse
import os

from config import get_config
from training import run_experiment
from utils import create_folders


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--class_threshold", type=float)
    parser.add_argument("--num_folds", type=int)
    parser.add_argument("--protocol", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def build_overrides(args):
    overrides = {}

    training_keys = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "seed": args.seed,
    }
    training_overrides = {k: v for k, v in training_keys.items() if v is not None}
    if training_overrides:
        overrides["training"] = training_overrides

    evaluation_keys = {
        "class_threshold": args.class_threshold,
        "num_folds": args.num_folds,
        "protocol": args.protocol,
    }
    evaluation_overrides = {k: v for k, v in evaluation_keys.items() if v is not None}
    if evaluation_overrides:
        overrides["evaluation"] = evaluation_overrides

    if args.model_name is not None:
        overrides["model"] = {"name": args.model_name}

    return overrides


def processed_data_exists(config):
    data_path = config["paths"]["processed_data"]
    if not os.path.isdir(data_path):
        return False
    return any(f.endswith(".npy") for f in os.listdir(data_path))


def main():
    args = parse_args()
    overrides = build_overrides(args)
    config = get_config(overrides if overrides else None)

    create_folders(config)

    if not processed_data_exists(config):
        from preprocessing import preprocess_dataset
        preprocess_dataset(config)

    run_experiment(config)


if __name__ == "__main__":
    main()
