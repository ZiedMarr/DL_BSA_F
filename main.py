# main.py

import argparse
import os

from config import get_config
from preprocessing import preprocess_dataset
from training import run_experiment
from utils import create_folders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn1d", "resnet"],
        help="Model to train. If not set, config.py is used.",
    )
    return parser.parse_args()


def processed_data_exists(config):
    data_path = config["paths"]["processed_data"]

    if not os.path.isdir(data_path):
        return False

    return any(file.endswith(".npy") for file in os.listdir(data_path))


def main():
    args = parse_args()
    config = get_config()

    if args.model is not None:
        config["model"]["name"] = args.model

    create_folders(config)

    if not processed_data_exists(config):
        preprocess_dataset(config)

    run_experiment(config)


if __name__ == "__main__":
    main()
