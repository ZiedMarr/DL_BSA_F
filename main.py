# main.py

import argparse
import os

from config import get_config
from training import run_experiment
from utils import create_folders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn1d", "resnet", "cnn_vit"],
        help="Model to train. If not set, config.py is used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. If not set, config.py is used.",
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

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    create_folders(config)

    if not processed_data_exists(config):
        from preprocessing import preprocess_dataset

        preprocess_dataset(config)

    run_experiment(config)


if __name__ == "__main__":
    main()
