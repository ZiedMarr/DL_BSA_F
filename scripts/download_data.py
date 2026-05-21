import subprocess


DATASET_ID = "physionet/china-physiological-signal-challenge-in-2018"
OUTPUT_DIR = "./data/raw"


if __name__ == "__main__":
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_ID,
            "-p",
            OUTPUT_DIR,
            "--unzip"
        ],
        check=True
    )
