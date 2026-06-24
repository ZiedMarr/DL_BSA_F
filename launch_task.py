# launch_task.py
# Per-task launcher for Slurm array jobs.
# Reads hparam_grid.json[task_id] and runs the corresponding experiment.
#
# Usage:
#   python launch_task.py <task_id>
#   python launch_task.py 0          # test locally against first job in the grid

import json
import sys

from config import get_config
from training import run_experiment
from utils import create_folders

# Explicit mapping: flat hparam key → (config section, config key)
# Add entries here whenever GRID in generate_grid.py gains a new key.
# An unrecognized key raises immediately so grid and launcher can't drift apart.
HPARAM_TO_CONFIG = {
    "learning_rate":   ("training",   "learning_rate"),
    "batch_size":      ("training",   "batch_size"),
    "epochs":          ("training",   "epochs"),
    "dropout":         ("training",   "dropout"),
    "seed":            ("training",   "seed"),
    "class_threshold": ("evaluation", "class_threshold"),
    "num_folds":       ("evaluation", "num_folds"),
    "protocol":        ("evaluation", "protocol"),
    "model_name":      ("model",      "name"),
}

GRID_FILE = "hparam_grid.json"


def hparams_to_overrides(hparams):
    overrides = {}
    for key, value in hparams.items():
        if key not in HPARAM_TO_CONFIG:
            raise KeyError(
                f"Unrecognized hyperparameter key '{key}'. "
                f"Add it to HPARAM_TO_CONFIG in launch_task.py."
            )
        section, config_key = HPARAM_TO_CONFIG[key]
        overrides.setdefault(section, {})[config_key] = value
    return overrides


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <task_id>")
        sys.exit(1)

    task_id = int(sys.argv[1])

    with open(GRID_FILE) as f:
        grid = json.load(f)

    if task_id < 0 or task_id >= len(grid):
        raise IndexError(
            f"Task ID {task_id} out of range — grid has {len(grid)} entries (0–{len(grid)-1})."
        )

    hparams = grid[task_id]
    print(f"Task {task_id}: {hparams}")

    overrides = hparams_to_overrides(hparams)
    config = get_config(overrides)

    create_folders(config)
    run_experiment(config)


if __name__ == "__main__":
    main()
