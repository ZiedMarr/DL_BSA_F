# generate_grid.py
# Defines the hyperparameter search space and writes hparam_grid.json.
#
# Usage:
#   python generate_grid.py               # full cartesian product
#   python generate_grid.py --random 20   # random sample of 20 combos
#
# To run multiple repeats with different seeds, add "seed" to GRID:
#   "seed": [42, 142, 242]
# training.py derives per-fold seeds as (seed + fold_index) internally.
#
# After running, update --array in submit_array.sbatch to match the printed range.

import argparse
import itertools
import json
import random

# ---------------------------------------------------------------------------
# Search space — edit values here before each sweep
# ---------------------------------------------------------------------------
GRID = {
    "model_name":    ["cnn1d", "resnet"],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size":    [32, 64],
    "dropout":       [0.2, 0.3, 0.5],
    # "seed":        [42, 142, 242],  # uncomment for variance-estimation repeats
}

# Number of random combos to sample; None → full cartesian product
N_RANDOM_SAMPLES = None

OUTPUT_FILE = "hparam_grid.json"
# ---------------------------------------------------------------------------


def cartesian_combos(grid):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def random_combos(grid, n, rng_seed=0):
    rng = random.Random(rng_seed)
    keys = list(grid.keys())
    value_lists = list(grid.values())
    max_total = 1
    for v in value_lists:
        max_total *= len(v)
    n = min(n, max_total)

    seen = set()
    combos = []
    while len(combos) < n:
        candidate = tuple(rng.choice(vals) for vals in value_lists)
        if candidate not in seen:
            seen.add(candidate)
            combos.append(dict(zip(keys, candidate)))

    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        default=None,
        help="Sample N random combos instead of full cartesian product.",
    )
    args = parser.parse_args()

    n_random = args.random if args.random is not None else N_RANDOM_SAMPLES

    if n_random is not None:
        jobs = random_combos(GRID, n=n_random)
    else:
        jobs = list(cartesian_combos(GRID))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

    n_jobs = len(jobs)
    print(f"Wrote {n_jobs} jobs to {OUTPUT_FILE}")
    print(f"Use in sbatch:  --array=0-{n_jobs - 1}")


if __name__ == "__main__":
    main()
