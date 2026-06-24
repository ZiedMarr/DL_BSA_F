# utils.py

import os
import random
import numpy as np
import torch
from sklearn.model_selection import KFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_subject_ids(data_path):
    subject_ids = set()

    for file in os.listdir(data_path):
        if not file.endswith(".npy"):
            continue

        try:
            data = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            subject_ids.add(data["subject_id"])
        except Exception:
            continue

    return sorted(list(subject_ids))


def loso_split(subject_ids):
    splits = []

    for test_sid in subject_ids:
        train_sids = [sid for sid in subject_ids if sid != test_sid]
        splits.append((train_sids, [test_sid]))

    return splits


def lmso_split(subject_ids, k=5, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    splits = []

    for train_idx, test_idx in kf.split(subject_ids):
        train_sids = [subject_ids[i] for i in train_idx]
        test_sids = [subject_ids[i] for i in test_idx]

        splits.append((train_sids, test_sids))

    return splits


def group_kfold_split(subject_ids, k=10, random_state=42):
    # Keep all segments from the same recording in the same fold.
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    splits = []

    for train_idx, test_idx in kf.split(subject_ids):
        train_sids = [subject_ids[i] for i in train_idx]
        test_sids = [subject_ids[i] for i in test_idx]
        splits.append((train_sids, test_sids))

    return splits


def kfold_split_indices(n_samples, k=5, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    splits = []
    indices = np.arange(n_samples)

    for train_idx, test_idx in kf.split(indices):
        splits.append((train_idx, test_idx))

    return splits


def create_folders(config):
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)


def check_no_leakage(train_sids, test_sids):
    overlap = set(train_sids).intersection(set(test_sids))
    assert len(overlap) == 0, f"Data leakage detected: {overlap}"


def save_file(dict):
    pass
