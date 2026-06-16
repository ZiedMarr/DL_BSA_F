# training.py

import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import BiosignalDataset
from models import get_model
from utils import (
    check_no_leakage,
    get_subject_ids,
    group_kfold_split,
    kfold_split_indices,
    lmso_split,
    loso_split,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def macro_auc(y_true, y_prob):
    aucs = []

    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]

        if len(np.unique(class_true)) < 2:
            continue

        aucs.append(roc_auc_score(class_true, y_prob[:, class_idx]))

    if len(aucs) == 0:
        return None

    return float(np.mean(aucs))


def evaluate(model, loader, device):
    model.eval()
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"]

            outputs = model(x)
            probs = torch.sigmoid(outputs).cpu()

            true_labels.append(y.numpy())
            probabilities.append(probs.numpy())

    y_true = np.concatenate(true_labels, axis=0)
    y_prob = np.concatenate(probabilities, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_auc": macro_auc(y_true, y_prob),
        "per_class_f1": per_class_f1.tolist(),
    }


def save_results(results, config):
    results_path = config["paths"]["results"]
    model_name = config["model"]["name"]
    json_path = os.path.join(results_path, f"{model_name}_results.json")
    csv_path = os.path.join(results_path, f"{model_name}_summary.csv")

    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "loss", "macro_f1", "micro_f1", "macro_auc"])

        for fold_result in results:
            final = fold_result["final"]
            writer.writerow(
                [
                    fold_result["fold"],
                    final["loss"],
                    final["macro_f1"],
                    final["micro_f1"],
                    final["macro_auc"],
                ]
            )


def make_splits(config):
    protocol = config["evaluation"]["protocol"]
    num_folds = config["evaluation"]["num_folds"]

    if protocol in ["loso", "lmso", "group_kfold"]:
        subject_ids = get_subject_ids(config["paths"]["processed_data"])

        if protocol == "loso":
            return loso_split(subject_ids)

        if protocol == "lmso":
            return lmso_split(subject_ids, k=num_folds)

        return group_kfold_split(subject_ids, k=num_folds)

    if protocol == "kfold":
        full_dataset = BiosignalDataset(config, subject_ids=None)
        return kfold_split_indices(len(full_dataset), num_folds)

    raise ValueError(f"Unknown evaluation protocol: {protocol}")


def run_experiment(config):
    device = config["training"]["device"]
    protocol = config["evaluation"]["protocol"]
    splits = make_splits(config)
    results = []

    for fold, split in enumerate(splits):
        print(f"\nFold {fold + 1}")

        if protocol in ["loso", "lmso", "group_kfold"]:
            train_sids, test_sids = split
            check_no_leakage(train_sids, test_sids)

            train_dataset = BiosignalDataset(config, train_sids)
            test_dataset = BiosignalDataset(config, test_sids)

        else:
            train_idx, test_idx = split
            full_dataset = BiosignalDataset(config, subject_ids=None)
            train_dataset = Subset(full_dataset, train_idx)
            test_dataset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        model = get_model(config).to(device)
        dummy = torch.randn(
            2,
            config["dataset"]["input_channels"],
            config["dataset"]["segment_length"],
        ).to(device)

        try:
            _ = model(dummy)
        except Exception as exc:
            raise RuntimeError(f"Model forward failed: {exc}") from exc

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
        )

        criterion = nn.BCEWithLogitsLoss()
        fold_history = []

        for epoch in range(config["training"]["epochs"]):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, test_loader, device)

            print(
                f"Epoch {epoch + 1}: "
                f"Loss={loss:.4f}, "
                f"Macro-F1={metrics['macro_f1']:.4f}, "
                f"Macro-AUC={metrics['macro_auc']}"
            )

            fold_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss),
                    **metrics,
                }
            )

        torch.save(
            model.state_dict(),
            os.path.join(
                config["paths"]["checkpoints"],
                f"{config['model']['name']}_fold{fold}.pt",
            ),
        )

        results.append(
            {
                "fold": fold + 1,
                "final": fold_history[-1],
                "history": fold_history,
            }
        )

    final_f1 = [fold["final"]["macro_f1"] for fold in results]
    final_auc = [
        fold["final"]["macro_auc"]
        for fold in results
        if fold["final"]["macro_auc"] is not None
    ]

    print("\nFinal Results")
    print(f"Mean Macro-F1: {sum(final_f1) / len(final_f1):.4f}")

    if len(final_auc) > 0:
        print(f"Mean Macro-AUC: {sum(final_auc) / len(final_auc):.4f}")

    save_results(results, config)

    return results
