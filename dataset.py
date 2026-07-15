# dataset.py

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class BiosignalDataset(Dataset):
    def __init__(self, config, subject_ids=None, transform=None):
        self.data = []
        self.transform = transform

        data_path = config["paths"]["processed_data"]
        expected_channels = config["dataset"]["input_channels"]
        expected_segment_length = config["dataset"]["segment_length"]
        expected_num_classes = config["dataset"]["num_classes"]

        if subject_ids is not None:
            subject_ids = set(subject_ids)

        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Processed data folder not found: {data_path}. "
                "Run preprocessing before creating the dataset."
            )

        for file in sorted(os.listdir(data_path)):
            if not file.endswith(".npy"):
                continue

            file_path = os.path.join(data_path, file)

            try:
                sample = np.load(file_path, allow_pickle=True).item()
            except Exception as exc:
                raise ValueError(f"Could not load processed file {file_path}: {exc}") from exc

            subject_id = sample["subject_id"]

            if subject_ids is not None and subject_id not in subject_ids:
                continue

            signals = np.asarray(sample["signals"], dtype=np.float32)
            labels = np.asarray(sample["labels"], dtype=np.float32)

            if signals.ndim != 3:
                raise ValueError(f"{file_path}: signals must have shape (N, C, T), got {signals.shape}")

            if signals.shape[1:] != (expected_channels, expected_segment_length):
                raise ValueError(
                    f"{file_path}: expected signals shape (N, {expected_channels}, "
                    f"{expected_segment_length}), got {signals.shape}"
                )

            if labels.ndim != 2:
                raise ValueError(f"{file_path}: labels must have shape (N, num_classes), got {labels.shape}")

            if labels.shape != (signals.shape[0], expected_num_classes):
                raise ValueError(
                    f"{file_path}: expected labels shape ({signals.shape[0]}, "
                    f"{expected_num_classes}), got {labels.shape}"
                )

            for i in range(len(signals)):
                self.data.append(
                    {
                        "signal": signals[i],
                        "label": labels[i],
                        "subject_id": subject_id,
                    }
                )

        if len(self.data) == 0:
            raise ValueError(
                f"No samples found in {data_path}. "
                "Check that preprocessing created .npy files and that subject_ids is correct."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        x = torch.tensor(item["signal"], dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(item["label"], dtype=torch.float32)

        return {
            "signal": x,
            "label": y,
            "subject_id": item["subject_id"],
        }

    def get_all_subject_ids(self):
        return sorted({item["subject_id"] for item in self.data})
