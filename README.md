# DL-BSA ECG Classification Project

This project trains a deep learning model for multi-label ECG classification using the China Physiological Signal Challenge 2018 dataset.

The pipeline is:

```text
raw WFDB ECG files
-> preprocessing
-> processed .npy files
-> PyTorch dataset
-> CNN + BiGRU + Attention model
-> 10-fold group cross-validation
```

## Data Layout

The raw data should be placed here:

```text
data/raw/Training_WFDB/
data/raw/REFERENCE.csv
```

The preprocessing step creates processed files here:

```text
data/processed/
```

Each processed subject file is saved as:

```python
{
    "signals": (num_segments, 12, 1500),
    "labels": (num_segments, 9),
    "subject_id": "A0001"
}
```

The 1500 samples correspond to 6 seconds after downsampling to 250 Hz.

## Files

`config.py`

Stores the main settings, including paths, sampling rate, segment length, batch size, number of epochs, and evaluation protocol.

`preprocessing.py`

Loads the raw ECG signals, reads the labels, applies filtering, downsamples the signals, normalizes them, splits them into 6-second segments, and saves the processed `.npy` files.

`dataset.py`

Loads the processed `.npy` files and returns PyTorch samples containing `signal`, `label`, and `subject_id`.

`model.py`

Defines the CNN + BiGRU + Attention model. The model takes input with shape `(batch_size, 12, 1500)` and outputs 9 logits.

`training.py`

Handles the training loop, loss function, evaluation metrics, 10-fold group splitting, checkpoint saving, and result saving.

`utils.py`

Contains helper functions for creating folders, getting subject IDs, making evaluation splits, and checking for subject leakage.

`main.py`

Runs the full pipeline. It creates folders, runs preprocessing if processed files are missing, then starts training.

## Model

The model uses:

```text
5 x 1D convolution layers
Batch normalization
ReLU
Max pooling
Bidirectional GRU
Attention
Dropout
Fully connected output layer
```

The model returns raw logits. Sigmoid is applied during evaluation, not inside the model.

## Training Setup

Current settings:

```text
Loss: BCEWithLogitsLoss
Evaluation protocol: 10-fold Group K-Fold
Epochs: 100
Batch size: 32
Metrics: macro F1, micro F1, macro ROC-AUC, per-class F1
```

Group K-Fold is used so that all segments from the same ECG recording stay in the same fold.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

On the first run, preprocessing creates the `.npy` files. On later runs, if processed files already exist, preprocessing is skipped and training starts directly.

## Outputs

Training outputs are saved in:

```text
outputs/models/
outputs/results/
```

The results folder contains:

```text
results.json
summary.csv
```

`results.json` stores the detailed fold and epoch history.

`summary.csv` stores the final metrics for each fold and can be opened in Excel.
