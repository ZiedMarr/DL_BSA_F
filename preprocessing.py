# preprocessing.py

import os
import subprocess
import numpy as np


SNOMED_MAP = {
    "426783006": 0,  # Normal Sinus Rhythm (N)
    "164889003": 1,  # Atrial Fibrillation (AF)
    "270492004": 2,  # 1st-degree AV Block (I-AVB)
    "164909002": 3,  # Left Bundle Branch Block (LBBB)
    "59118001": 4,   # Right Bundle Branch Block (RBBB)
    "429622005": 5,  # ST-Depression (STD)
    "164931005": 6,  # ST-Elevation (STE)
    "284470004": 7,  # Premature Atrial Contraction (PAC)
    "164884008": 8   # Premature Ventricular Contraction (PVC)
}


def read_wfdb_record(file_path):
    """
    Read a WFDB record path without extension and return signal data plus
    metadata needed by the preprocessing pipeline.
    """
    try:
        import wfdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'wfdb' package is required to read this dataset. "
            "Install project dependencies first, for example with 'pip install -e .'."
        ) from exc

    record = wfdb.rdrecord(file_path)

    return {
        "signals": record.p_signal,
        "sig_len": record.sig_len,
        "sampling_rate": record.fs
    }


def _find_raw_layout(search_root):
    if not os.path.isdir(search_root):
        return None

    if os.path.basename(os.path.normpath(search_root)) == "Training_WFDB":
        hea_files = [file for file in os.listdir(search_root) if file.endswith(".hea")]

        if hea_files:
            return search_root

    for root, dirs, files in os.walk(search_root):
        if "Training_WFDB" in dirs:
            return os.path.join(root, "Training_WFDB")

    return None


def _raw_data_exists(config):
    if _find_raw_layout(config["paths"]["raw_data"]) is not None:
        return True

    if _find_raw_layout("./data") is not None:
        return True

    return False


def download_dataset(config, force=False):
    """
    Download and unzip the Kaggle dataset when the raw WFDB files are missing.

    This requires:
    - the kaggle package/CLI to be installed
    - Kaggle credentials configured locally
    """
    if _raw_data_exists(config) and not force:
        print("Raw dataset already exists. Skipping Kaggle download.")
        return

    raw_path = config["paths"]["raw_data"]
    dataset_id = config["raw_dataset"]["kaggle_dataset"]

    os.makedirs(raw_path, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        raw_path,
        "--unzip"
    ]

    print(f"Downloading Kaggle dataset '{dataset_id}' to {raw_path}...")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find the 'kaggle' command. Install/configure Kaggle first, "
            "then rerun preprocessing."
        ) from exc


def _resolve_raw_paths(config):
    for search_root in [config["paths"]["raw_data"], "./data"]:
        raw_layout = _find_raw_layout(search_root)

        if raw_layout is not None:
            return raw_layout

    raise FileNotFoundError(
        "Could not find raw WFDB data. Expected either "
        f"'{os.path.join(config['paths']['raw_data'], 'Training_WFDB')}', or the "
        "same folder somewhere under './data'."
    )


def _get_labels_from_header(header_path):
    with open(header_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("#Dx:"):
                continue

            dx_value = line.split(":", 1)[1].strip()

            if not dx_value:
                return []

            return [label.strip() for label in dx_value.split(",") if label.strip()]

    raise ValueError(f"Could not find '#Dx:' diagnosis labels in header: {header_path}")


def _normalize_label_key(label):
    return str(int(label)) if isinstance(label, (int, float, np.integer, np.floating)) else str(label)


def _labels_to_target(labels, config):
    label_mode = config.get("preprocessing", {}).get("label_mode", "snomed_multilabel")

    if label_mode == "snomed_multilabel":
        target = np.zeros(config["dataset"]["num_classes"], dtype=np.float32)

        for label in labels:
            label_key = _normalize_label_key(label)

            if label_key in SNOMED_MAP:
                target[SNOMED_MAP[label_key]] = 1.0

        if target.sum() == 0:
            raise ValueError(f"No supported SNOMED label found in labels: {labels}")

        return target

    if label_mode == "first_label":
        return int(labels[0])

    if label_mode != "binary":
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    normal_label = config.get("preprocessing", {}).get("normal_label", 1)
    has_abnormal_label = any(label != normal_label for label in labels)

    return int(has_abnormal_label)


def _repeat_target_for_segments(target, num_segments):
    target = np.asarray(target)

    if target.ndim == 0:
        return np.full((num_segments,), target, dtype=np.int64)

    return np.repeat(target[np.newaxis, :], num_segments, axis=0).astype(np.float32)


def _ensure_channels_first(signals):
    signals = np.asarray(signals, dtype=np.float32)

    if signals.ndim == 1:
        return signals[np.newaxis, :]

    if signals.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D signal array, got shape {signals.shape}")

    # WFDB returns p_signal as (time, channels). The model expects (channels, time).
    return signals.T


def _zscore_normalize(signals):
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1.0

    return (signals - mean) / std


def _butterworth_filter(signals, sampling_rate, config):
    try:
        from scipy.signal import butter, sosfiltfilt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'scipy' package is required for Butterworth filtering. "
            "Install project dependencies first, for example with 'pip install -e .'."
        ) from exc

    prep_config = config.get("preprocessing", {})
    lowcut = prep_config.get("butterworth_lowcut", 0.5)
    highcut = prep_config.get("butterworth_highcut", 40.0)
    order = prep_config.get("butterworth_order", 4)
    nyquist = sampling_rate / 2.0

    if lowcut is None and highcut is None:
        return signals

    if lowcut is not None and highcut is not None:
        btype = "bandpass"
        cutoff = [lowcut / nyquist, highcut / nyquist]
    elif lowcut is not None:
        btype = "highpass"
        cutoff = lowcut / nyquist
    else:
        btype = "lowpass"
        cutoff = highcut / nyquist

    sos = butter(order, cutoff, btype=btype, output="sos")
    return sosfiltfilt(sos, signals, axis=1).astype(np.float32)


def _notch_filter(signals, sampling_rate, config):
    try:
        from scipy.signal import filtfilt, iirnotch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'scipy' package is required for notch filtering. "
            "Install project dependencies first, for example with 'pip install -e .'."
        ) from exc

    prep_config = config.get("preprocessing", {})
    notch_frequency = prep_config.get("notch_frequency", 50.0)
    quality_factor = prep_config.get("notch_quality_factor", 30.0)

    if notch_frequency is None:
        return signals

    nyquist = sampling_rate / 2.0

    if notch_frequency >= nyquist:
        return signals

    b, a = iirnotch(notch_frequency / nyquist, quality_factor)
    return filtfilt(b, a, signals, axis=1).astype(np.float32)


def _downsample(signals, source_sampling_rate, target_sampling_rate):
    from math import gcd
    try:
        from scipy.signal import resample_poly
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'scipy' package is required for downsampling. "
            "Install project dependencies first, for example with 'pip install -e .'."
        ) from exc

    if source_sampling_rate == target_sampling_rate:
        return signals

    divisor = gcd(int(source_sampling_rate), int(target_sampling_rate))
    up = int(target_sampling_rate // divisor)
    down = int(source_sampling_rate // divisor)

    return resample_poly(signals, up, down, axis=1).astype(np.float32)


def _segment_signal(signals, segment_length, segment_stride):
    n_channels, n_samples = signals.shape

    if n_samples < segment_length:
        padded = np.zeros((n_channels, segment_length), dtype=np.float32)
        padded[:, :n_samples] = signals
        return padded[np.newaxis, :, :]

    segments = []

    for start in range(0, n_samples - segment_length + 1, segment_stride):
        stop = start + segment_length
        segments.append(signals[:, start:stop])

    return np.stack(segments).astype(np.float32)


def _resolve_segment_samples(config, target_sampling_rate):
    prep_config = config.get("preprocessing", {})

    if "segment_duration_seconds" in prep_config:
        segment_length = int(round(prep_config["segment_duration_seconds"] * target_sampling_rate))
    else:
        segment_length = prep_config.get("segment_length", config["dataset"]["segment_length"])

    if "segment_stride_seconds" in prep_config:
        segment_stride = int(round(prep_config["segment_stride_seconds"] * target_sampling_rate))
    else:
        segment_stride = prep_config.get("segment_stride", segment_length)

    if segment_length <= 0:
        raise ValueError(f"segment_length must be positive, got {segment_length}")

    if segment_stride <= 0:
        raise ValueError(f"segment_stride must be positive, got {segment_stride}")

    return segment_length, segment_stride


def preprocessing_pipeline(signals, config, sampling_rate=None):
    prep_config = config.get("preprocessing", {})
    source_sampling_rate = sampling_rate or prep_config.get(
        "source_sampling_rate",
        config["raw_dataset"].get("sampling_rate", 500)
    )
    target_sampling_rate = prep_config.get("target_sampling_rate", source_sampling_rate)
    segment_length, segment_stride = _resolve_segment_samples(config, target_sampling_rate)

    signals = _ensure_channels_first(signals)
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    signals = _butterworth_filter(signals, source_sampling_rate, config)
    signals = _notch_filter(signals, source_sampling_rate, config)
    signals = _downsample(signals, source_sampling_rate, target_sampling_rate)
    signals = _zscore_normalize(signals)

    return _segment_signal(signals, segment_length, segment_stride)


def save_subject_file(subject_dict, save_path):
    os.makedirs(save_path, exist_ok=True)
    subject_id = subject_dict["subject_id"]
    file_path = os.path.join(save_path, f"{subject_id}.npy")

    np.save(file_path, subject_dict)


def preprocess_data(config):
    if config.get("preprocessing", {}).get("download_if_missing", True):
        download_dataset(config)

    signals_path = _resolve_raw_paths(config)
    save_path = config["paths"]["processed_data"]

    subject_files = sorted(file for file in os.listdir(signals_path) if file.endswith(".hea"))
    max_subjects = config.get("preprocessing", {}).get("max_subjects")

    if max_subjects is not None:
        subject_files = subject_files[:max_subjects]

    os.makedirs(save_path, exist_ok=True)

    print(f"Running preprocessing for {len(subject_files)} WFDB records...")

    saved_count = 0

    for file in subject_files:
        subject_id = os.path.splitext(file)[0]
        record_path = os.path.join(signals_path, subject_id)
        header_path = os.path.join(signals_path, file)

        record = read_wfdb_record(record_path)
        raw_labels = _get_labels_from_header(header_path)

        try:
            target = _labels_to_target(raw_labels, config)
        except ValueError as e:
            print(f"Skipping {subject_id}: {e}")
            continue

        processed_signals = preprocessing_pipeline(
            record["signals"],
            config,
            sampling_rate=record["sampling_rate"]
        )

        labels = _repeat_target_for_segments(target, processed_signals.shape[0])

        subject_dict = {
            "subject_id": subject_id,
            "signals": processed_signals,
            "labels": labels,
            "raw_labels": raw_labels,
            "source_sampling_rate": record["sampling_rate"],
            "target_sampling_rate": config["preprocessing"]["target_sampling_rate"]
        }

        save_subject_file(subject_dict, save_path)
        saved_count += 1

        if saved_count % 100 == 0:
            print(f"Saved {saved_count}/{len(subject_files)} subjects...")

    print(f"Preprocessing complete. Saved {saved_count} files to {save_path}.")


def preprocess_dataset(config):
    """
    Read the China Physiological Signal Challenge WFDB records, parse diagnosis
    labels from '#Dx:' lines in each .hea header, and save one processed .npy
    file per subject.

    Output format:
        {
            "signals": (N, C, T),
            "labels": (N, num_classes) for SNOMED multi-label mode,
            "subject_id": str,
            "raw_labels": list[int]
        }
    """
    preprocess_data(config)
