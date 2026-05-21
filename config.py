# config.py

def get_config():
    config = {
        "paths": {
            "raw_data": "./data/raw/",
            "processed_data": "./data/processed/",
            "outputs": "./outputs/",
            "checkpoints": "./outputs/checkpoints/",
            "results": "./outputs/results/",
            "plots": "./outputs/plots/"
        },
        "raw_dataset": {
            "kaggle_dataset": "physionet/china-physiological-signal-challenge-in-2018",
            "num_subjects": 6877,
            "sampling_rate": 500
        },

        "dataset": {
            "name": "china_physiological_signal_challenge_2018",
            "input_channels": 12,
            "segment_length": 1000,
            "num_classes": 9,
        },

        "preprocessing": {
            "download_if_missing": True,
            "source_sampling_rate": 500,
            "target_sampling_rate": 250,
            "butterworth_lowcut": 0.5,
            "butterworth_highcut": 40.0,
            "butterworth_order": 4,
            "notch_frequency": 50.0,
            "notch_quality_factor": 30.0,
            "segment_duration_seconds": 6,
            "segment_stride_seconds": 6,
            "label_mode": "snomed_multilabel",
            "max_subjects": None
        },

        "training": {
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 1e-3,
            "device": "cuda"  # change to "cpu" if needed
        },

        "evaluation": {
            "protocol": "loso",  # "loso" or "kfold"
            "num_folds": 5
        }
    }

    target_sampling_rate = config["preprocessing"]["target_sampling_rate"]
    segment_duration = config["preprocessing"]["segment_duration_seconds"]
    config["dataset"]["segment_length"] = int(round(segment_duration * target_sampling_rate))

    return config
