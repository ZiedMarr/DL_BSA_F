# config.py

def _deep_merge(base, overrides):
    """Recursively merge overrides into base, raising KeyError on unknown keys."""
    for key, value in overrides.items():
        if key not in base:
            raise KeyError(
                f"Override key '{key}' not found in config. "
                f"Valid keys at this level: {list(base.keys())}"
            )
        if isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def get_config(overrides=None):
    config = {
        "paths": {
            "raw_data": "./data/raw/",
            "processed_data": "./data/processed/",
            "outputs": "./outputs/",
            "checkpoints": "./outputs/models/",
            "results": "./outputs/results/",
            "plots": "./outputs/plots/"
        },
        "raw_dataset": {
            "num_subjects" : 6877,
            "sampling_rate" : 500

        },

        "dataset": {
            "name": "example",
            "input_channels": 12,
            "segment_length": 1500,
            "num_classes": 9,
            "sampling_rate" : 500
        },

        "model": {
            "name": "cnn1d"
        },

        "preprocess": {
            "powerline" : 50,
            "lowcut" : 0.5,
            "highcut" :45 ,
            "downsampled_rate" : 250,
            "overlap_ratio" : 0.5
        },

        "training": {
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 1e-3,
            "device": "cuda",  # change to "cpu" if needed
            "dropout" : 0.3,
            "seed": 42
        },

        "evaluation": {
            "protocol": "group_kfold",
            "num_folds": 10,
            "class_threshold" : 0.5
        }
    }

    if overrides is not None:
        _deep_merge(config, overrides)

    return config
