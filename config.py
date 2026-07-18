# config.py

def get_config():
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
            "epochs": 100,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "scheduler": "none",
            "warmup_epochs": 0,
            "device": "cuda",  # change to "cpu" if needed
            "class_weights": False,
            "class_weight_mode": "full",
            "weight_decay": 0,
            "threshold_tuning": False,
            "experiment_name": None
        },

        "evaluation": {
            "protocol": "group_kfold",
            "num_folds": 10
        }
    }

    return config
