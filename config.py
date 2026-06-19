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
            "epochs": 1,
            "learning_rate": 1e-3,
            "device": "cuda"  # change to "cpu" if needed
        },

        "evaluation": {
            "protocol": "group_kfold",
            "num_folds": 10
        }
    }

    return config
