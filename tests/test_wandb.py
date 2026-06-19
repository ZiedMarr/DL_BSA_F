import wandb

# Initialize a run
wandb.init(project="ecg-classification", name="test-run-2")

# Log some fake metrics
for epoch in range(10):
    wandb.log({
        "epoch": epoch,
        "train_loss": 1.0 - epoch * 0.1,
        "val_loss": 1.2 - epoch * 0.08,
        "Macro-F1" : 2* epoch
    })

print("Done!")
wandb.finish()