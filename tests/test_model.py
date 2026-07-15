
import torch
from config import get_config
from models import get_model
from transforms import stft_transform

config = get_config()
config["model"]["name"] = "cnn_vit"
config["training"]["device"] = "cpu"

# Build model
model = get_model(config)
model.eval()

# Build the same transform training.py would use
STFT_PARAMS = {"n_fft": 128, "hop_length": 32, "win_length": 128}
transform = stft_transform(**STFT_PARAMS)

# Fake a small batch of raw signals: (batch, 12, 1500)
batch_size = 2
raw_signals = torch.randn(batch_size, 12, 1500)

# Apply transform per-sample (mirrors what Dataset.__getitem__ does)
spectrograms = torch.stack([transform(sig) for sig in raw_signals])
print("Spectrogram batch shape:", spectrograms.shape)  # expect (2, 12, 65, 47)

# Forward pass
with torch.no_grad():
    out = model(spectrograms)

print("Output shape:", out.shape)   # expect (2, 9)
print("Output values:", out)
print("Any NaN:", torch.isnan(out).any().item())