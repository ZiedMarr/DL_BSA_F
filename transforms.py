# transforms.py

import torch


def stft_transform(n_fft, hop_length, win_length):
    window = torch.hann_window(win_length)
    eps = 1e-8

    def transform(signal):
        magnitudes = [
            torch.stft(
                lead,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            ).abs()
            for lead in signal
        ]
        return torch.log(torch.stack(magnitudes, dim=0) + eps)

    return transform
