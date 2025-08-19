import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_saved_spectrogram(pt_path, title="Spectrogram", cmap="magma"):
    # Load spectrogram tensor from .pt file
    spec = torch.load(pt_path)

    # If it's a dict, fetch the spectrogram key
    if isinstance(spec, dict):
        if 'spec' in spec:
            spec = spec['spec']
        else:
            raise KeyError("Expected key 'spec' not found in the dict.")

    # If it's a tensor with batch/channel dim, remove it
    if isinstance(spec, torch.Tensor):
        spec = spec.squeeze()  # remove dimensions of size 1
        spec = spec.cpu().numpy()

    # Final check for shape
    if len(spec.shape) != 2:
        raise ValueError(f"Expected spectrogram shape (freq, time), got {spec.shape}")

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.show()

# Example usage
plot_saved_spectrogram("sep_v2/data/test/mixed_mag/000000.pt")
