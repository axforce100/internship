import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
# 20 * np.log10(mag.cpu().numpy().T + 1e-8)
def plot_spectrogram(mag, title):
    mag = mag.cpu().numpy().T  # Transpose to match (time, freq) format
    plt.figure(figsize=(10, 4))
    plt.imshow(mag, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_waveform(wav, title):
    plt.figure(figsize=(10, 2))
    plt.plot(wav.cpu().numpy())
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_dataset_samples(val_loader, device, num_samples=3):
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        mixed_mag = batch['mixed_mag'].to(device)
        target_mag = batch['target_mag'].to(device)
        mixed_wav = batch['mixed_wav'].to(device)
        target_wav = batch['target_wav'].to(device)

        print(f"Sample {i+1}:")

        # Plot first example in batch
        plot_spectrogram(mixed_mag[0], title="Mixed Magnitude Spectrogram")
        plot_spectrogram(target_mag[0], title="Target Magnitude Spectrogram")
        plot_waveform(mixed_wav[0], title="Mixed Waveform")
        plot_waveform(target_wav[0], title="Target Waveform")

if __name__ == "__main__":
    # --- Setup your dataset and loader here ---
    from dataloader import VoiceFilterDataset  # Replace with your actual dataset class/module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VoiceFilterDataset('sep_v2/data/train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)

    visualize_dataset_samples(train_loader, device, num_samples=3)