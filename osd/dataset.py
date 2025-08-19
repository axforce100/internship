import torch
from torch.utils.data import Dataset
import os
import numpy as np
from features import extract_features
import librosa

class OverlapSpeechDataset(Dataset):
    def __init__(self, audio_dir, label_dir):
        self.audio_files = sorted(os.listdir(audio_dir))
        self.audio_dir = audio_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        label_path = os.path.join(self.label_dir, self.audio_files[idx].replace('.wav', '.npy'))

        features = extract_features(audio_path)
        label = np.load(label_path)
        label = librosa.util.frame(label, frame_length=512, hop_length=256).mean(axis=0) > 0.5
        label = label.astype(np.float32)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


