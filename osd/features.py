import librosa
import numpy as np

def extract_features(audio_path, sr=16000, n_mels=40,n_fft=512, hop_length=256):
    y, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft = n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.T  # shape: (frames, features)
