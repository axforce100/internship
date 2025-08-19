""" Import necessary libraries """
import os
import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
import librosa
from scipy.ndimage import median_filter
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference import EncoderClassifier
from osd.model import CRNN 

""" Device setup """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Variables """
SR = 16000  # Sample rate for audio processing
FRAME_SIZE = 0.5  # Frame size in seconds
HOP_SIZE = 0.25  # Hop size in seconds
osd_model_path = 'Models/model_osd1.pth'


""" Load and preprocess audio """
def preprocess_audio(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.to(device)

    # Convert to mono and resample
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SR:
        resample_transform = T.Resample(orig_freq=sr, new_freq=SR).to(device)
        waveform = resample_transform(waveform)
        sr = SR

    return waveform, sr

def load_vad_model(device = None):
    model, utils = torch.hub.load(
        './Models/silero-vad',          # Path to local repo
        'silero_vad',            # Model name (defined in hubconf.py)
        source='local'           # Critical for local loading
    )
    model = model.to(device)  # Move model to GPU
    return model, utils

def load_ecapa_model(device):
    ecapa = EncoderClassifier.from_hparams(
        source="./Models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )
    return ecapa

""" Load overlapping speech detection model """
def load_overlap_model(model_path=osd_model_path):
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def detect_speech_regions(waveform, model, utils, sr = SR, 
                         threshold=0.5, speech_ms=250, silence_ms=250):
    get_speech_timestamps = utils[0]

    return get_speech_timestamps(
        waveform.squeeze(),
        model,
        sampling_rate=sr,
        threshold=threshold,  # Adjust based on sensitivity needs
        min_speech_duration_ms=speech_ms,
        min_silence_duration_ms=silence_ms
    )

def extract_ecapa_embedding(segment, ecapa_model):
    with torch.no_grad():
        segment = segment.to(device)
        embedding = ecapa_model.encode_batch(segment)  # [1, 1, 192]
        return embedding.squeeze().cpu().numpy()  # [192]

def extract_logmel_for_overlap(segment, sr):
    segment_np = segment.squeeze().cpu().numpy()
    mel_spec = librosa.feature.melspectrogram(
        y=segment_np, sr=SR, n_mels=40, n_fft=512, hop_length=256)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Transpose to [time_steps, features] → [time, 40]
    log_mel_tensor = torch.tensor(log_mel.T).unsqueeze(0).float()  # [1, time, 40]
    return log_mel_tensor

""" Split speech segment into fixed-length frames """
def split_into_frames(waveform, start, end, sr, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, pad_final_frame=True):
    frame_length = int(frame_size * sr)
    hop_length = int(hop_size * sr)

    frames, timestamps = [], []

    i = start
    while i <= end - frame_length:
        frame = waveform[:, i:i + frame_length]
        frames.append(frame)
        timestamps.append((i, i + frame_length))
        i += hop_length

    # Handle final partial frame
    if pad_final_frame and i < end:
        remaining = waveform[:, i:end]
        valid_len = remaining.shape[1]
        pad_width = frame_length - valid_len

        padded = torch.nn.functional.pad(remaining, (0, pad_width))  # Right pad with zeros
        frames.append(padded)
        # Ensure timestamp doesn't exceed true end
        timestamps.append((i, min(i + frame_length, end)))

    return frames, timestamps

def cluster_speakers(embeddings_scaled, num_speakers, affinity='nearest_neighbors', n_neighbors=10):
    # Spectral Clustering
    clustering = SpectralClustering(
        n_clusters=num_speakers,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans',  # or 'discretize'
        random_state=42
    ).fit(embeddings_scaled)

    return clustering.labels_

# """ Main diarization pipeline """
if __name__ == "__main__":
    # global waveform, sr, frame_times, smoothened_labels, osd_labels
    wav_path = 'data/testing_audio/tests_data_trn09.wav'
    waveform, sr = preprocess_audio(wav_path)

    vad_model, vad_utils = load_vad_model(device)
    ecapa_model = load_ecapa_model(device)

    osd_model = load_overlap_model()

    # Assume `speech_timestamps` comes from your VAD
    speech_timestamps = detect_speech_regions(
        waveform = waveform.to(device),
        model = vad_model,
        utils = vad_utils)

    all_embeddings = []
    frame_times = []
    osd_threshold = 0.5

    osd_probs = []
    osd_labels = []

    # Process each speech segment detected by VAD
    for segment in speech_timestamps:
        start_sample, end_sample = segment["start"], segment["end"]
        print(f"Processing segment: {start_sample} - {end_sample}")
        frames, timestamps = split_into_frames(waveform, start_sample, end_sample, sr, frame_size = FRAME_SIZE, hop_size = HOP_SIZE)
        print(len(timestamps))

        for i, frame in enumerate(frames):
            # --- Speaker Embedding ---
            embedding = extract_ecapa_embedding(frame, ecapa_model)
            all_embeddings.append(embedding.flatten())
            frame_times.append(timestamps[i])

            # --- Overlapping Speech Detection ---
            mel_spec = extract_logmel_for_overlap(frame, sr)
            mel_spec = mel_spec.to(device)
            # print(mel_spec.shape)

            with torch.no_grad():
                osd_output = osd_model(mel_spec)
                # osd_prob = torch.sigmoid(osd_output).item()
                osd_prob = osd_output.mean().item()  # Averaged confidence for this frame
                osd_pred = int(osd_prob >= osd_threshold)

                osd_probs.append(osd_prob)
                osd_labels.append(osd_pred)

    
    all_embeddings = np.vstack(all_embeddings)  # Convert to NumPy array

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings) # Standardize features (important for spectral clustering)

    # Cluster frames (assume 2 speakers)
    speaker_labels = cluster_speakers(
        embeddings_scaled,
        num_speakers=2,
        affinity='nearest_neighbors',
        n_neighbors=10
        )
    
    smoothened_labels = median_filter(speaker_labels, size=1)
    print(f"Speaker labels: {smoothened_labels}")
    print(len(smoothened_labels))

    print("\nVoice Activity Detection (VAD) Segments:")
    print("=" * 50)
    for seg in speech_timestamps:
        start_sec = seg["start"] / sr
        end_sec = seg["end"] / sr
        print(f"{start_sec:.2f}s - {end_sec:.2f}s")

    # Print diarization results
    print("\nRefined Speaker Diarization Results (with OSD):")
    print("=" * 70)
    prev_label = smoothened_labels[0]
    prev_osd = osd_labels[0]
    prev_prob = osd_probs[0]
    start_time = frame_times[0][0]

    for i in range(1, len(smoothened_labels)):
        current_label = smoothened_labels[i]
        current_osd = osd_labels[i]
        current_prob = osd_probs[i]
        current_start = frame_times[i][0]

        if current_label != prev_label or current_osd != prev_osd:
            end_time = current_start
            print(f"{start_time / sr:.2f}s - {end_time / sr:.2f}s | Speaker {prev_label} | Overlap: {bool(prev_osd)} | Prob: {prev_prob:.2f}")
            start_time = current_start
            prev_label = current_label
            prev_osd = current_osd
            prev_prob = current_prob

    end_time = frame_times[-1][1]
    print(f"{start_time / sr:.2f}s - {end_time / sr:.2f}s | Speaker {prev_label} | Overlap: {bool(prev_osd)} | Prob: {prev_prob:.2f}")