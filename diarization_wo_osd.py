""" Import necessary libraries """
import os
import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
from scipy.ndimage import median_filter
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from speechbrain.inference import SpeakerRecognition

from osd.model import CRNN 

""" Device setup """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000  # Sample rate for audio processing
FRAME_SIZE = 0.5  # Frame size in seconds
HOP_SIZE = 0.2  # Hop size in seconds

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

    return waveform, 16000

def load_vad_model(device = None):
    model, utils = torch.hub.load(
        './Models/silero-vad',          # Path to local repo
        'silero_vad',            # Model name (defined in hubconf.py)
        source='local'           # Critical for local loading
    )
    model = model.to(device)  # Move model to GPU
    return model, utils

def load_ecapa_model(device):
    ecapa = SpeakerRecognition.from_hparams(
        source="./Models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )
    return ecapa

def detect_speech_regions(waveform, model, utils, sr=SR, 
                         threshold=0.5, speech_ms=200, silence_ms=200):
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
        segment = segment.squeeze(0).cpu()  # Convert to [time]
        embedding = ecapa_model.encode_batch(segment.unsqueeze(0))  # [1, 1, 192]
        return embedding.squeeze().cpu().numpy()  # [192]

""" Split speech segment into fixed-length frames """
def split_into_frames(waveform, start, end, sr, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    frame_length = int(frame_size * sr)  # Convert seconds to samples
    hop_length = int(hop_size * sr)

    frames, timestamps = [], [] # Initialize lists to store frames and timestamps

    for i in range(start, end - frame_length + 1, hop_length):
        frame = waveform[:, i:i + frame_length]
        frames.append(frame)
        timestamps.append((i, i + frame_length))

    return frames, timestamps

def is_frame_in_speech_region(start, end, speech_segments):
    for segment in speech_segments:
        if start >= segment["start"] and end <= segment["end"]:
            return True
    return False

def cluster_speakers(embeddings_scaled, num_speakers, affinity='nearest_neighbors', n_neighbors=7):
    # Spectral Clustering
    clustering = SpectralClustering(
        n_clusters=num_speakers,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans'  # or 'discretize'
    ).fit(embeddings_scaled)

    return clustering.labels_

# """ Main diarization pipeline """
if __name__ == "__main__":
    wav_path = 'data/testing_audio/tests_data_trn09.wav'
    waveform, sr = preprocess_audio(wav_path)
    
    vad_model, vad_utils = load_vad_model(device)
    ecapa_model = load_ecapa_model(device)

    # Assume `speech_timestamps` comes from your VAD
    speech_timestamps = detect_speech_regions(
        waveform = waveform.to(device),
        model = vad_model,
        utils = vad_utils)

    all_embeddings = []
    frame_labels = []
    frame_times = []

    # Process each speech segment detected by VAD
    for segment in speech_timestamps:
        start_sample, end_sample = segment["start"], segment["end"]
        frames, timestamps = split_into_frames(waveform, start_sample, end_sample, sr, frame_size= FRAME_SIZE ,hop_size= HOP_SIZE)

        for i, frame in enumerate(frames):
            embedding = extract_ecapa_embedding(frame, ecapa_model)
            all_embeddings.append(embedding.flatten())
            frame_times.append(timestamps[i])


    # Print speech timestamps in seconds
    print("\nSpeech Timestamps:")
    print("=" * 50)
    for segment in speech_timestamps:
        start_time = segment["start"] / sr
        end_time = segment["end"] / sr
        print(f"{start_time:.2f}s - {end_time:.2f}s")
    
    all_embeddings = np.vstack(all_embeddings)  # Convert to NumPy array

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings) # Standardize features (important for spectral clustering)

    # Cluster frames (assume 2 speakers)
    speaker_labels = cluster_speakers(
        embeddings_scaled,
        num_speakers=2,
        affinity='nearest_neighbors',
        n_neighbors=10,
        )
    
    # filter_size = max(3, int(1 / HOP_SIZE))  # ~1 second smoothing
    smoothened_labels = median_filter(speaker_labels, size=9)

    # Print diarization results
    print("\nRefined Speaker Diarization Results:")
    print("=" * 50)
    prev_label = smoothened_labels[0]
    start_time = frame_times[0][0]

    for i in range(1, len(smoothened_labels)):
        current_label = smoothened_labels[i]
        current_start, current_end = frame_times[i]

        # Skip if frame not in VAD-detected region
        if not is_frame_in_speech_region(current_start, current_end, speech_timestamps):
            continue

        if current_label != prev_label:
            end_time = current_start
            print(f"{start_time / sr:.2f}s - {end_time / sr:.2f}s | Speaker {prev_label}")
            start_time = current_start
            prev_label = current_label


    # for i in range(1, len(smoothened_labels)):
    #     current_label = smoothened_labels[i]
    #     current_start = frame_times[i][0]

    #     # If label changes or we're at the final frame
    #     if current_label != prev_label:
    #         end_time = current_start
    #         print(f"{start_time / sr:.2f}s - {end_time / sr:.2f}s | Speaker {prev_label}")
    #         start_time = current_start
    #         prev_label = current_label

    # Print last segment
    end_time = frame_times[-1][1]
    print(f"{start_time / sr:.2f}s - {end_time / sr:.2f}s | Speaker {prev_label}")

    # for i, (start, end) in enumerate(frame_times):
    #     start_sec, end_sec = start / sr, end / sr
    #     speaker = speaker_labels[i]
    #     print(f"{start_sec:.2f}s - {end_sec:.2f}s | Speaker {speaker}")