# import os
# import random
# import librosa
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm
# from collections import defaultdict

# # ---------------- Config ----------------
# TRAIN_SPEAKER_DIR = 'LibriSpeech/train/train-clean-100'
# VAL_SPEAKER_DIR = 'LibriSpeech/test/test-clean'
# OUTPUT_DIR = 'data/synthetic_split'
# SAMPLE_RATE = 16000
# SEGMENT_DURATION = 2.0  # seconds
# SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
# TOTAL_TRAIN_SAMPLES = 80000
# TOTAL_VAL_SAMPLES = 20000
# OVERLAP_RATIO = 0.6
# SINGLE_RATIO = 0.4  # Adjusted ratio to fill 100%
# # ---------------------------------------


# for split in ['train', 'val']:
#     os.makedirs(f"{OUTPUT_DIR}/{split}/audio", exist_ok=True)
#     os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# def collect_speaker_files(speaker_dir):
#     speaker_files = defaultdict(list)
#     for speaker_id in os.listdir(speaker_dir):
#         speaker_path = os.path.join(speaker_dir, speaker_id)
#         if os.path.isdir(speaker_path):
#             for chapter_dir in os.listdir(speaker_path):
#                 chapter_path = os.path.join(speaker_path, chapter_dir)
#                 if os.path.isdir(chapter_path):
#                     for fname in os.listdir(chapter_path):
#                         if fname.lower().endswith(('.wav', '.flac')):
#                             speaker_files[speaker_id].append(os.path.join(chapter_path, fname))
#     return speaker_files

# train_speaker_files_dict = collect_speaker_files(TRAIN_SPEAKER_DIR)
# val_speaker_files_dict = collect_speaker_files(VAL_SPEAKER_DIR)

# train_speaker_files = [file for files in train_speaker_files_dict.values() for file in files]
# val_speaker_files = [file for files in val_speaker_files_dict.values() for file in files]

# print(f"Train files: {len(train_speaker_files)} from {len(train_speaker_files_dict)} speakers")
# print(f"Val files: {len(val_speaker_files)} from {len(val_speaker_files_dict)} speakers")

# def load_random_clip(file_path):
#     audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
#     if len(audio) < SEGMENT_SAMPLES:
#         return None
#     start = random.randint(0, len(audio) - SEGMENT_SAMPLES)
#     return audio[start:start + SEGMENT_SAMPLES]

# def create_single_speaker_segment(file_list):
#     f = random.choice(file_list)
#     audio = load_random_clip(f)
#     if audio is not None:
#         label = np.zeros(SEGMENT_SAMPLES, dtype=np.int16)  # 1 = single speaker
#         return audio, label
#     return None, None

# def create_overlapping_segment(file_list):
#     f1, f2 = random.sample(file_list, 2)
#     audio1 = load_random_clip(f1)
#     audio2 = load_random_clip(f2)
#     if audio1 is None or audio2 is None:
#         return None, None

#     overlap_start = random.randint(0, SEGMENT_SAMPLES // 2)
#     overlap_end = min(SEGMENT_SAMPLES, overlap_start + len(audio2))
#     length = overlap_end - overlap_start

#     mixed = np.copy(audio1)
#     mixed[overlap_start:overlap_end] += audio2[:length]
#     mixed = mixed / np.max(np.abs(mixed) + 1e-6)

#     label = np.zeros(SEGMENT_SAMPLES, dtype=np.int16)
#     label[overlap_start:overlap_end] = 1  # 2 = overlapping speech
#     return mixed, label

# def save_sample(audio, label, sample_id, split):
#     sf.write(f"{OUTPUT_DIR}/{split}/audio/sample_{sample_id}.wav", audio, SAMPLE_RATE)
#     np.save(f"{OUTPUT_DIR}/{split}/labels/sample_{sample_id}.npy", label)

# def generate_samples(file_list, split, total_samples):
#     num_single = int(total_samples * SINGLE_RATIO)
#     num_overlap = int(total_samples * OVERLAP_RATIO)

#     task_types = (["single"] * num_single + ["overlap"] * num_overlap)
#     random.shuffle(task_types)

#     print(f"\n🚀 Generating {split} set ({total_samples} samples)...")
#     sample_id = 0
#     for task in tqdm(task_types, desc=f"{split.capitalize()} Progress"):
#         if task == "single":
#             audio, label = create_single_speaker_segment(file_list)
#         elif task == "overlap":
#             audio, label = create_overlapping_segment(file_list)
#         else:
#             continue

#         if audio is not None:
#             save_sample(audio, label, sample_id, split)
#             sample_id += 1

#     print(f"✅ {split} set generation complete.\n")

# # Generate both sets
# generate_samples(train_speaker_files, 'train', TOTAL_TRAIN_SAMPLES)
# generate_samples(val_speaker_files, 'val', TOTAL_VAL_SAMPLES)


import os
import random
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

# ---------------- Config ----------------
TRAIN_SPEAKER_DIR = 'LibriSpeech/train/train-clean-100'
VAL_SPEAKER_DIR = 'LibriSpeech/test/test-clean'
OUTPUT_DIR = 'data/synthetic_split'
SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0  # seconds
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
TOTAL_TRAIN_SAMPLES = 80000
TOTAL_VAL_SAMPLES = 20000
OVERLAP_RATIO = 0.6
SINGLE_RATIO = 0.4
# ----------------------------------------

for split in ['train', 'val']:
    os.makedirs(f"{OUTPUT_DIR}/{split}/audio", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

def collect_speaker_files(speaker_dir):
    speaker_files = defaultdict(list)
    for speaker_id in os.listdir(speaker_dir):
        speaker_path = os.path.join(speaker_dir, speaker_id)
        if os.path.isdir(speaker_path):
            for chapter_dir in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if os.path.isdir(chapter_path):
                    for fname in os.listdir(chapter_path):
                        if fname.lower().endswith(('.wav', '.flac')):
                            speaker_files[speaker_id].append(os.path.join(chapter_path, fname))
    return speaker_files

train_speaker_files_dict = collect_speaker_files(TRAIN_SPEAKER_DIR)
val_speaker_files_dict = collect_speaker_files(VAL_SPEAKER_DIR)

print(f"Train files: {sum(len(v) for v in train_speaker_files_dict.values())} from {len(train_speaker_files_dict)} speakers")
print(f"Val files: {sum(len(v) for v in val_speaker_files_dict.values())} from {len(val_speaker_files_dict)} speakers")

def load_random_clip(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) < SEGMENT_SAMPLES:
        return None
    start = random.randint(0, len(audio) - SEGMENT_SAMPLES)
    return audio[start:start + SEGMENT_SAMPLES]

def create_single_speaker_segment(speaker_files):
    all_files = [file for files in speaker_files.values() for file in files]
    f = random.choice(all_files)
    audio = load_random_clip(f)
    if audio is not None:
        label = np.zeros(SEGMENT_SAMPLES, dtype=np.int16)  # 0 = single speaker
        return audio, label
    return None, None

def create_overlapping_segment(speaker_files):
    speakers = list(speaker_files.keys())
    speaker1, speaker2 = random.sample(speakers, 2)  # Make sure 2 different speakers
    f1 = random.choice(speaker_files[speaker1])
    f2 = random.choice(speaker_files[speaker2])

    audio1 = load_random_clip(f1)
    audio2 = load_random_clip(f2)
    if audio1 is None or audio2 is None:
        return None, None

    overlap_start = random.randint(0, SEGMENT_SAMPLES // 2)
    overlap_end = min(SEGMENT_SAMPLES, overlap_start + len(audio2))
    length = overlap_end - overlap_start

    mixed = np.copy(audio1)
    mixed[overlap_start:overlap_end] += audio2[:length]
    mixed = mixed / np.max(np.abs(mixed) + 1e-6)

    label = np.zeros(SEGMENT_SAMPLES, dtype=np.int16)
    label[overlap_start:overlap_end] = 1  # 1 = overlapping speech
    return mixed, label

def save_sample(audio, label, sample_id, split):
    sf.write(f"{OUTPUT_DIR}/{split}/audio/sample_{sample_id}.wav", audio, SAMPLE_RATE)
    np.save(f"{OUTPUT_DIR}/{split}/labels/sample_{sample_id}.npy", label)

def generate_samples(speaker_files, split, total_samples):
    num_single = int(total_samples * SINGLE_RATIO)
    num_overlap = int(total_samples * OVERLAP_RATIO)

    task_types = (["single"] * num_single + ["overlap"] * num_overlap)
    random.shuffle(task_types)

    print(f"\n🚀 Generating {split} set ({total_samples} samples)...")
    sample_id = 0
    pbar = tqdm(total=total_samples, desc=f"{split.capitalize()} Progress")

    while sample_id < total_samples:
        if not task_types:
            # If somehow tasks are exhausted, reshuffle and refill
            task_types = (["single"] * num_single + ["overlap"] * num_overlap)
            random.shuffle(task_types)

        task = task_types.pop()

        if task == "single":
            audio, label = create_single_speaker_segment(speaker_files)
        elif task == "overlap":
            audio, label = create_overlapping_segment(speaker_files)
        else:
            continue

        if audio is not None:
            save_sample(audio, label, sample_id, split)
            sample_id += 1
            pbar.update(1)

    pbar.close()
    print(f"✅ {split} set generation complete.\n")

# Generate datasets
generate_samples(train_speaker_files_dict, 'train', TOTAL_TRAIN_SAMPLES)
generate_samples(val_speaker_files_dict, 'val', TOTAL_VAL_SAMPLES)

