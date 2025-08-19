import os
import json
import numpy as np
from tqdm import tqdm

# Update with your config
OUTPUT_DIR = 'data/synthetic_split'
METADATA_FILE = os.path.join(OUTPUT_DIR, 'metadata.json')

def determine_label_type(label_path):
    label = np.load(label_path)
    return 'overlap' if np.any(label == 1) else 'single'

metadata = []

for split in ['train', 'val']:
    audio_dir = os.path.join(OUTPUT_DIR, split, 'audio')
    label_dir = os.path.join(OUTPUT_DIR, split, 'labels')

    for fname in tqdm(os.listdir(audio_dir), desc=f"Processing {split} set"):
        if not fname.endswith('.wav'):
            continue

        sample_id = fname.replace('.wav', '')
        audio_path = os.path.join(split, 'audio', fname)
        label_path = os.path.join(split, 'labels', f"{sample_id}.npy")
        label_type = determine_label_type(os.path.join(OUTPUT_DIR, label_path))

        metadata.append({
            'id': sample_id,
            'path': audio_path,
            'label_path': label_path,
            'label_type': label_type,
            'split': split
        })

# Save as JSON
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved to {METADATA_FILE} ({len(metadata)} samples)")
