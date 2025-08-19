CRNN Overlapping Speech Detection Pipeline
==========================================

Files:
- dataset.py: Loads the synthetic audio/label pairs into a PyTorch dataset.
- model.py: Defines the CRNN model.
- features.py: Audio feature extraction (log-mel spectrograms).
- train.py: Training loop for CRNN.
- osd_data_gen.py: Data Generator

Output Data Format:
- data/synthetic_split/audio/*.wav
- data/synthetic_split/labels/*.npy

You can run train.py to begin training:
$ python train.py