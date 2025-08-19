import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Parameters
TARGET_DURATION = 3.0  # seconds

# Load audio files
audio1, sr1 = librosa.load("sep_v2/data/train7/000000-target.wav", sr=None)
audio2, sr2 = librosa.load("sep_v2/data/train7/000001-target.wav", sr=None)

# Ensure same sample rate
if sr1 != sr2:
    raise ValueError("Sample rates must match.")
sr = sr1

# Target length in samples
target_len = int(TARGET_DURATION * sr)

def pad_or_trim(audio, target_len):
    if len(audio) > target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')

# Adjust both to 3 seconds
audio1 = pad_or_trim(audio1, target_len)
audio2 = pad_or_trim(audio2, target_len)

# Time axis for plotting
time_axis = np.linspace(0, TARGET_DURATION, target_len)

# Plot
plt.figure(figsize=(8, 4))

# Audio 1 in red, Audio 2 in blue
line1, = plt.plot(time_axis, audio1, color='red', alpha=0.9, label='Target Speech', linewidth=1.5)
line2, = plt.plot(time_axis, audio2, color='blue', alpha=0.9, label='Interference Speech', linewidth=1.5)

plt.title("Audio for Target Speech Extraction", fontsize=16)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Legend with thicker lines
legend = plt.legend(loc='upper right', prop={'size': 12})
for legline in legend.legend_handles:
    legline.set_linewidth(10)  # Thicker lines in legend

plt.grid(False)
plt.tight_layout()
plt.show()


# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# TARGET_DURATION = 3.0  # seconds

# # Load target audio file
# audio, sr = librosa.load("sep_v2/data/train7/000000-target.wav", sr=None)

# # Target length in samples
# target_len = int(TARGET_DURATION * sr)

# # Pad or trim function
# def pad_or_trim(audio, target_len):
#     if len(audio) > target_len:
#         return audio[:target_len]
#     else:
#         return np.pad(audio, (0, target_len - len(audio)), mode='constant')

# # Adjust audio to 3 seconds
# audio = pad_or_trim(audio, target_len)

# # Time axis for plotting
# time_axis = np.linspace(0, TARGET_DURATION, target_len)

# # Plot
# plt.figure(figsize=(8, 4))
# plt.plot(time_axis, audio, color='red', alpha=0.9, label='Target Speech', linewidth=1.5)
# plt.title("Target Speech", fontsize=16)
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.legend(loc='upper right', prop={'size': 12})
# plt.grid(False)
# plt.tight_layout()
# plt.show()