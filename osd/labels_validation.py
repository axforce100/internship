import numpy as np
import matplotlib.pyplot as plt

label_path = 'data/synthetic_split/train/labels/sample_0.npy'
label = np.load(label_path)

print(f"Label shape: {label.shape}")
print(f"Unique values: {np.unique(label)}")

# Optional: visualize label over time
plt.figure(figsize=(10, 2))
plt.plot(label)
plt.title("Label Timeline (0: single speaker, 1: overlap)")
plt.xlabel("Time (samples)")
plt.ylabel("Label")
plt.show()