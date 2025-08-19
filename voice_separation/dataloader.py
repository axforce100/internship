# import os
# from torch.utils.data import Dataset, DataLoader
# import torch
# import torchaudio

# class VoiceFilterDataset(Dataset):
#     def __init__(self, root_dir):
#         """
#         Args:
#             root_dir (str): root path to dataset folder containing
#                             subfolders: dvec/, target_wav/, target_mag/,
#                                          mixed_wav/, mixed_mag/
#         """
#         self.root_dir = root_dir
#         self.mixed_mag_dir = os.path.join(root_dir, "mixed_mag")
#         self.target_mag_dir = os.path.join(root_dir, "target_mag")
#         self.mixed_wav_dir = os.path.join(root_dir, "mixed_wav")
#         self.target_wav_dir = os.path.join(root_dir, "target_wav")
#         self.dvec_dir = os.path.join(root_dir, "dvec")

#         # Use mixed_mag filenames as the reference keys (without extension)
#         self.file_ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(self.mixed_mag_dir)) if f.endswith('.pt')]

#     def __len__(self):
#         return len(self.file_ids)

#     def __getitem__(self, idx):
#         file_id = self.file_ids[idx]

#         # Load mixed_mag spectrogram tensor (.pt)
#         mixed_mag_path = os.path.join(self.mixed_mag_dir, file_id + ".pt")
#         mixed_mag = torch.load(mixed_mag_path)

#         # Load target_mag spectrogram tensor (.pt)
#         target_mag_path = os.path.join(self.target_mag_dir, file_id + ".pt")
#         target_mag = torch.load(target_mag_path)

#         # Load mixed_wav waveform tensor (.wav)
#         mixed_wav_path = os.path.join(self.mixed_wav_dir, file_id + ".wav")
#         mixed_wav, sr1 = torchaudio.load(mixed_wav_path)
#         mixed_wav = mixed_wav.squeeze(0)  # mono

#         # Load target_wav waveform tensor (.wav)
#         target_wav_path = os.path.join(self.target_wav_dir, file_id + ".wav")
#         target_wav, sr2 = torchaudio.load(target_wav_path)
#         target_wav = target_wav.squeeze(0)  # mono

#         # Load dvec reference path string from .txt file
#         dvec_path_file = os.path.join(self.dvec_dir, file_id + ".txt")
#         with open(dvec_path_file, 'r') as f:
#             dvec_ref_path = f.readline().strip()

#         # Return dict matching trainer expected keys
#         return {
#             'mixed_mag': mixed_mag,
#             'target_mag': target_mag,
#             'mixed_wav': mixed_wav,
#             'target_wav': target_wav,
#             'dvec': dvec_ref_path
#         }


import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio

def train_collate_fn(batch):
    dvec_list = list()
    target_mag_list = list()
    mixed_mag_list = list()

    for dvec_mel, target_mag, mixed_mag in batch:
        dvec_list.append(dvec_mel)
        target_mag_list.append(target_mag)
        mixed_mag_list.append(mixed_mag)
    target_mag_list = torch.stack(target_mag_list, dim=0)
    mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

    return dvec_list, target_mag_list, mixed_mag_list

def test_collate_fn(batch):
    return batch
    
def create_dataloader(hp, args, train):
    if train:
        return DataLoader(dataset=VFDataset(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFDataset(hp, args, False),
                          collate_fn=test_collate_fn,
                          batch_size=hp.test.batch_size, shuffle=False, num_workers=hp.train.num_workers)


class VFDataset(Dataset):
    def __init__(self, hp, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        self.dvec_list = find_all(hp.form.dvec)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        self.target_mag_list = find_all(hp.form.target.mag)
        self.mixed_mag_list = find_all(hp.form.mixed.mag)

        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()

        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train: # need to be fast
            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])
            return dvec_mel, target_mag, mixed_mag
        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr = self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr = self.hp.audio.sample_rate)
            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        wav, _ = librosa.load(path, sr = self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase

