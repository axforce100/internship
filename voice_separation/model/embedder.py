# # embedder.py

# import torch
# import torchaudio
# from speechbrain.inference import EncoderClassifier

# class ECAPAEmbedder:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.model = EncoderClassifier.from_hparams(
#             source="speechbrain/spkrec-ecapa-voxceleb",
#             run_opts={"device": self.device}
#         )

#     def get_embedding(self, wav_path):
#         signal, fs = torchaudio.load(wav_path)
#         signal = signal.to(self.device)
#         if fs != 16000:
#             resample = torchaudio.transforms.Resample(fs, 16000).to(self.device)
#             signal = resample(signal)
#         with torch.no_grad():
#             embedding = self.model.encode_batch(signal).squeeze(0)
#         return embedding

#     def get_batch_embeddings(self, wav_paths):
#         embeddings = []
#         for path in wav_paths:
#             embeddings.append(self.get_embedding(path))
#         return torch.stack(embeddings)

import torch
import torch.nn as nn


class LinearNorm(nn.Module):
    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hp.embedder.lstm_hidden, hp.embedder.emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp.embedder.num_mels,
                            hp.embedder.lstm_hidden,
                            num_layers=hp.embedder.lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.hp.embedder.window, self.hp.embedder.stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x
