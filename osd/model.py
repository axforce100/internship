import torch
import torch.nn as nn

# class CRNN(nn.Module):
#     def __init__(self, input_dim=40, hidden_dim=64, rnn_layers=2, dropout=0.3):
#         super(CRNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=rnn_layers,
#                            batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)       # (B, F, T)
#         x = self.conv(x)
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)       # (B, T, F)
#         x, _ = self.rnn(x)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x.squeeze(-1)         # return raw logits (B, T)

import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, rnn_layers=2, dropout_rate = 0.3):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.rnn = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=rnn_layers,
                           batch_first=True, bidirectional=True,dropout=dropout_rate)
        self.fc = nn.Linear(128 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.squeeze(-1) 
        # return self.sigmoid(x).squeeze(-1)


# class CRNN(nn.Module):
#     def __init__(self, input_dim=40, hidden_dim=128, rnn_layers=2):
#         super(CRNN, self).__init__()
        
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),

#             nn.Conv1d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.3)
#         )
        
#         self.rnn = nn.LSTM(
#             input_size=128,
#             hidden_size=hidden_dim,
#             num_layers=rnn_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.3
#         )
        
#         self.fc = nn.Linear(hidden_dim * 2, 1)  # Output a single logit for binary classification
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         """
#         Input x shape: [batch, time, features]
#         """
#         x = x.permute(0, 2, 1)  # [batch, features, time]
#         x = self.conv(x)        # [batch, features_out, time//2]
#         x = x.permute(0, 2, 1)  # [batch, time//2, features_out]

#         x, _ = self.rnn(x)      # [batch, time//2, hidden_dim*2]
#         x = self.fc(x)          # [batch, time//2, 1]
        
#         return self.sigmoid(x).squeeze(-1)  # [batch, time//2]
    




    # class CRNN(nn.Module):
#     def __init__(self, input_dim=40, hidden_dim=64, rnn_layers=2, num_classes=2):
#         super(CRNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.Conv1d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#         )
#         self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=rnn_layers,
#                            batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Output logits for each class

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.conv(x)
#         x = x.permute(0, 2, 1)
#         x, _ = self.rnn(x)
#         x = self.fc(x)  # (batch, frames, classes)
#         return x