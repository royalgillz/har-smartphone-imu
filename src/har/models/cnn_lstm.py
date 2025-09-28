import torch
import torch.nn as nn

from .. import config


class CNNLSTM(nn.Module):
    def __init__(self, input_features=config.NUM_FEATURES, num_classes=17):
        super(CNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # → (batch, features, seq)
        x = self.cnn(x)         # → (batch, channels, seq)
        x = x.permute(0, 2, 1)  # → (batch, seq, channels)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]             # last hidden state
        x = self.fc(x)
        return x
