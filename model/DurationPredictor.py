import torch.nn as nn


class DurationPredictor(nn.Module):
    def __init__(self, input_size, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, padding='same')
        self.additional1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_size),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Conv1d(input_size, input_size, kernel_size, padding='same')
        self.additional2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_size),
            nn.Dropout(dropout),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.additional1(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.additional2(x)
        return x.squeeze(-1)
