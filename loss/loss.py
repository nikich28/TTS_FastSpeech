from torch import nn


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.spect_mse = nn.MSELoss()
        self.duration_mse = nn.MSELoss()

    def forward(self, spect, spect_preds, duration, duration_preds):
        min_len_dur = min(duration.size(-1), duration_preds.size(-1))
        min_len_spec = min(spect.size(-1), spect_preds.size(-1))

        return (self.spect_mse(spect[:, :, :min_len_spec], spect_preds[:, :, :min_len_spec]),
                self.duration_mse(duration[:, :, :min_len_dur], duration_preds[:, :, :min_len_dur])
                )
