import torch
from torch import nn
from DurationPredictor import DurationPredictor


class LengthRegulator(nn.Module):
    def __init__(self, input_size, duration_kernel_size, dropout=0.1, alpha=1.0):
        super().__init__()
        self.dp = DurationPredictor(input_size, duration_kernel_size, dropout)
        self.alpha = alpha

    def forward(self, x, target_duration=None):
        predicted_duration = self.dp(x)

        # check if we have target (is it training or not)
        if target_duration is None:
            duration = torch.exp(predicted_duration).cpu()
        else:
            duration = target_duration.cpu()
        duration = torch.round(duration * self.alpha).int()

        lens = duration.sum(dim=-1)
        max_len = int(lens.max())
        mask = torch.zeros((x.size(0), x.size(1), max_len))
        for i in range(mask.size(0)):
            tmp = 0
            for j in range(mask.size(1)):
                step = duration[i, j]
                mask[i, j, tmp: tmp + step] = 1
                tmp += step
        x = (x.transpose(1, 2) @ mask).transpose(1, 2)
        return x, predicted_duration
