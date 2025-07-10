import torch


class CenterImage(torch.nn.Module):
    def __init__(self, channel_means: tuple, device):
        super().__init__()
        channel_means_tensor = torch.tensor(channel_means).to(device)
        self.channel_means = channel_means_tensor[None, :, None, None]

    def forward(self, x):
        return x - self.channel_means
