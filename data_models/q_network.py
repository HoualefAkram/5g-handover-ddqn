import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)

    def hard_update(self, network):
        self.load_state_dict(network.state_dict())

    @staticmethod
    def from_state_dict(state_dict):
        net = QNetwork()
        net.load_state_dict(state_dict)
        return net
