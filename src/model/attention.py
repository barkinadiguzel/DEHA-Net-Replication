import torch
import torch.nn as nn
import torch.nn.functional as F

class HardAttentionGate(nn.Module):
    def __init__(self, g_channels, l_channels):
        super().__init__()

        self.Wg = nn.Conv2d(g_channels, l_channels, kernel_size=1)
        self.Wl = nn.Conv2d(l_channels, l_channels, kernel_size=1)
        self.psi = nn.Conv2d(l_channels, 1, kernel_size=1)

    def forward(self, g, l):
        g_proj = self.Wg(g)
        l_proj = self.Wl(l)

        x = F.relu(g_proj + l_proj)
        psi = self.psi(x)
        alpha = torch.sigmoid(psi)

        return l * alpha
