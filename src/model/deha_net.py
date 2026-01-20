import torch
import torch.nn as nn
from encoders import Encoder
from decoder import Decoder
from attention import HardAttentionGate
import torch.nn.functional as F


class DEHANet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Dual Encoder
        self.global_encoder = Encoder(in_channels)
        self.local_encoder  = Encoder(in_channels)

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Attention Gates 
        self.att4 = HardAttentionGate(256, 256)
        self.att3 = HardAttentionGate(128, 128)
        self.att2 = HardAttentionGate(64, 64)
        self.att1 = HardAttentionGate(32, 32)

        # Decoder 
        self.decoder = Decoder()

    def forward(self, img, roi):

        # Global encoder
        g_x, g_skips = self.global_encoder(img)

        # Local encoder
        l_x, l_skips = self.local_encoder(roi)

        # Bottleneck
        x = self.bottleneck(g_x + l_x)

        # Attention applied to skip connections
        s1_g, s2_g, s3_g, s4_g = g_skips
        s1_l, s2_l, s3_l, s4_l = l_skips

        a4 = self.att4(s4_g, s4_l)
        a3 = self.att3(s3_g, s3_l)
        a2 = self.att2(s2_g, s2_l)
        a1 = self.att1(s1_g, s1_l)

        # Decoder
        out = self.decoder(x,
                           skips_img=[s1_g, s2_g, s3_g, s4_g],
                           skips_roi=[a1, a2, a3, a4])

        return out
