import torch
import torch.nn as nn

from .blocks import EncoderBlock, DecoderBlock, HardAttentionGate

class DEHANet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        # Global Encoder
        self.enc1_g = EncoderBlock(in_channels, 64)
        self.enc2_g = EncoderBlock(64, 128)
        self.enc3_g = EncoderBlock(128, 256)

        # Local Encoder (ROI branch)
        self.enc1_l = EncoderBlock(in_channels, 64)
        self.enc2_l = EncoderBlock(64, 128)
        self.enc3_l = EncoderBlock(128, 256)

        # Hard Attention
        self.att = HardAttentionGate(256)

        # Decoder
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, img, roi):
        # Global path
        g1, p1g = self.enc1_g(img)
        g2, p2g = self.enc2_g(p1g)
        g3, p3g = self.enc3_g(p2g)

        # Local path (ROI masked input)
        roi_img = img * roi
        l1, p1l = self.enc1_l(roi_img)
        l2, p2l = self.enc2_l(p1l)
        l3, p3l = self.enc3_l(p2l)

        # Hard Attention Fusion
        fused = self.att(p3g, p3l)

        # Decoder
        d3 = self.dec3(fused, g3)
        d2 = self.dec2(d3, g2)
        d1 = self.dec1(d2, g1)

        out = self.out_conv(d1)
        return out
