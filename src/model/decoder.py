import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.d4 = DecoderBlock(256 + 256 + 256, 128)
        self.d3 = DecoderBlock(128 + 128 + 128, 64)
        self.d2 = DecoderBlock(64 + 64 + 64, 32)
        self.d1 = DecoderBlock(32 + 32 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, skips_img, skips_roi):
        s1_i, s2_i, s3_i, s4_i = skips_img
        s1_r, s2_r, s3_r, s4_r = skips_roi

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, s4_i, s4_r], dim=1)
        x = self.d4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, s3_i, s3_r], dim=1)
        x = self.d3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, s2_i, s2_r], dim=1)
        x = self.d2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, s1_i, s1_r], dim=1)
        x = self.d1(x)

        x = self.final(x)
        x = torch.sigmoid(x)

        return x
