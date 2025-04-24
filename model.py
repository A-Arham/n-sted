import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ========= Define Model Components =========
def crop_or_pad(source, target):
    """
    Adjust the temporal dimension of `source` to match that of `target`
    via center cropping (if larger) or symmetric padding (if smaller).
    """
    target_len = target.shape[2]
    source_len = source.shape[2]
    if source_len > target_len:
        diff = source_len - target_len
        start = diff // 2
        end = start + target_len
        return source[:, :, start:end]
    elif source_len < target_len:
        diff = target_len - source_len
        pad_left = diff // 2
        pad_right = diff - pad_left
        return F.pad(source, (pad_left, pad_right))
    else:
        return source

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x1 = self.enc1(x)                     # [batch, 64, L]
        x2 = self.enc2(self.pool(x1))           # [batch, 128, ceil(L/2)]
        x3 = self.enc3(self.pool(x2))           # [batch, 256, ceil(L/4)]
        x4 = self.enc4(self.pool(x3))           # [batch, 512, ceil(L/8)]
        return x1, x2, x3, x4

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        # Using output_padding=0 so output length depends on input length and pooling
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, output_padding=0)
        self.dec1 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, output_padding=0)
        self.dec2 = UNetBlock(256, 128)
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, output_padding=0)
        self.dec3 = UNetBlock(128, 64)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=1)  # one output channel

    # Note: We now pass target_length, which is the original input length (from x1)
    def forward(self, x1, x2, x3, x4, target_length):
        x = self.up1(x4)
        x3_adj = crop_or_pad(x3, x)
        x = self.dec1(torch.cat([x, x3_adj], dim=1))
        
        x = self.up2(x)
        x2_adj = crop_or_pad(x2, x)
        x = self.dec2(torch.cat([x, x2_adj], dim=1))
        
        x = self.up3(x)
        x1_adj = crop_or_pad(x1, x)
        x = self.dec3(torch.cat([x, x1_adj], dim=1))
        
        x = self.final_conv(x)
        # Crop or pad the final output to match the original length
        if x.shape[2] != target_length:
            if x.shape[2] > target_length:
                diff = x.shape[2] - target_length
                start = diff // 2
                x = x[:, :, start:start+target_length]
            else:
                diff = target_length - x.shape[2]
                pad_left = diff // 2
                pad_right = diff - pad_left
                x = F.pad(x, (pad_left, pad_right))
        x = torch.sigmoid(x)
        return x

class UNetEEG(nn.Module):
    def __init__(self, in_channels=129):
        super(UNetEEG, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        # Use the length of x1 (which uses "same" padding) as target length
        target_length = x1.shape[2]
        out = self.decoder(x1, x2, x3, x4, target_length)
        return out

