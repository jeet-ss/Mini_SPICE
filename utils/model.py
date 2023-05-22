import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=1):
        super().__init__()
        # check id 1D or 2D conv
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()