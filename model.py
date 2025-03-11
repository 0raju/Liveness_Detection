from typing import List
import torch
torch.cuda.empty_cache()
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size=3, padding="same", bias=False)
        self.drop = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(channels_out)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.drop(out)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x
        return out

class CustomResNet(nn.Module):
    def __init__(self, initial_channels: int, channels_list: List[int], num_classes: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        c_in = initial_channels
        for c_out in channels_list:
            if c_in != c_out:
                layers += [
                    nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                    nn.Dropout(p=dropout),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                ]
            layers.append(ConvBlock(c_out, c_out, dropout=dropout))
            c_in = c_out
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        ]
        layers.append(nn.Linear(c_in, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze()

def CustomResNet18():
    return CustomResNet(2, [64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512, 512], num_classes=1, dropout=0.0)
