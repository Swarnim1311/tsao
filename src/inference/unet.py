"""Locked M1 U-Net architecture (NB04/NB06, no attention, no fusion branches)."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """UNet-32: 3ch in (VV/VH/DEM) -> 1ch logits. Locked NB06 M1 definition."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.in_ch = in_ch
        self.base = base
        self.e1 = DoubleConv(in_ch, base);      self.p1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(base, base * 2);  self.p2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(base * 2, base * 4); self.p3 = nn.MaxPool2d(2)
        self.e4 = DoubleConv(base * 4, base * 8); self.p4 = nn.MaxPool2d(2)
        self.neck = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.d4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2);  self.d3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2);  self.d2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2);       self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.e1(x); x2 = self.e2(self.p1(x1)); x3 = self.e3(self.p2(x2)); x4 = self.e4(self.p3(x3))
        n = self.neck(self.p4(x4))
        m = self.d4(torch.cat([self.u4(n), x4], 1)); m = self.d3(torch.cat([self.u3(m), x3], 1))
        m = self.d2(torch.cat([self.u2(m), x2], 1)); m = self.d1(torch.cat([self.u1(m), x1], 1))
        return self.out(m)  # logits (B,1,H,W)
