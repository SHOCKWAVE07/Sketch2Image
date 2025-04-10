import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = [16, 32, 64, 128, 256]
        self.down1 = DownBlock(3, f[0])
        self.down2 = DownBlock(f[0], f[1])
        self.down3 = DownBlock(f[1], f[2])
        self.down4 = DownBlock(f[2], f[3])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f[3], f[4], 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f[4], f[4], 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up1 = UpBlock(f[4]+f[3], f[3])
        self.up2 = UpBlock(f[3]+f[2], f[2])
        self.up3 = UpBlock(f[2]+f[1], f[1])
        self.up4 = UpBlock(f[1]+f[0], f[0])
        self.final = nn.Conv2d(f[0], 3, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        bn = self.bottleneck(p4)
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = torch.sigmoid(self.final(u4))
        return out