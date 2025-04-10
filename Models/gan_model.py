import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        # Downsampling
        self.down1 = self._down(in_channels, features)        # 256→128
        self.down2 = self._down(features, features*2)        # 128→64
        self.down3 = self._down(features*2, features*4)      # 64→32
        self.down4 = self._down(features*4, features*8)      # 32→16
        self.down5 = self._down(features*8, features*8)      # 16→8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1),      # 8→4
            nn.ReLU()
        )
        
        # Upsampling
        self.up1 = self._up(features*16, features*8)         # 4→8
        self.up2 = self._up(features*16, features*4)         # 8→16
        self.up3 = self._up(features*8, features*2)          # 16→32
        self.up4 = self._up(features*4, features)            # 32→64
        self.up5 = self._up(features*2, features)            # 64→128
        
        # Final output
        self.out = nn.ConvTranspose2d(features, in_channels, 4, 2, 1)  # 128→256
    
    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    
    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)    # 256→128
        d2 = self.down2(d1)   # 128→64
        d3 = self.down3(d2)   # 64→32
        d4 = self.down4(d3)   # 32→16
        d5 = self.down5(d4)   # 16→8
        
        # Bottleneck
        b = self.bottleneck(d5)  # 8→4
        
        # Decoder with skip connections (using interpolation for size matching)
        u1 = self.up1(torch.cat([b, F.interpolate(d5, size=b.shape[2:])], dim=1))
        u2 = self.up2(torch.cat([u1, F.interpolate(d4, size=u1.shape[2:])], dim=1))
        u3 = self.up3(torch.cat([u2, F.interpolate(d3, size=u2.shape[2:])], dim=1))
        u4 = self.up4(torch.cat([u3, F.interpolate(d2, size=u3.shape[2:])], dim=1))
        u5 = self.up5(torch.cat([u4, F.interpolate(d1, size=u4.shape[2:])], dim=1))
        
        return torch.tanh(self.out(u5))
    



