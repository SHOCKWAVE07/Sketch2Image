import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(-np.log(10000) * torch.arange(half_dim) / (half_dim - 1)).to(t.device)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4),
        )
    def forward(self, x): return self.encoder(x)

class UNetBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_c, out_c, 3, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.block(x)

class UNetScheduler(nn.Module):
    def __init__(self, base_c=64, time_emb_dim=128):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(time_emb_dim, base_c * 4), nn.ReLU())
        self.down = UNetBlock(base_c * 4, base_c * 4, base_c * 4)
        self.middle = UNetBlock(base_c * 4, base_c * 4, base_c * 4)
        self.up = UNetBlock(base_c * 8, base_c * 2, base_c * 2)

    def forward(self, z, t_emb):
        t_feat = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        z = z + t_feat
        d = self.down(z)
        m = self.middle(F.avg_pool2d(d, 2))
        u = F.interpolate(m, scale_factor=2, mode='nearest')
        return self.up(torch.cat([u, d], dim=1))

# SimpleDecoder
class SimpleDecoder(nn.Module):
    def __init__(self, in_c, base_c=64, out_c=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_c, base_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_c, base_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_c, out_c, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)  # <-- Removed upsampling


class SketchToColorDiffusionLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.time_embed = TimeEmbedding(128)
        self.scheduler = UNetScheduler(time_emb_dim=128)
        self.decoder = SimpleDecoder(in_c=64*2)

    def forward(self, x, t):
        z = self.encoder(x)
        t_emb = self.time_embed(t)
        z_denoised = self.scheduler(z, t_emb)
        return self.decoder(z_denoised)