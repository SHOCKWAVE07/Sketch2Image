import torch
from torch import nn
from torch.nn import functional as F
import numpy as np  

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# Helper: Upsampling Block with Skip Connections
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Transformer U-Net Model
class TransformerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 num_transformer_layers=6, num_heads=8, img_size=512):
        """
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_channels: Feature channels in the first conv block.
        num_transformer_layers: Number of transformer encoder layers in the bottleneck.
        num_heads: Number of attention heads.
        img_size: Spatial dimension of the input image (assumed square).
        """
        super(TransformerUNet, self).__init__()
        # Encoder
        self.down1 = double_conv(in_channels, base_channels)          # (B, 64, 512, 512)
        self.pool1 = nn.MaxPool2d(2)                                    # (B, 64, 256, 256)
        self.down2 = double_conv(base_channels, base_channels * 2)      # (B, 128, 256, 256)
        self.pool2 = nn.MaxPool2d(2)                                    # (B, 128, 128, 128)
        self.down3 = double_conv(base_channels * 2, base_channels * 4)  # (B, 256, 128, 128)
        self.pool3 = nn.MaxPool2d(2)                                    # (B, 256, 64, 64)
        self.down4 = double_conv(base_channels * 4, base_channels * 8)  # (B, 512, 64, 64)
        self.pool4 = nn.MaxPool2d(2)                                    # (B, 512, 32, 32)
        
        # Bottleneck (Convolution + Transformer)
        self.bottleneck = double_conv(base_channels * 8, base_channels * 16)  # (B, 1024, 32, 32)
        # For a 512x512 input, after 4 poolings, spatial dims become 32x32.
        self.num_tokens = (img_size // 16) ** 2  # 32*32 = 1024 tokens
        self.emb_dim = base_channels * 16       # 1024
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim, nhead=num_heads, dim_feedforward=self.emb_dim * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Decoder with Skip Connections
        self.up4 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2 + base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder pathway with skip connections
        c1 = self.down1(x)   # (B, 64, 512, 512)
        p1 = self.pool1(c1)  # (B, 64, 256, 256)
        
        c2 = self.down2(p1)  # (B, 128, 256, 256)
        p2 = self.pool2(c2)  # (B, 128, 128, 128)
        
        c3 = self.down3(p2)  # (B, 256, 128, 128)
        p3 = self.pool3(c3)  # (B, 256, 64, 64)
        
        c4 = self.down4(p3)  # (B, 512, 64, 64)
        p4 = self.pool4(c4)  # (B, 512, 32, 32)
        
        # Bottleneck
        bn = self.bottleneck(p4)  # (B, 1024, 32, 32)
        B, C, H, W = bn.shape     # C should be 1024, H=W=32
        tokens = bn.view(B, H * W, C)  # (B, 1024, 1024)
        tokens = tokens + self.pos_embedding  # add learned positional encoding
        tokens = self.transformer(tokens)     # (B, 1024, 1024)
        bn_trans = tokens.view(B, C, H, W)      # (B, 1024, 32, 32)
        
        # Decoder pathway with skip connections
        d4 = self.up4(bn_trans, c4)  # (B, 512, 64, 64)
        d3 = self.up3(d4, c3)        # (B, 256, 128, 128)
        d2 = self.up2(d3, c2)        # (B, 128, 256, 256)
        d1 = self.up1(d2, c1)        # (B, 64, 512, 512)
        out = self.out_conv(d1)      # (B, out_channels, 512, 512)
        out = torch.tanh(out)        # Constrain output to [-1, 1]
        return out
