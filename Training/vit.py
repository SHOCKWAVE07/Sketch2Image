import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ----------------------
# Data Preparation
# ----------------------
image_zip_path = '/content/drive/MyDrive/Images.zip'
sketch_zip_path = '/content/drive/MyDrive/Sketches.zip'
extract_path = '/content/dataset'

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
with zipfile.ZipFile(sketch_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

image_dir = os.path.join(extract_path, 'Images')
sketch_dir = os.path.join(extract_path, 'Sketches')

class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        input_files = sorted(os.listdir(input_dir))
        self.paired_images = []
        for inp in input_files:
            if inp.startswith("sketch_"):
                target_file = inp[len("sketch_" ): ]
            else:
                target_file = inp
            target_path = os.path.join(target_dir, target_file)
            if os.path.exists(target_path):
                self.paired_images.append((inp, target_file))

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        input_file, target_file = self.paired_images[idx]
        input_path = os.path.join(self.input_dir, input_file)
        target_path = os.path.join(self.target_dir, target_file)
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

full_dataset = PairedImageDataset(sketch_dir, image_dir, transform=transform)
dataset = Subset(full_dataset, list(range(min(10000, len(full_dataset)))))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ----------------------
# Model: UNet + ViT
# ----------------------
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ViTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x

class UNetViT(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.conv_to_vit = nn.Conv2d(512, 512, 1)
        self.vit = ViTBlock(dim=512, heads=8)
        self.conv_back = nn.Conv2d(512, 512, 1)

        self.up1 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up3 = UpBlock(256, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x4_pooled = self.pool(x4)

        B, C, H, W = x4_pooled.shape
        x_flat = x4_pooled.flatten(2).transpose(1, 2)
        x_vit = self.vit(x_flat)
        x_recon = x_vit.transpose(1, 2).reshape(B, C, H, W)

        x = self.up1(x_recon, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        return self.final(x)

# ----------------------
# Training
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetViT().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.4f}")

# ----------------------
# Visualization
# ----------------------
model.eval()
inputs, targets = next(iter(dataloader))
inputs = inputs.to(device)
with torch.no_grad():
    preds = model(inputs).cpu()

for i in range(3):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow((inputs[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).numpy())
    plt.title("Sketch")

    plt.subplot(1, 3, 2)
    plt.imshow((targets[i].permute(1, 2, 0) * 0.5 + 0.5).numpy())
    plt.title("Target")

    plt.subplot(1, 3, 3)
    plt.imshow((preds[i].permute(1, 2, 0) * 0.5 + 0.5).numpy())
    plt.title("Prediction")
    plt.show()
