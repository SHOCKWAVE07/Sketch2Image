import os
import zipfile
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# === SETUP ===

sketches_folder = Path("input_of_cv/Sketches")
images_folder = Path("input_of_cv/Images")



# === DATASET ===
class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        input_files = sorted(os.listdir(input_dir))
        self.paired_images = []
        for inp in input_files:
            if inp.startswith("sketch_"):
                target_file = inp[len("sketch_"):]
            else:
                target_file = inp
            target_path = os.path.join(target_dir, target_file)
            if os.path.exists(target_path):
                self.paired_images.append((inp, target_file))
            else:
                print(f"Warning: Target file {target_file} does not exist for input {inp}")

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


# --- Sinusoidal Time Embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(-np.log(10000) * torch.arange(half_dim) / (half_dim - 1)).to(t.device)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# --- Encoder ---
class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):  # Change from in_channels=1 to in_channels=3
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4),
        )

    def forward(self, x): return self.encoder(x)

# --- UNet Block ---
class UNetBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, x): return self.block(x)

# --- Scheduler (Lite UNet) ---
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

# --- Decoder ---
class SimpleDecoder(nn.Module):
    def __init__(self, in_c, base_c=64, out_c=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_c, base_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_c, base_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_c, out_c, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x): 
        # Remove the interpolation
        return self.decoder(x)

# --- Full Model ---
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


# === DIFFUSION UTILS ===
def get_beta_schedule(T): return torch.linspace(1e-4, 0.02, T)
def q_sample(x, t, noise, betas):
    sqrt_alphas = torch.sqrt(1 - betas).to(x.device)
    sqrt_one_minus = torch.sqrt(betas).to(x.device)
    return sqrt_alphas[t].view(-1, 1, 1, 1) * x + sqrt_one_minus[t].view(-1, 1, 1, 1) * noise

# === TRAIN ===
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Changed from 128 to 256
        transforms.ToTensor()
    ])

    dataset = PairedImageDataset(sketches_folder, images_folder, transform=transform)
    
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = SketchToColorDiffusionLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    T = 1000
    betas = get_beta_schedule(T)
    EPOCHS = 20

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for sketch, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            sketch, target = sketch.to(device), target.to(device)
            z_clean = model.encoder(sketch).detach()
            t = torch.randint(0, T, (sketch.size(0),), dtype=torch.long).to(device)
            noise = torch.randn_like(z_clean)
            z_noisy = q_sample(z_clean, t, noise, betas)
            t_norm = t.float() / T
            output = model.decoder(model.scheduler(z_noisy, model.time_embed(t_norm)))
            loss = loss_fn(output, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

        # Evaluation
        model.eval()
        with torch.no_grad():
            sketch, target = next(iter(val_loader))
            sketch = sketch.to(device)
            t_test = torch.full((sketch.size(0),), T-1, dtype=torch.long).to(device)
            z = model.encoder(sketch)
            z_denoised = model.scheduler(z, model.time_embed(t_test.float() / T))
            output = model.decoder(z_denoised)
            utils.save_image(output, f"eval_outputs/epoch_{epoch+1}_output.png", nrow=4)
            utils.save_image(sketch.expand(-1,3,-1,-1), f"eval_outputs/epoch_{epoch+1}_sketch.png", nrow=4)
            utils.save_image(target, f"eval_outputs/epoch_{epoch+1}_target.png", nrow=4)

# Uncomment to run training:
train()