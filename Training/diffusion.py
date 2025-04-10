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
sketches_folder = Path("Sketches")
images_folder = Path("Images")

device = "cuda" if torch.cuda.is_available() else "cpu"
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
    def __init__(self, in_channels=3, base_channels=64):  # Updated to accept 3 channels
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
        # Remove upsampling to match dimensions
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
    
    # New method to predict noise
    def predict_noise(self, x, z_noisy, t):
        """Predict noise for diffusion process"""
        t_emb = self.time_embed(t)
        z_pred = self.scheduler(z_noisy, t_emb)
        pred = self.decoder(z_pred)
        return pred

# === DIFFUSION UTILS ===
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    
    # Use index_select for proper indexing
    out = torch.index_select(vals, 0, t)
    
    # Reshape to match the batch dimension and broadcast along other dimensions
    return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original DDPM paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

# === IMPROVED DIFFUSION UTILS ===
class DiffusionUtils:
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = linear_beta_schedule(timesteps).to(device)
        
        # Pre-calculate diffusion variables
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # For sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_sample(self, model, sketch, x_t, t, t_index):
        """
        Sample from p(x_{t-1} | x_t) - the denoising process
        """
        betas_t = get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)
        
        # Use model to predict noise
        t_norm = t.float() / self.timesteps
        pred = model(sketch, t_norm)
        
        # Get the mean for p(x_{t-1} | x_t, x_0)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            # Add noise for stochasticity
            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, sketch, shape):
        """
        Generate samples from the model using the sampling procedure
        """
        device = sketch.device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, sketch, img, t, i)
            imgs.append(img)
        return imgs
    
    @torch.no_grad()
    def sample(self, model, sketch, batch_size=1):
        """
        Sample new images
        """
        image_size = sketch.shape[-2:]
        channels = 3
        return self.p_sample_loop(model, sketch, shape=(batch_size, channels, *image_size))[-1]
    
    def get_loss(self, model, x_0, sketch, t):
        """
        Calculate the diffusion loss
        """
        x_noisy, noise = self.q_sample(x_0, t)
        t_norm = t.float() / self.timesteps
        
        # Get model prediction (direct image prediction approach)
        predicted = model(sketch, t_norm)
        
        # Calculate loss (MSE between prediction and target)
        return F.mse_loss(predicted, x_0)

# === TRAIN ===
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Add transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Maintain 128x128 size
        transforms.ToTensor()
    ])

    dataset = PairedImageDataset(sketches_folder, images_folder, transform=transform)
    
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Initialize model and diffusion utilities
    model = SketchToColorDiffusionLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Set up diffusion process
    T = 1000
    diffusion = DiffusionUtils(timesteps=T, device=device)
    
    EPOCHS = 20

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval_outputs", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for sketch, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            sketch, target = sketch.to(device), target.to(device)
            
            # Sample time steps
            t = torch.randint(0, T, (sketch.shape[0],), device=device).long()
            
            # Calculate the loss
            loss = diffusion.get_loss(model, target, sketch, t)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")
        
        # Save model with .pth extension as requested
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

        # Evaluation and visualization
        model.eval()
        with torch.no_grad():
            sketches, targets = next(iter(val_loader))
            sketches = sketches.to(device)
            
            # Generate samples using the diffusion process
            samples = diffusion.sample(model, sketches)
            
            # Save output images
            utils.save_image(samples, f"eval_outputs/epoch_{epoch+1}_output.png", nrow=4)
            utils.save_image(sketches, f"eval_outputs/epoch_{epoch+1}_sketch.png", nrow=4)
            utils.save_image(targets, f"eval_outputs/epoch_{epoch+1}_target.png", nrow=4)

# Function to generate samples using a trained model
@torch.no_grad()
def sample(model_path, sketch_path, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = SketchToColorDiffusionLite().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform for the input sketch
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Load and preprocess the sketch
    sketch = Image.open(sketch_path).convert("RGB")
    sketch = transform(sketch).unsqueeze(0).to(device)
    
    # Initialize diffusion utilities
    diffusion = DiffusionUtils(timesteps=1000, device=device)
    
    # Generate sample
    sample = diffusion.sample(model, sketch)
    
    # Save or return the sample
    if output_path:
        utils.save_image(sample, output_path)
        print(f"Sample saved to {output_path}")
    
    return sample

# Uncomment to run training:
train()