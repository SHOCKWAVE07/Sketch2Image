#!/usr/bin/env python
# coding: utf-8

# In[53]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random



# In[2]:





# In[3]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device


# In[4]:


class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, img_dir, transform=None):
        self.Sketch = sketch_dir
        self.Img = img_dir
        self.transform = transform
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.sketch_files = sorted([f for f in os.listdir(self.Sketch) if f.lower().endswith(valid_exts)])
        self.img_files = sorted([f for f in os.listdir(self.Img) if f.lower().endswith(valid_exts)])

        assert len(self.sketch_files) == len(self.img_files), \
            "Number of sketch and image files must be equal"
        
        for sketch_file, img_file in zip(self.sketch_files, self.img_files):
            assert os.path.splitext(sketch_file)[0] == os.path.splitext(img_file)[0], \
                f"Mismatched file pair: {sketch_file} and {img_file}"
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        sketch_path = os.path.join(self.Sketch, self.sketch_files[index])
        img_path = os.path.join(self.Img, self.img_files[index])
        
        sketch = Image.open(sketch_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            sketch = self.transform(sketch)
            img = self.transform(img)

        return sketch, img


# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),                  # PIL input
    transforms.ToTensor(),                          # Converts to tensor, [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5),            # Normalize to [-1, 1]
                         (0.5, 0.5, 0.5)),
])


full_dataset = SketchToImageDataset(
    sketch_dir="Sketches",
    img_dir="Images",
    transform=transform
)

# Split into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
trainset, testset = random_split(full_dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

print(f"Training samples: {len(trainset)}")
print(f"Testing samples: {len(testset)}")


# In[ ]:



# In[6]:




# In[37]:



# In[54]:


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

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        
        # Initial layer
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Intermediate layers
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                self._block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        # Final layer
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
                nn.Sigmoid()
            )
        )
        
        self.model = nn.Sequential(*layers)
    
    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        return x 



# In[55]:


# Test with your actual input size
test_input = torch.randn(1, 3, 256, 256).to(device)
gen = Generator().to(device)
output = gen(test_input)
print(f"Output shape: {output.shape}")  # Should be [1, 3, 256, 256]


# In[61]:


## 3. Training Setup
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Initialize models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Initialize weights
initialize_weights(gen)
initialize_weights(disc)

# Loss and optimizers
opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))


# In[63]:



# 1. Initialize models and optimizers
gen = Generator().to(device)
disc = Discriminator().to(device)

opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# 2. Updated loss functions (AMP compatible)
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# 3. Initialize GradScaler
scaler = GradScaler()

# 4. Updated training loop
def train(loader, gen, disc, opt_gen, opt_disc, criterion_GAN, criterion_L1, epochs):
    for epoch in range(epochs):
        for batch_idx, (sketches, reals) in enumerate(loader):
            sketches = sketches.to(device)
            reals = reals.to(device)
            
            # --- Train Discriminator ---
            opt_disc.zero_grad()
            
            # Mixed precision context
            with autocast():
                # Generate fake images
                fakes = gen(sketches)
                
                # Discriminator outputs
                disc_real = disc(sketches, reals)
                disc_fake = disc(sketches, fakes.detach())
                
                # Calculate losses
                loss_disc_real = criterion_GAN(disc_real, torch.ones_like(disc_real))
                loss_disc_fake = criterion_GAN(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            # Backpropagate discriminator
            scaler.scale(loss_disc).backward()
            scaler.step(opt_disc)
            
            # --- Train Generator ---
            opt_gen.zero_grad()
            
            # Mixed precision context
            with autocast():
                # Generator outputs
                fakes = gen(sketches)
                disc_fake = disc(sketches, fakes)
                
                # Calculate losses
                loss_GAN = criterion_GAN(disc_fake, torch.ones_like(disc_fake))
                loss_L1 = criterion_L1(fakes, reals) * 100  # L1 weight
                loss_gen = loss_GAN + loss_L1
            
            # Backpropagate generator
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            
            # Update scaler
            scaler.update()
            
            # Print training progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} "
                      f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

# Start training
print("Starting Training...")
train(train_loader, gen, disc, opt_gen, opt_disc, criterion_GAN, criterion_L1, epochs=20)

# Save models
torch.save(gen.state_dict(), "generator.pth")
torch.save(disc.state_dict(), "discriminator.pth")


# In[59]:



# In[ ]:



def evaluate_and_save_samples(gen, test_loader, device, save_dir="eval_samples", num_samples=50):
    """
    Evaluate generator on test set and save random samples
    
    Args:
        gen (nn.Module): Trained generator model
        test_loader (DataLoader): Test set dataloader
        device (torch.device): Device to run evaluation on
        save_dir (str): Directory to save samples
        num_samples (int): Number of samples to save
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to eval mode
    gen.eval()
    
    # Get random indices for samples
    total_samples = len(test_loader.dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Prepare for evaluation
    saved_count = 0
    with torch.no_grad():
        for batch_idx, (sketches, reals) in enumerate(test_loader):
            # Move data to device
            sketches = sketches.to(device)
            reals = reals.to(device)
            
            # Generate fake images
            fakes = gen(sketches)
            
            # Convert from [-1,1] to [0,1]
            sketches = (sketches + 1) / 2
            reals = (reals + 1) / 2
            fakes = (fakes + 1) / 2
            
            # Process each sample in batch
            for i in range(sketches.size(0)):
                global_idx = batch_idx * test_loader.batch_size + i
                
                if global_idx in sample_indices:
                    # Create side-by-side image
                    combined = torch.cat([sketches[i], fakes[i], reals[i]], dim=-1)
                    
                    # Save image
                    save_path = os.path.join(save_dir, f"sample_{global_idx:04d}.png")
                    save_image(combined, save_path)
                    
                    saved_count += 1
                    print(f"Saved sample {saved_count}/{num_samples} to {save_path}")
                    
                    # Stop if we've saved enough samples
                    if saved_count >= num_samples:
                        return


gen = Generator().to(device)
gen.load_state_dict(torch.load("generator.pth"))

evaluate_and_save_samples(
    gen=gen,
    test_loader=test_loader,
    device=device,
    save_dir="evaluation_results",
    num_samples=50
)


