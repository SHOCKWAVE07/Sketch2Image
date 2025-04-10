import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

#############################
# Data Preparation
#############################

# ----- Extract ZIP Files -----
image_zip_path = '/home/m24csa026/CV-Project/Images.zip'
sketch_zip_path = '/home/m24csa026/CV-Project/Sketches.zip'
extract_path = '/home/m24csa026/CV-Project/Asutosh/'

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Images extracted.")

with zipfile.ZipFile(sketch_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Sketches extracted.")

# Define subfolder paths
image_dir = os.path.join(extract_path, 'Images')
sketch_dir = os.path.join(extract_path, 'Sketches')

# ----- Custom Dataset for Paired Images -----
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

# ----- Image Transformations -----
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # Normalize images to [-1, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

input_folder = sketch_dir    # sketches as input
target_folder = image_dir    # color images as targets
full_dataset = PairedImageDataset(input_folder, target_folder, transform=transform)

# ----- Limit to 10,000 Data Points -----
max_data_points = 10000
if len(full_dataset) > max_data_points:
    indices = list(range(max_data_points))
    dataset = Subset(full_dataset, indices)
else:
    dataset = full_dataset

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#############################
# Model: Transformer U-Net
#############################

# Helper: Double Convolution Block
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

#############################
# Training Setup
#############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerUNet(in_channels=3, out_channels=3, base_channels=64, 
                        num_transformer_layers=6, num_heads=8, img_size=512).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

num_epochs = 20
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for step, (input_images, target_images) in enumerate(dataloader):
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_images)
            loss = criterion(outputs, target_images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
print("Training complete!")

# ----- Save the Model -----
torch.save(model.state_dict(), "pure_transformer.pt")
print("Model saved as 'pure_transformer.pth'")

#############################
# Testing & Visualization
#############################

def denormalize(tensor):
    """Convert tensor from [-1,1] to [0,1]."""
    return tensor * 0.5 + 0.5

model.eval()
with torch.no_grad():
    for input_images, target_images in dataloader:
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        predicted_images = model(input_images)
        break

input_images = denormalize(input_images.cpu()).clamp(0, 1)
target_images = denormalize(target_images.cpu()).clamp(0, 1)
predicted_images = denormalize(predicted_images.cpu()).clamp(0, 1)

num_samples = min(3, input_images.size(0))
plt.figure(figsize=(12, 4 * num_samples))
for i in range(num_samples):
    inp_img = input_images[i].permute(1, 2, 0).numpy()
    tgt_img = target_images[i].permute(1, 2, 0).numpy()
    pred_img = predicted_images[i].permute(1, 2, 0).numpy()
    
    plt.subplot(num_samples, 3, i*3 + 1)
    plt.imshow(inp_img)
    plt.title("Input Sketch")
    plt.axis("off")
    
    plt.subplot(num_samples, 3, i*3 + 2)
    plt.imshow(tgt_img)
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(num_samples, 3, i*3 + 3)
    plt.imshow(pred_img)
    plt.title("Predicted Color")
    plt.axis("off")
plt.tight_layout()
plt.savefig("predicted_output.png")
print("Plot saved as 'predicted_output.png'")
plt.show()