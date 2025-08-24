
"""
Train ResCGAN: conditional generator to map angle maps -> FFT-MA realizations.
This is a cleaned version of the user's script with English comments and repo-friendly paths.

Original source: Train_ResCGAN.txt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Dataset
# -------------------------------
class CustomDataset(Dataset):
    """
    Each row of the CSV concatenates angle map (121x121) and FFT-MA realization (121x121).
    angle: first 121*121 values (reshaped to [1,121,121])
    fftma: last  121*121 values (reshaped to [1,121,121])
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values.astype(np.float32)
        angle = sample[:121*121].reshape(1, 121, 121)   # angle map
        fftma = sample[121*121:].reshape(1, 121, 121)   # FFT-MA realization
        if self.transform:
            angle = self.transform(angle)
            fftma = self.transform(fftma)
        return torch.tensor(angle, dtype=torch.float32), torch.tensor(fftma, dtype=torch.float32)

# -------------------------------
# Residual Block
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.lrelu(out)

# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_deconv = nn.ConvTranspose2d(1, 128, kernel_size=3, stride=1, padding=1)
        # Name each residual block explicitly (matches original intent)
        self.residual_block1 = ResidualBlock(128, 128)
        self.residual_block2 = ResidualBlock(128, 128)
        self.residual_block3 = ResidualBlock(128, 128)
        self.residual_block4 = ResidualBlock(128, 128)
        self.residual_block5 = ResidualBlock(128, 128)
        self.residual_block6 = ResidualBlock(128, 128)
        self.residual_block7 = ResidualBlock(128, 128)
        self.residual_block8 = ResidualBlock(128, 128)
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, angle):
        x0 = self.initial_deconv(angle)
        x1 = self.residual_block1(x0)
        x2 = self.residual_block2(x1)
        x3 = self.residual_block3(x2)
        x4 = self.residual_block4(x3)
        x5 = self.residual_block5(x4) + x3
        x6 = self.residual_block6(x5) + x2
        x7 = self.residual_block7(x6) + x1
        x8 = self.residual_block8(x7)
        return self.final_conv(x8)

# -------------------------------
# Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = spectral_norm(nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1))
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
        )
        self.final_conv = spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=2, padding=1))
        self.fc = spectral_norm(nn.Linear(961, 1))

    def forward(self, angle, fftma):
        x = torch.cat((angle, fftma), dim=1)
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# -------------------------------
# Metrics / Utils
# -------------------------------
def calculate_ssim(real_images, generated_images, data_range=1.0):
    """Compute mean SSIM across a batch of single-channel images."""
    scores = []
    for real, gen in zip(real_images, generated_images):
        real = real.squeeze().cpu().numpy()
        gen  = gen.squeeze().cpu().numpy()
        score, _ = ssim(real, gen, data_range=data_range, full=True)
        scores.append(score)
    return float(np.mean(scores))

def save_models(generator, discriminator,
                generator_path='artifacts/generator.pth',
                discriminator_path='artifacts/discriminator.pth'):
    os.makedirs(os.path.dirname(generator_path), exist_ok=True)
    torch.save(generator.state_dict(), generator_path)
    if discriminator is not None:
        torch.save(discriminator.state_dict(), discriminator_path)
    print(f"✅ Models saved to {generator_path} and {discriminator_path}")

def load_models(generator, discriminator=None, generator_path=None, discriminator_path=None):
    """Load weights; safely strips 'module.' when present."""
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if generator_path:
        sd_g = torch.load(generator_path, map_location=map_location)
        sd_g = {k.replace('module.', ''): v for k, v in sd_g.items()}
        generator.load_state_dict(sd_g, strict=True)
        print(f"🔹 Loaded generator from {generator_path}")
    if discriminator is not None and discriminator_path:
        sd_d = torch.load(discriminator_path, map_location=map_location)
        sd_d = {k.replace('module.', ''): v for k, v in sd_d.items()}
        discriminator.load_state_dict(sd_d, strict=True)
        print(f"🔹 Loaded discriminator from {discriminator_path}")

# -------------------------------
# Training Loop
# -------------------------------
def train(generator, discriminator, dataloader, device, num_epochs=5):
    criterion   = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    sch_g = ExponentialLR(opt_g, gamma=0.96)
    sch_d = ExponentialLR(opt_d, gamma=0.96)

    losses_G, losses_D, ssim_scores = [], [], []

    for epoch in range(num_epochs):
        for angle, fftma in dataloader:
            angle, fftma = angle.to(device), fftma.to(device)
            b = angle.size(0)
            labels_real = torch.ones(b, 1, device=device)
            labels_fake = torch.zeros(b, 1, device=device)

            # --- D step ---
            opt_d.zero_grad()
            out_real = discriminator(angle, fftma)
            loss_D_real = criterion(out_real, labels_real)

            gen = generator(angle).detach()
            out_fake = discriminator(angle, gen)
            loss_D_fake = criterion(out_fake, labels_fake)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_d.step()

            # --- G step ---
            opt_g.zero_grad()
            gen = generator(angle)
            out_fake = discriminator(angle, gen)
            loss_G = criterion(out_fake, labels_real)
            loss_G.backward()
            opt_g.step()

        sch_g.step(); sch_d.step()
        losses_D.append(float(loss_D.item())); losses_G.append(float(loss_G.item()))

        # quick SSIM sample every epoch
        with torch.no_grad():
            idx = torch.randint(0, len(dataloader.dataset), (min(50, len(dataloader.dataset)),))
            angles = torch.stack([dataloader.dataset[i][0] for i in idx]).to(device)
            reals  = torch.stack([dataloader.dataset[i][1] for i in idx]).to(device)
            gens   = generator(angles)
            ssim_epoch = calculate_ssim(reals, gens, data_range=1.0)
            ssim_scores.append(ssim_epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"LossD: {losses_D[-1]:.4f} | LossG: {losses_G[-1]:.4f} | SSIM: {ssim_epoch:.4f} | "
              f"LR_G: {sch_g.get_last_lr()[0]:.6f} | LR_D: {sch_d.get_last_lr()[0]:.6f}")

    return losses_G, losses_D, ssim_scores

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to samples_angle_fftma.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CustomDataset(args.csv)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    gen = Generator()
    disc = Discriminator()

    if torch.cuda.device_count() > 1:
        gen  = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)

    gen  = gen.to(device)
    disc = disc.to(device)

    losses_G, losses_D, ssim_scores = train(gen, disc, dataloader, device, num_epochs=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    save_models(gen.module if isinstance(gen, nn.DataParallel) else gen,
                disc.module if isinstance(disc, nn.DataParallel) else disc,
                generator_path=os.path.join(args.out, "Generator.pth"),
                discriminator_path=os.path.join(args.out, "Discriminator.pth"))
