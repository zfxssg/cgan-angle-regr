
"""
Train Angle Regression Network: map FFT-MA realizations -> angle maps.
Cleaned English-comment version of the user's script.

Original source: Train_AngleRegressionModel.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Dataset
# -------------------------------
class CustomDataset(Dataset):
    """
    Each row of the CSV concatenates angle map (121x121) and FFT-MA realization (121x121).
    For regression we swap I/O so that input is FFT-MA and target is angle map.
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values.astype(np.float32)
        # Swap: input=FFTMA, target=Angle
        angle = sample[121*121:].reshape(1, 121, 121)  # FFT-MA realization (input)
        fftma = sample[:121*121].reshape(1, 121, 121)  # Angle map (target)
        if self.transform:
            angle = self.transform(angle)
            fftma = self.transform(fftma)
        return torch.tensor(angle, dtype=torch.float32), torch.tensor(fftma, dtype=torch.float32)

# -------------------------------
# Model
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

class AngleRegressionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

def calculate_ssim(x, y, data_range=1.0):
    """Compute mean SSIM for batched single-channel images."""
    scores = []
    for i in range(x.size(0)):
        s = ssim(x[i,0].cpu().numpy(), y[i,0].cpu().numpy(), data_range=data_range)
        scores.append(s)
    return torch.tensor(scores).mean()

def train_regression(model, train_loader, test_loader, device, num_epochs=5, lr=1e-4, save_path='artifacts/Angle_regression.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    train_losses, test_losses = [], []
    train_ssims,  test_ssims  = [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        preds, targs = [], []
        for angle, fftma in train_loader:
            angle, fftma = angle.to(device), fftma.to(device)
            out = model(angle)
            loss = criterion(out, fftma)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run_loss += loss.item()
            preds.append(out.detach()); targs.append(fftma)

        scheduler.step()
        train_loss = run_loss / len(train_loader)
        train_pred = torch.cat(preds, dim=0); train_targ = torch.cat(targs, dim=0)
        train_ssim = float(calculate_ssim(train_pred, train_targ, data_range=1.0).item())
        train_losses.append(train_loss); train_ssims.append(train_ssim)

        # eval
        model.eval()
        e_loss, e_preds, e_targs = 0.0, [], []
        with torch.no_grad():
            for angle, fftma in test_loader:
                angle, fftma = angle.to(device), fftma.to(device)
                out = model(angle)
                loss = criterion(out, fftma)
                e_loss += loss.item()
                e_preds.append(out); e_targs.append(fftma)
        e_loss /= len(test_loader)
        e_pred = torch.cat(e_preds, dim=0); e_targ = torch.cat(e_targs, dim=0)
        e_ssim = float(calculate_ssim(e_pred, e_targ, data_range=1.0).item())
        test_losses.append(e_loss); test_ssims.append(e_ssim)

        print(f"Epoch {epoch+1}/{num_epochs} | LR={scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss={train_loss:.6f} SSIM={train_ssim:.4f} | "
              f"Test Loss={e_loss:.6f} SSIM={e_ssim:.4f}")

    # save weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Regression model saved to {save_path}")

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to samples_angle_fftma.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out", default="artifacts/Angle_regression.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full = CustomDataset(args.csv)
    n_total = len(full)
    n_train = int((1.0 - args.val_split) * n_total)
    n_test  = n_total - n_train
    train_ds, test_ds = random_split(full, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = AngleRegressionNet()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    train_regression(model, train_loader, test_loader, device, num_epochs=args.epochs, lr=1e-4, save_path=args.out)
