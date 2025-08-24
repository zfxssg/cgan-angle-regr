
"""
Testing & visualization for trained models.
- Loads trained Generator and AngleRegressionNet
- Compares DSS-LA / FFT-MA / Generator outputs
- Saves figures and CSVs for the paper

Original source: Test_TrainedModel.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import math
import os

# ------------- Dataset -------------
class CustomDataset(Dataset):
    """
    CSV layout: angle (121x121) then FFT-MA (121x121), per original dataset.
    This loader returns (angle_map, fftma_realization).
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values.astype(np.float32)
        angle = sample[:121*121].reshape(1, 121, 121)  # angle map
        fftma = sample[121*121:].reshape(1, 121, 121)  # FFT-MA realization
        if self.transform:
            angle = self.transform(angle)
            fftma = self.transform(fftma)
        return torch.tensor(angle, dtype=torch.float32), torch.tensor(fftma, dtype=torch.float32)

# ------------- Residual blocks (regression & generator) -------------
class ResidualBlock_reg(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=0.3)
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

# ------------- Models -------------
class AngleRegressionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=6):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock_reg(base_channels) for _ in range(num_blocks)])
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.initial(x); x = self.res_blocks(x); return self.output_layer(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_deconv = nn.ConvTranspose2d(1, 128, kernel_size=3, stride=1, padding=1)
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

# ------------- FFT-MA -------------
class BatchFFTMA(nn.Module):
    """
    Batched FFT-MA simulation driven by a spatially varying angle field.
    """
    def __init__(self, size=(121, 121), a=15, b=3, r=1, window_radius=45, device=None):
        super().__init__()
        self.H, self.W = size
        self.a, self.b, self.r = a, b, r
        self.d = window_radius
        self.win_size = 2 * self.d + 1
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.linspace(-self.d, self.d, self.win_size, device=self.device)
        y = torch.linspace(-self.d, self.d, self.win_size, device=self.device)
        Xm, Ym = torch.meshgrid(x, y, indexing="ij")
        self.register_buffer("Xm", Xm)
        self.register_buffer("Ym", Ym)

    def forward(self, angle_matrix):
        """
        angle_matrix: [1,1,H,W] or [H,W] angle (radians)
        returns: [H,W] simulated FFT-MA field
        """
        if angle_matrix.ndim == 4:
            angle_matrix = angle_matrix.squeeze(0).squeeze(0)
        angle_matrix = angle_matrix.to(self.device)

        noise = torch.randn(1, 1, self.H + 2*self.d, self.W + 2*self.d, device=self.device)

        patches = F.unfold(noise, kernel_size=self.win_size).squeeze(0).transpose(0, 1)
        patches = patches.view(-1, self.win_size, self.win_size)  # [HW, K, K]

        angles = angle_matrix.view(-1)  # [HW]
        R_stack = self.batch_autocorrelation(angles)              # [HW, K, K]
        fftma_values = self.batch_fftma(patches, R_stack)         # [HW]
        return fftma_values.view(self.H, self.W)

    def batch_autocorrelation(self, theta_vec):
        cos_t = torch.cos(theta_vec).unsqueeze(1).unsqueeze(2)
        sin_t = torch.sin(theta_vec).unsqueeze(1).unsqueeze(2)
        Xm = self.Xm.unsqueeze(0); Ym = self.Ym.unsqueeze(0)
        a_part = ((Xm * cos_t + Ym * sin_t) ** 2) / (self.a ** 2)
        b_part = ((-Xm * sin_t + Ym * cos_t) ** 2) / (self.b ** 2)
        return torch.exp(-((a_part + b_part) ** (1 / (1 + self.r))))

    def batch_fftma(self, W_batch, R_batch):
        Wf = torch.fft.fft2(W_batch)
        Rf = torch.fft.fft2(torch.fft.fftshift(R_batch, dim=(-2, -1)))
        G  = torch.sqrt(Rf + 1e-8)
        v  = torch.fft.ifft2(Wf * G).real
        v  = (v - v.mean(dim=(1,2), keepdim=True)) / (v.std(dim=(1,2), keepdim=True) + 1e-6)
        return v[:, self.d, self.d]

# ------------- DSS-LA -------------
def generalized_cov(hx, hy, angle, range_major, range_minor, r=1.0):
    cosa, sina = torch.cos(angle), torch.sin(angle)
    x_rot = hx * cosa + hy * sina
    y_rot = -hx * sina + hy * cosa
    a_part = (x_rot / range_major) ** 2
    b_part = (y_rot / range_minor) ** 2
    exponent = (a_part + b_part) ** (1.0 / (1.0 + r))
    return torch.exp(-exponent)

def dss_la_unconditional(grid_size=(121,121), angle_map=None, range_major=15, range_minor=3, search_radius=10):
    H, W = grid_size
    sim = torch.full((H, W), float('nan'))
    if angle_map is None:
        angle_map = torch.rand(H, W) * math.pi
    all_coords = [(i, j) for i in range(H) for j in range(W)]
    random.shuffle(all_coords)
    for (i, j) in all_coords:
        imin, imax = max(i - search_radius, 0), min(i + search_radius + 1, H)
        jmin, jmax = max(j - search_radius, 0), min(j + search_radius + 1, W)
        neighborhood, values = [], []
        for ii in range(imin, imax):
            for jj in range(jmin, jmax):
                if not math.isnan(sim[ii, jj]):
                    neighborhood.append((ii, jj)); values.append(sim[ii, jj])
        if len(neighborhood) == 0:
            sim[i, j] = torch.randn(1).item(); continue
        coords = torch.tensor(neighborhood, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dx = coords[:,0] - i; dy = coords[:,1] - j
        angle = angle_map[i, j]
        cov_vector = generalized_cov(dx, dy, angle, range_major, range_minor, r=1.0)
        dx_mat = coords[:,0].unsqueeze(1) - coords[:,0].unsqueeze(0)
        dy_mat = coords[:,1].unsqueeze(1) - coords[:,1].unsqueeze(0)
        cov_matrix = generalized_cov(dx_mat, dy_mat, angle, range_major, range_minor, r=1.0)
        try:
            weights = torch.linalg.solve(cov_matrix + 1e-5 * torch.eye(len(values)), cov_vector)
            mean = torch.dot(weights, values)
            var = 1 - torch.dot(weights, cov_vector)
            var = max(float(var), 1e-6)
        except Exception:
            mean = values.mean(); var = 1.0
        sim[i, j] = torch.normal(float(mean), math.sqrt(var))
    return sim

# ------------- Loading -------------
def load_state_safely(model, path):
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

# ------------- Visualization Helpers -------------
def visualize_comparison_grid(batch_fftma_model, generator, dataset, indices, device, out_pdf="comparison_results.pdf"):
    """
    4 rows: Angle | DSS-LA | FFT-MA | Generator, for several indices.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    def run_dssla(angle_tensor):
        angle_np = angle_tensor.squeeze().cpu().numpy()
        angle_map = torch.tensor(angle_np, dtype=torch.float32)
        output = dss_la_unconditional(grid_size=angle_map.shape, angle_map=angle_map)
        return output.numpy()

    num_cols, num_rows = len(indices), 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for col, idx in enumerate(indices):
        angle, _ = dataset[idx]
        angle = angle.unsqueeze(0).to(device)
        angle_np = angle.squeeze().cpu().numpy()
        angle_img = np.rot90(angle_np, k=-1)

        axes[0, col].imshow(angle_img, cmap='viridis'); axes[0, col].axis('on')

        dssla_img = np.rot90(run_dssla(angle), k=-1)
        axes[1, col].imshow(dssla_img, cmap='viridis', vmin=-3.5, vmax=3.5); axes[1, col].axis('on')

        with torch.no_grad():
            fftma_img = np.rot90(batch_fftma_model(angle.squeeze(0)).cpu().squeeze().numpy(), k=-1)
        axes[2, col].imshow(fftma_img, cmap='viridis', vmin=-3.5, vmax=3.5); axes[2, col].axis('on')

        with torch.no_grad():
            gen_img = np.rot90(generator(angle).cpu().squeeze().numpy(), k=-1)
        axes[3, col].imshow(gen_img, cmap='viridis', vmin=-3.5, vmax=3.5); axes[3, col].axis('on')

    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', dpi=300)
    print(f"✅ Saved {out_pdf}")

def visualize_and_evaluate_single_index(batch_fftma_model, generator, angle_regressor, dataset, idx, device, save_path="result_single.pdf"):
    """
    6 rows x 5 cols grid: DSS-LA samples + regression, FFT-MA samples + regression,
    Generator samples + regression, with GT angle shown in relevant rows.
    Also saves flattened samples CSV and prints MSE/SSIM/Corr table.
    """
    import numpy as np, matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr
    import pandas as pd

    def run_dssla(angle_tensor):
        angle_np = angle_tensor.squeeze().cpu().numpy()
        angle_map = torch.tensor(angle_np, dtype=torch.float32)
        return dss_la_unconditional(grid_size=angle_map.shape, angle_map=angle_map).numpy()

    def compute_metrics(gt, pred):
        gt_flat = gt.flatten(); pred_flat = pred.flatten()
        mse_val = np.mean((gt_flat - pred_flat) ** 2)
        ssim_val = ssim(gt, pred, data_range=pred.max() - pred.min())
        corr_val = pearsonr(gt_flat, pred_flat)[0]
        return mse_val, ssim_val, corr_val

    angle_gt, _ = dataset[idx]
    angle_gt = angle_gt.unsqueeze(0).to(device)
    angle_np = angle_gt.squeeze().cpu().numpy()
    angle_img = np.rot90(angle_np, k=-1)

    num_samples = 4
    dssla_imgs, fftma_imgs, gen_imgs = [], [], []
    dssla_regr, fftma_regr, gen_regr = [], [], []

    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    axes[0, 0].axis('off')
    for i in range(num_samples):
        dssla_out = run_dssla(angle_gt)
        dssla_imgs.append(dssla_out)
        axes[0, i + 1].imshow(np.rot90(dssla_out, k=-1), 'viridis', vmin=-3.5, vmax=3.5); axes[0, i + 1].axis('on')

    axes[1, 0].imshow(angle_img, cmap='viridis'); axes[1, 0].axis('on')
    for i in range(num_samples):
        pred_angle = angle_regressor(torch.tensor(dssla_imgs[i]).unsqueeze(0).unsqueeze(0).to(device))
        pred_np = np.rot90(pred_angle.squeeze().detach().cpu().numpy(), k=-1)
        dssla_regr.append(pred_np)
        axes[1, i + 1].imshow(pred_np, cmap='viridis'); axes[1, i + 1].axis('on')

    axes[2, 0].axis('off')
    for i in range(num_samples):
        with torch.no_grad():
            fftma_out = batch_fftma_model(angle_gt.squeeze(0)).cpu().squeeze().numpy()
        fftma_imgs.append(fftma_out)
        axes[2, i + 1].imshow(np.rot90(fftma_out, k=-1), 'viridis', vmin=-3.5, vmax=3.5); axes[2, i + 1].axis('on')

    axes[3, 0].imshow(angle_img, cmap='viridis'); axes[3, 0].axis('on')
    for i in range(num_samples):
        pred_angle = angle_regressor(torch.tensor(fftma_imgs[i]).unsqueeze(0).unsqueeze(0).to(device))
        pred_np = np.rot90(pred_angle.squeeze().detach().cpu().numpy(), k=-1)
        fftma_regr.append(pred_np)
        axes[3, i + 1].imshow(pred_np, cmap='viridis'); axes[3, i + 1].axis('on')

    axes[4, 0].axis('off')
    for i in range(num_samples):
        with torch.no_grad():
            gen_out = generator(angle_gt).cpu().squeeze().numpy()
        gen_imgs.append(gen_out)
        axes[4, i + 1].imshow(np.rot90(gen_out, k=-1), 'viridis', vmin=-3.5, vmax=3.5); axes[4, i + 1].axis('on')

    axes[5, 0].imshow(angle_img, cmap='viridis'); axes[5, 0].axis('on')
    for i in range(num_samples):
        pred_angle = angle_regressor(torch.tensor(gen_imgs[i]).unsqueeze(0).unsqueeze(0).to(device))
        pred_np = np.rot90(pred_angle.squeeze().detach().cpu().numpy(), k=-1)
        gen_regr.append(pred_np)
        axes[5, i + 1].imshow(pred_np, cmap='viridis'); axes[5, i + 1].axis('on')

    plt.tight_layout(); plt.savefig(save_path, format='pdf', dpi=300); print(f"✅ Saved {save_path}")

    # metrics table
    all_results = []
    methods = ['DSSLA', 'FFTMA', 'Gen']
    regr_sets = [dssla_regr, fftma_regr, gen_regr]
    for method, preds in zip(methods, regr_sets):
        for i, pred in enumerate(preds):
            mse_val, ssim_val, corr_val = compute_metrics(angle_img, pred)
            all_results.append({'Method': method, 'Sample': i + 1, 'MSE': mse_val, 'SSIM': ssim_val, 'Corr': corr_val})
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    # Save flattened samples for reproducibility
    def flatten_samples(samples):
        return [img.flatten() for img in samples]
    all_flat = np.vstack(flatten_samples(dssla_imgs) + flatten_samples(fftma_imgs) + flatten_samples(gen_imgs))
    out_csv = f"samples_idx_{idx}.csv"
    pd.DataFrame(all_flat).to_csv(out_csv, index=False, header=False)
    print(f"✅ Saved samples to: {out_csv}")

# ------------- CLI -------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Dataset CSV with angle and FFT-MA fields concatenated.")
    parser.add_argument("--gen", required=True, help="Path to trained Generator .pth")
    parser.add_argument("--angle_reg", required=True, help="Path to trained Angle Regression .pth")
    parser.add_argument("--indices", type=int, nargs="+", default=[121, 241, 281, 301, 341])
    parser.add_argument("--single_idx", type=int, default=241)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = CustomDataset(args.csv)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    generator = Generator().to(device)
    angle_reg_net = AngleRegressionNet().to(device)

    load_state_safely(generator, args.gen)
    load_state_safely(angle_reg_net, args.angle_reg)

    fftma_model = BatchFFTMA(size=(121, 121), device=device).to(device)

    visualize_comparison_grid(fftma_model, generator, dataset, args.indices, device, out_pdf=os.path.join(args.outdir, "comparison_grid.pdf"))
    visualize_and_evaluate_single_index(fftma_model, generator, angle_reg_net, dataset, args.single_idx, device, save_path=os.path.join(args.outdir, "comparison_single.pdf"))
