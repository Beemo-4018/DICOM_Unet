import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 전처리 함수 ───────────────────────────────────────────────
def kspace_to_image(kspace_slice):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice)))
    return np.abs(image).astype(np.float32)

def create_undersampling_mask(num_cols, acceleration=4, center_fraction=0.08):
    num_low_freqs = int(round(num_cols * center_fraction))
    mask = np.zeros(num_cols, dtype=np.float32)
    pad  = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 1
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = mask + (1 - mask) * (np.random.uniform(size=num_cols) < prob)
    return mask.astype(np.float32)

def normalize(image):
    min_val, max_val = image.min(), image.max()
    if max_val - min_val < 1e-8:
        return None
    return (image - min_val) / (max_val - min_val)

def center_crop(image, size=320):
    h, w = image.shape
    if h < size or w < size:
        padded = np.zeros((size, size), dtype=image.dtype)
        ph, pw = min(h, size), min(w, size)
        padded[:ph, :pw] = image[:ph, :pw]
        return padded
    sh = (h - size) // 2
    sw = (w - size) // 2
    return image[sh:sh+size, sw:sw+size]

# ── 2. Dataset ───────────────────────────────────────────────────
class FastMRIDataset(Dataset):
    def __init__(self, data_dir, acceleration=4, max_files=20):
        self.samples = []
        skipped = 0

        files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".h5")
        ])[:max_files]

        print(f"로드할 파일 수: {len(files)}")

        for fpath in files:
            with h5py.File(fpath, "r") as f:
                kspace = f['kspace'][()]
                target = f['reconstruction_rss'][()]

            for i in range(kspace.shape[0]):
                mask         = create_undersampling_mask(kspace.shape[-1], acceleration)
                kspace_under = kspace[i] * mask
                under_img    = center_crop(kspace_to_image(kspace_under), 320)
                target_img   = center_crop(target[i], 320)

                under_norm  = normalize(under_img)
                target_norm = normalize(target_img)

                if under_norm is None or target_norm is None:
                    skipped += 1
                    continue

                self.samples.append((under_norm, target_norm))

        print(f"총 슬라이스 수: {len(self.samples)} (필터링: {skipped}개 제거)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        under, target = self.samples[idx]
        return (torch.tensor(under).unsqueeze(0),
                torch.tensor(target).unsqueeze(0))

# ── 3. UNet ──────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ── 4. SSIM Loss + Combined Loss ─────────────────────────────────
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window      = self._create_window(window_size)

    def _gaussian(self, window_size, sigma=1.5):
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        return gauss / gauss.sum()

    def _create_window(self, window_size):
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d

    def forward(self, pred, target):
        window = self.window.to(pred.device)
        mu1    = F.conv2d(pred,   window, padding=self.window_size//2)
        mu2    = F.conv2d(target, window, padding=self.window_size//2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = F.conv2d(pred*pred,     window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=self.window_size//2) - mu2_sq
        sigma12   = F.conv2d(pred*target,   window, padding=self.window_size//2) - mu1_mu2
        C1, C2   = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.ssim  = SSIMLoss()

    def forward(self, pred, target):
        return (1 - self.alpha) * self.mse(pred, target) + \
                    self.alpha  * self.ssim(pred, target)

# ── 5. 학습 설정 ─────────────────────────────────────────────────
data_dir = "/Users/admin/Downloads/singlecoil_val"
device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"디바이스: {device}")

dataset    = FastMRIDataset(data_dir, acceleration=4, max_files=20)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model     = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = CombinedLoss(alpha=0.5)  # ← 여기가 핵심 변경점

EPOCHS   = 10
loss_log = []

# ── 6. 학습 루프 ─────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (under, target) in enumerate(dataloader):
        under, target = under.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(under)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}")

    avg = epoch_loss / len(dataloader)
    loss_log.append(avg)
    print(f"\n▶ Epoch {epoch+1} 평균 Loss: {avg:.6f}\n")

# ── 7. PSNR / SSIM 평가 ──────────────────────────────────────────
model.eval()
psnr_input_list, psnr_pred_list = [], []
ssim_input_list, ssim_pred_list = [], []

with torch.no_grad():
    for under, target in dataloader:
        pred = model(under.to(device)).cpu()
        for i in range(under.shape[0]):
            u = under[i, 0].numpy()
            t = target[i, 0].numpy()
            p = pred[i, 0].numpy()
            psnr_input_list.append(psnr(t, u, data_range=1.0))
            psnr_pred_list.append(psnr(t, p, data_range=1.0))
            ssim_input_list.append(ssim(t, u, data_range=1.0))
            ssim_pred_list.append(ssim(t, p, data_range=1.0))

print("\n=== 정량 평가 결과 ===")
print(f"PSNR  언더샘플링 입력: {np.mean(psnr_input_list):.2f} dB")
print(f"PSNR  모델 예측      : {np.mean(psnr_pred_list):.2f} dB")
print(f"SSIM  언더샘플링 입력: {np.mean(ssim_input_list):.4f}")
print(f"SSIM  모델 예측      : {np.mean(ssim_pred_list):.4f}")

# ── 8. 시각화 ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), loss_log, marker='o')
plt.title("MRI 복원 Loss 곡선 (MSE + SSIM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mri_loss_v4.png", dpi=150)
plt.show()

with torch.no_grad():
    under_batch, target_batch = next(iter(dataloader))
    pred_batch = model(under_batch.to(device)).cpu()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
row_titles = ["언더샘플링 입력", "정답 (완전 복원)", "모델 예측"]
data       = [under_batch, target_batch, pred_batch]

for row, (title, imgs) in enumerate(zip(row_titles, data)):
    for col in range(4):
        axes[row][col].imshow(imgs[col, 0].numpy(), cmap='gray')
        axes[row][col].set_title(f"{title} {col+1}", fontsize=9)
        axes[row][col].axis('off')

plt.suptitle("MRI 복원 결과 v4 (MSE + SSIM Loss)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("mri_result_v4.png", dpi=150)
plt.show()

torch.save(model.state_dict(), "unet_mri_v4.pth")
print("완료! unet_mri_v4.pth 저장됨")