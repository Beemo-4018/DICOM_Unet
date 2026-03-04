import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 함수 ─────────────────────────────────────────────────────────
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
    if max_val - min_val < 1e-6:  # ← 비정상 슬라이스 감지
        return None
    return (image - min_val) / (max_val - min_val)

# ── Dataset (비정상 슬라이스 필터링 추가) ────────────────────────
class FastMRIDataset(Dataset):
    def __init__(self, data_dir, acceleration=4, max_files=20):
        self.samples  = []
        self.skipped  = 0

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
                under_img    = kspace_to_image(kspace_under)
                target_img   = target[i]

                # center crop (640,368) → (320,320)
                h, w = under_img.shape
                sh   = (h - 320) // 2
                sw   = (w - 320) // 2
                under_img = under_img[sh:sh+320, sw:sw+320]

                # 비정상 슬라이스 필터링
                under_norm  = normalize(under_img)
                target_norm = normalize(target_img)

                if under_norm is None or target_norm is None:
                    self.skipped += 1
                    continue

                self.samples.append((under_norm, target_norm))

        print(f"총 슬라이스 수: {len(self.samples)} (필터링: {self.skipped}개 제거)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        under, target = self.samples[idx]
        return (torch.tensor(under).unsqueeze(0),
                torch.tensor(target).unsqueeze(0))

# ── UNet ─────────────────────────────────────────────────────────
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

# ── 학습 ─────────────────────────────────────────────────────────
data_dir = "/Users/admin/Downloads/singlecoil_val"
device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"디바이스: {device}")

dataset    = FastMRIDataset(data_dir, acceleration=4, max_files=20)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model     = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

EPOCHS   = 5
loss_log = []

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
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")

    avg = epoch_loss / len(dataloader)
    loss_log.append(avg)
    print(f"\n▶ Epoch {epoch+1} 평균 Loss: {avg:.6f}\n")

# ── PSNR / SSIM 평가 ─────────────────────────────────────────────
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
print(f"PSNR  모델 예측      : {np.mean(psnr_pred_list):.2f} dB  ← 높을수록 좋음")
print(f"SSIM  언더샘플링 입력: {np.mean(ssim_input_list):.4f}")
print(f"SSIM  모델 예측      : {np.mean(ssim_pred_list):.4f}  ← 1에 가까울수록 좋음")

# ── 시각화 ───────────────────────────────────────────────────────
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

plt.suptitle("MRI 복원 결과 (필터링 + PSNR/SSIM 평가)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("mri_result_v2.png", dpi=150)
plt.show()

torch.save(model.state_dict(), "unet_mri_v2.pth")
print("완료! unet_mri_v2.pth 저장됨")
