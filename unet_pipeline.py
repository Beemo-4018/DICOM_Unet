import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. k-space 변환 & 마스크 함수 ───────────────────────────────
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
    """0~1 정규화"""
    min_val, max_val = image.min(), image.max()
    if max_val - min_val == 0:
        return image
    return (image - min_val) / (max_val - min_val)

# ── 2. MRI Dataset ───────────────────────────────────────────────
class FastMRIDataset(Dataset):
    """
    입력: 언더샘플링된 MRI (k-space 25% 사용)
    정답: 완전 샘플링 MRI
    → CT 때랑 동일한 구조, 데이터만 다름
    """
    def __init__(self, data_dir, acceleration=4, max_files=20):
        self.samples = []  # (kspace_slice, target_slice) 쌍

        files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".h5")
        ])[:max_files]  # 빠른 실습을 위해 20개만 사용

        print(f"로드할 파일 수: {len(files)}")

        for fpath in files:
            with h5py.File(fpath, "r") as f:
                kspace = f['kspace'][()]          # (슬라이스, 640, 368)
                target = f['reconstruction_rss'][()] # (슬라이스, 320, 320)

            for i in range(kspace.shape[0]):
                # 언더샘플링 적용
                mask         = create_undersampling_mask(kspace.shape[-1], acceleration)
                kspace_under = kspace[i] * mask
                under_img    = normalize(kspace_to_image(kspace_under))  # (640, 368)
                target_img   = normalize(target[i])                       # (320, 320)

                # 크기 맞추기 (640,368) → center crop → (320, 320)
                h, w   = under_img.shape
                ch, cw = 320, 320
                sh     = (h - ch) // 2
                sw     = (w - cw) // 2
                under_img = under_img[sh:sh+ch, sw:sw+cw]

                self.samples.append((under_img, target_img))

        print(f"총 슬라이스 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        under, target = self.samples[idx]
        under  = torch.tensor(under).unsqueeze(0)   # (1, 320, 320)
        target = torch.tensor(target).unsqueeze(0)  # (1, 320, 320)
        return under, target

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

# ── 4. 학습 설정 ─────────────────────────────────────────────────
data_dir = "/Users/admin/Downloads/singlecoil_val"
device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"디바이스: {device}")

dataset    = FastMRIDataset(data_dir, acceleration=4, max_files=20)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model     = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {total_params:,}")
print(f"배치 수: {len(dataloader)}")

# ── 5. 학습 루프 ─────────────────────────────────────────────────
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
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}")

    avg = epoch_loss / len(dataloader)
    loss_log.append(avg)
    print(f"\n▶ Epoch {epoch+1} 평균 Loss: {avg:.6f}\n")

# ── 6. Loss 곡선 ─────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), loss_log, marker='o')
plt.title("MRI 복원 학습 Loss 곡선")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mri_loss_curve.png", dpi=150)
plt.show()

# ── 7. 결과 시각화 ───────────────────────────────────────────────
model.eval()
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

plt.suptitle("MRI 복원 결과 비교 (언더샘플링 → 복원)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("mri_result.png", dpi=150)
plt.show()

torch.save(model.state_dict(), "unet_mri.pth")
print("학습 완료! unet_mri.pth 저장됨")