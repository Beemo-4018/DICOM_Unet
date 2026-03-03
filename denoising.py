import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 전처리 함수 ──────────────────────────────────────────────────
def apply_window(image, center=40, width=400):
    low  = center - width / 2
    high = center + width / 2
    return ((np.clip(image, low, high) - low) / (high - low)).astype(np.float32)

def load_volume(dicom_dir):
    slices = []
    for fname in sorted(os.listdir(dicom_dir)):
        if fname.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(dicom_dir, fname))
            slices.append(ds)
    try:
        slices.sort(key=lambda x: float(x.SliceLocation))
    except AttributeError:
        slices.sort(key=lambda x: float(x.InstanceNumber))
    volume = []
    for s in slices:
        hu = s.pixel_array * float(s.get("RescaleSlope", 1)) + float(s.get("RescaleIntercept", 0))
        volume.append(apply_window(hu))
    return np.stack(volume, axis=0)

# ── Denoising Dataset ────────────────────────────────────────────
class DenoisingDataset(Dataset):
    """
    입력: 원본 + 가우시안 노이즈 (noisy)
    정답: 원본 (clean)
    → 모델이 노이즈를 제거하도록 학습
    """
    def __init__(self, dicom_dir, noise_std=0.05):
        self.volume    = load_volume(dicom_dir)
        self.noise_std = noise_std
        print(f"볼륨 로드 완료: {self.volume.shape}")

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, idx):
        clean = torch.tensor(self.volume[idx]).unsqueeze(0)  # (1, 512, 512)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0, 1)             # 노이즈 추가
        return noisy, clean  # 입력, 정답

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
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ── 학습 설정 ────────────────────────────────────────────────────
dicom_dir  = "/Users/admin/Downloads/dicom/PAT034"
device     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 디바이스: {device}")  # Mac이면 mps (GPU) 사용

dataset    = DenoisingDataset(dicom_dir, noise_std=0.05)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model      = UNet().to(device)
optimizer  = optim.Adam(model.parameters(), lr=1e-4)
criterion  = nn.MSELoss()  # 노이즈 제거는 MSE Loss 사용

# ── 학습 루프 ────────────────────────────────────────────────────
EPOCHS    = 5
loss_log  = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss   = criterion(output, clean)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}")

    avg_loss = epoch_loss / len(dataloader)
    loss_log.append(avg_loss)
    print(f"\n▶ Epoch {epoch+1} 평균 Loss: {avg_loss:.6f}\n")

# ── 학습 곡선 저장 ───────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), loss_log, marker='o')
plt.title("학습 Loss 곡선")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# ── 결과 시각화 (노이즈 전/후/예측 비교) ────────────────────────
model.eval()
with torch.no_grad():
    sample_noisy, sample_clean = next(iter(dataloader))
    sample_noisy = sample_noisy.to(device)
    sample_pred  = model(sample_noisy).cpu()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
titles = ["노이즈 입력", "정답 (원본)", "모델 예측"]
data   = [sample_noisy.cpu(), sample_clean, sample_pred]

for row, (title, imgs) in enumerate(zip(titles, data)):
    for col in range(4):
        axes[row][col].imshow(imgs[col, 0].numpy(), cmap="gray")
        axes[row][col].set_title(f"{title} {col+1}")
        axes[row][col].axis("off")

plt.suptitle("노이즈 제거 결과 비교", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("denoising_result.png", dpi=150)
plt.show()
print("학습 완료! denoising_result.png 저장됨")

# ── 모델 저장 ────────────────────────────────────────────────────
torch.save(model.state_dict(), "unet_denoising.pth")
print("모델 저장 완료: unet_denoising.pth")