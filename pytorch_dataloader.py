import pydicom
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

# ── 1. 전처리 함수 ───────────────────────────────────────────────
def apply_window(image, center=40, width=400):
    low  = center - width / 2
    high = center + width / 2
    windowed = np.clip(image, low, high)
    windowed = (windowed - low) / (high - low)  # 0~1 정규화
    return windowed.astype(np.float32)

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
        volume.append(hu)
    return np.stack(volume, axis=0)  # (189, 512, 512)

# ── 2. Dataset 클래스 ────────────────────────────────────────────
class CTSliceDataset(Dataset):
    """
    DICOM 볼륨에서 2D 슬라이스를 하나씩 꺼내주는 Dataset
    실제 모델 학습 시 이 구조를 기반으로 확장함
    """
    def __init__(self, dicom_dir, center=40, width=400):
        self.volume = load_volume(dicom_dir)        # (189, 512, 512)
        self.center = center
        self.width  = width
        print(f"볼륨 로드 완료: {self.volume.shape}")

    def __len__(self):
        return self.volume.shape[0]  # 슬라이스 수 = 189

    def __getitem__(self, idx):
        slice_hu = self.volume[idx]                          # (512, 512)
        slice_w  = apply_window(slice_hu, self.center, self.width)  # 0~1 정규화
        tensor   = torch.tensor(slice_w).unsqueeze(0)       # (1, 512, 512) 채널 추가
        return tensor

# ── 3. DataLoader 생성 ───────────────────────────────────────────
dicom_dir = "/Users/admin/Downloads/dicom/PAT034"

dataset    = CTSliceDataset(dicom_dir, center=40, width=400)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

print(f"\nDataset 크기: {len(dataset)}장")
print(f"총 배치 수  : {len(dataloader)}개 (batch_size=8 기준)")

# ── 4. 배치 하나 꺼내서 확인 ─────────────────────────────────────
batch = next(iter(dataloader))
print(f"\n배치 shape : {batch.shape}")   # (8, 1, 512, 512)
print(f"값 범위    : {batch.min():.3f} ~ {batch.max():.3f}")  # 0~1 사이여야 정상
print(f"dtype      : {batch.dtype}")

# ── 5. 배치 시각화 ───────────────────────────────────────────────
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    axes[i].imshow(batch[i, 0].numpy(), cmap="gray")
    axes[i].set_title(f"슬라이스 {i+1}")
    axes[i].axis("off")

plt.suptitle("DataLoader 배치 확인 (batch_size=8)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("dataloader_batch.png", dpi=150)
plt.show()
print("\ndataloader_batch.png 저장 완료")