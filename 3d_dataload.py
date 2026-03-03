import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 슬라이스 전체 로드 ────────────────────────────────────────
dicom_dir = "/Users/admin/Downloads/dicom/PAT034"

slices = []
for fname in sorted(os.listdir(dicom_dir)):
    if fname.endswith(".dcm"):
        path = os.path.join(dicom_dir, fname)
        ds = pydicom.dcmread(path)
        slices.append(ds)

print(f"총 슬라이스 수: {len(slices)}")

# ── 2. SliceLocation 기준으로 정렬 ──────────────────────────────
# 슬라이스를 공간적으로 올바른 순서로 정렬
try:
    slices.sort(key=lambda x: float(x.SliceLocation))
    print("SliceLocation 기준 정렬 완료")
except AttributeError:
    slices.sort(key=lambda x: float(x.InstanceNumber))
    print("InstanceNumber 기준 정렬 완료")

# ── 3. 3D 볼륨으로 쌓기 ─────────────────────────────────────────
pixel_arrays = []
for s in slices:
    hu = s.pixel_array * float(s.get("RescaleSlope", 1)) + float(s.get("RescaleIntercept", 0))
    pixel_arrays.append(hu)

volume = np.stack(pixel_arrays, axis=0)
print(f"3D 볼륨 shape: {volume.shape}")  # (슬라이스수, H, W)
print(f"HU 범위: {volume.min():.1f} ~ {volume.max():.1f}")

# ── 4. 3방향 단면 시각화 ─────────────────────────────────────────
def apply_window(image, center, width):
    low  = center - width / 2
    high = center + width / 2
    return (np.clip(image, low, high) - low) / (high - low)

mid_z = volume.shape[0] // 2  # 축면 (Axial)
mid_y = volume.shape[1] // 2  # 관상면 (Coronal)
mid_x = volume.shape[2] // 2  # 시상면 (Sagittal)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(apply_window(volume[mid_z], 40, 400), cmap="gray")
axes[0].set_title(f"Axial (슬라이스 {mid_z})")
axes[0].axis("off")

axes[1].imshow(apply_window(volume[:, mid_y, :], 40, 400), cmap="gray")
axes[1].set_title(f"Coronal (y={mid_y})")
axes[1].axis("off")

axes[2].imshow(apply_window(volume[:, :, mid_x], 40, 400), cmap="gray")
axes[2].set_title(f"Sagittal (x={mid_x})")
axes[2].axis("off")

plt.suptitle("3D CT 볼륨 - 3방향 단면", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("volume_3view.png", dpi=150)
plt.show()
print("volume_3view.png 저장 완료")