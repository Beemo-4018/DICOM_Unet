import pydicom
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Mac 한글 폰트 설정 ──────────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 파일 로드 & HU 변환
path = pydicom.data.get_testdata_file("CT_small.dcm")
ds = pydicom.dcmread(path)

pixel_array = ds.pixel_array
rescale_slope     = float(ds.get("RescaleSlope", 1))
rescale_intercept = float(ds.get("RescaleIntercept", 0))
hu_image = pixel_array * rescale_slope + rescale_intercept

# ── Windowing 함수 ──────────────────────────────────────────────
def apply_window(image, center, width):
    low  = center - width / 2
    high = center + width / 2
    windowed = np.clip(image, low, high)
    windowed = (windowed - low) / (high - low)
    return windowed

# ── 윈도우 프리셋 정의 ──────────────────────────────────────────
presets = [
    ("뇌  (Brain)  ",    40,   80),
    ("폐  (Lung)   ",  -600, 1500),
    ("뼈  (Bone)   ",   400, 1800),
    ("복부 (Abdomen)",    60,  400),
    ("넓게 (Wide)  ",     40, 4000),
    ("좁게 (Narrow)",     40,  100),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (name, center, width) in enumerate(presets):
    windowed = apply_window(hu_image, center, width)
    axes[i].imshow(windowed, cmap="gray")
    axes[i].set_title(f"{name}\nC:{center}  W:{width}", fontsize=11)
    axes[i].axis("off")

plt.suptitle("CT Windowing 비교", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("windowing_compare.png", dpi=150)
plt.show()
print("windowing_compare.png 저장 완료")