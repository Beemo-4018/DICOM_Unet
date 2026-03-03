import pydicom
import matplotlib.pyplot as plt
import numpy as np

# ── 1단계: 샘플 파일 목록 확인 후 로드 ──────────────────────────
files = pydicom.data.get_testdata_files()
print("=== 사용 가능한 샘플 파일 ===")
for f in files:
    print(f)

# CT 파일 로드
path = pydicom.data.get_testdata_file("CT_small.dcm")
ds = pydicom.dcmread(path)

# ── 2단계: Transfer Syntax & UID 확인 ───────────────────────────
print("\n=== Transfer Syntax ===")
print(ds.file_meta.TransferSyntaxUID)

print("\n=== UID 계층 구조 ===")
print("StudyInstanceUID :", ds.StudyInstanceUID)
print("SeriesInstanceUID:", ds.SeriesInstanceUID)
print("SOPInstanceUID   :", ds.SOPInstanceUID)

# ── 3단계: 주요 메타데이터 출력 ─────────────────────────────────
print("\n=== 메타데이터 ===")
print("환자 이름    :", ds.get("PatientName", "없음"))
print("Modality     :", ds.get("Modality", "없음"))
print("픽셀 간격    :", ds.get("PixelSpacing", "없음"))
print("슬라이스 두께:", ds.get("SliceThickness", "없음"))
print("Window Center:", ds.get("WindowCenter", "없음"))
print("Window Width :", ds.get("WindowWidth", "없음"))

# ── 4단계: 픽셀 데이터 + HU값 변환 ─────────────────────────────
print("\n=== 픽셀 데이터 ===")
pixel_array = ds.pixel_array
print("shape:", pixel_array.shape)
print("픽셀 최솟값/최댓값:", pixel_array.min(), "/", pixel_array.max())

# HU 변환
rescale_slope     = float(ds.get("RescaleSlope", 1))
rescale_intercept = float(ds.get("RescaleIntercept", 0))
hu_image = pixel_array * rescale_slope + rescale_intercept
print("HU 범위:", hu_image.min(), "/", hu_image.max())

# ── 5단계: Windowing 함수 ────────────────────────────────────────
def apply_window(image, center, width):
    low  = center - width / 2
    high = center + width / 2
    windowed = np.clip(image, low, high)
    windowed = (windowed - low) / (high - low)
    return windowed

lung = apply_window(hu_image, center=-600, width=1500)
bone = apply_window(hu_image, center=400,  width=1800)

# ── 6단계: 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].set_title("Raw Pixel")
axes[0].imshow(pixel_array, cmap="gray")

axes[1].set_title("HU Image")
axes[1].imshow(hu_image, cmap="gray", vmin=-1000, vmax=1000)

axes[2].set_title("Lung Window\n(C:-600 W:1500)")
axes[2].imshow(lung, cmap="gray")

axes[3].set_title("Bone Window\n(C:400 W:1800)")
axes[3].imshow(bone, cmap="gray")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("dicom_result.png", dpi=150)
plt.show()
print("\n=== dicom_result.png 저장 완료 ===")