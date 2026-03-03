# 🏥 Medical Image Denoising with UNet & DICOM

> 실제 CT DICOM 데이터를 활용한 딥러닝 노이즈 제거 파이프라인

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![pydicom](https://img.shields.io/badge/pydicom-latest-green)

---

## 📌 프로젝트 개요

의료 AI 분야 취업 준비를 위해 DICOM 표준부터 딥러닝 모델 학습까지 전체 파이프라인을 직접 구현한 프로젝트입니다.

실제 흉부 CT 데이터(189 슬라이스)를 사용하여 DICOM 데이터 처리 → 3D 볼륨 구성 → PyTorch DataLoader → UNet 학습까지 end-to-end로 구성했습니다.

---

## 🗂️ 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [FUMPE Dataset - figshare](https://figshare.com/collections/FUMPE/4107803) |
| 환자 | Patient 34 (흉부 CT) |
| 슬라이스 수 | 189장 |
| 해상도 | 512 × 512 |
| Modality | CT (HELICAL, AXIAL) |
| Transfer Syntax | Explicit VR Little Endian |
| HU 범위 | -1024 ~ 3071 |

---

## 🔧 기술 스택

- **언어**: Python 3.10
- **딥러닝**: PyTorch (MPS GPU 활용)
- **데이터 처리**: pydicom, NumPy
- **시각화**: Matplotlib
- **모델**: UNet (31M 파라미터)

---

## 📁 프로젝트 구조

```
dicom/
├── PAT034/                  # DICOM 파일 (189개 슬라이스)
│   ├── D0001.dcm
│   ├── D0002.dcm
│   └── ...
├── prac_dicom.py            # DICOM 기초 탐색 (Transfer Syntax, UID, 메타데이터)
├── 3d_dataload.py           # 3D 볼륨 구성 & 3방향 단면 시각화
├── pytorch_dataloader.py    # PyTorch Dataset & DataLoader 구성
├── unet.py                  # UNet 아키텍처 구현
└── denoising.py             # 학습 파이프라인 (노이즈 제거)
```

---

## 🧠 모델 아키텍처 - UNet

```
입력 (B, 1, 512, 512)
    │
    ├── Encoder
    │     enc1: (1 → 64)   + MaxPool
    │     enc2: (64 → 128) + MaxPool
    │     enc3: (128 → 256) + MaxPool
    │     enc4: (256 → 512) + MaxPool
    │
    ├── Bottleneck: (512 → 1024)
    │
    ├── Decoder (Skip Connection)
    │     dec4: up(1024→512) + cat(enc4) → 512
    │     dec3: up(512→256)  + cat(enc3) → 256
    │     dec2: up(256→128)  + cat(enc2) → 128
    │     dec1: up(128→64)   + cat(enc1) → 64
    │
    └── 출력 (B, 1, 512, 512)  ← Sigmoid
```

Skip Connection을 통해 Encoder의 공간 정보를 Decoder에 전달하여 세밀한 복원이 가능합니다.

---

## 🔄 학습 파이프라인

### 1단계 - DICOM 로드 & HU 변환
```python
ds = pydicom.dcmread("D0001.dcm")
hu = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
```

### 2단계 - Windowing (0~1 정규화)
```python
def apply_window(image, center=40, width=400):
    low  = center - width / 2
    high = center + width / 2
    return (np.clip(image, low, high) - low) / (high - low)
```

### 3단계 - 3D 볼륨 구성
```python
# (189, 512, 512) 볼륨
slices.sort(key=lambda x: float(x.SliceLocation))
volume = np.stack([apply_window(s.pixel_array * ...) for s in slices], axis=0)
```

### 4단계 - DataLoader
```python
dataset    = DenoisingDataset(dicom_dir, noise_std=0.05)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# 배치 shape: (4, 1, 512, 512)
```

### 5단계 - 학습 (Denoising)
- 입력: 원본 + 가우시안 노이즈 (std=0.05)
- 정답: 원본 이미지
- Loss: MSE Loss
- Optimizer: Adam (lr=1e-4)
- Device: Apple MPS (GPU)

---

## 📊 학습 결과

### Loss 곡선

| Epoch | 평균 MSE Loss |
|-------|--------------|
| 1     | 0.043266     |
| 2     | 0.016364     |
| 3     | 0.012760     |
| 4     | 0.010583     |
| 5     | 0.009024     |

5 에폭 만에 Loss가 **0.043 → 0.009**로 약 79% 감소했습니다.

---

## 💡 핵심 학습 내용

### DICOM 구조 이해
- **Tag/VR/UID 체계**: 모든 메타데이터는 `(Group, Element)` 태그로 식별
- **Transfer Syntax**: 데이터 인코딩 방식 (`1.2.840.10008.1.2.1` = Explicit VR Little Endian)
- **UID 계층**: `StudyInstanceUID > SeriesInstanceUID > SOPInstanceUID`
- **HU (Hounsfield Unit)**: CT 픽셀값의 물리적 의미 (공기=-1000, 물=0, 뼈=+400~1000)

### Windowing
같은 CT 이미지도 Center/Width 값에 따라 전혀 다른 정보가 보임

| 윈도우 | Center | Width | 용도 |
|--------|--------|-------|------|
| Brain  | 40     | 80    | 뇌 조직 |
| Lung   | -600   | 1500  | 폐/공기 |
| Bone   | 400    | 1800  | 뼈 구조 |
| Abdomen| 60     | 400   | 복부 장기 |

### 3D 볼륨
- 2D 슬라이스 → `np.stack()` → `(189, 512, 512)` 3D 볼륨
- SliceLocation 기준 정렬로 공간적 연속성 보장
- Axial / Coronal / Sagittal 3방향 단면 시각화

---

## 🚀 실행 방법

```bash
# 환경 설치
pip install pydicom torch numpy matplotlib

# 1. DICOM 기초 탐색
python prac_dicom.py

# 2. 3D 볼륨 시각화
python 3d_dataload.py

# 3. DataLoader 확인
python pytorch_dataloader.py

# 4. UNet 구조 확인
python unet.py

# 5. 노이즈 제거 학습
python denoising.py
```

---

## 📚 참고 자료

- [DICOM 3.0의 기초 - 대한의학영상정보학회](https://ksiim.org/api/society/journal/download/123/157.pdf)
- [FUMPE Dataset](https://figshare.com/collections/FUMPE/4107803)
- [U-Net 원본 논문 (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- pydicom 공식 문서

---

## 🔜 다음 목표

- [ ] ResNet 기반 분류 모델 추가
- [ ] 3D UNet으로 확장 (볼륨 단위 학습)
- [ ] Validation set 분리 & 정량 평가 (PSNR, SSIM)
- [ ] 데이터 증강 (Augmentation) 적용
