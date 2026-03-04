# 🏥 Medical Image AI Pipeline — CT & MRI 복원

> DICOM 파싱부터 딥러닝 모델 학습까지, 실제 의료 데이터로 구현한 end-to-end 파이프라인

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![pydicom](https://img.shields.io/badge/pydicom-latest-green)
![fastMRI](https://img.shields.io/badge/fastMRI-knee-red)

---

## 📌 프로젝트 개요

의료 AI 분야에서 요구하는 핵심 기술 스택(DICOM, PyTorch, UNet, 3D 처리)을 실제 데이터로 직접 구현한 프로젝트입니다.

CT 노이즈 제거와 MRI 가속화 복원, 두 가지 태스크를 통해 실제 필드에서 쓰는 것과 동일한 파이프라인 구조를 구현했습니다.

---

## 🗂️ 프로젝트 구성

```
medical-image-ai/
│
├── 📁 CT_denoising/               # CT 노이즈 제거 프로젝트
│   ├── prac_dicom.py              # DICOM 기초 탐색
│   ├── 3d_dataload.py             # 3D 볼륨 구성 & 시각화
│   ├── pytorch_dataloader.py      # PyTorch DataLoader
│   ├── unet.py                    # UNet 아키텍처
│   └── denoising.py               # 학습 파이프라인
│
├── 📁 MRI_reconstruction/         # MRI 복원 프로젝트
│   ├── h5_read.py                 # fastMRI h5 파일 탐색
│   ├── unet_pipeline.py           # UNet + MSE Loss
│   └── revision_pipeline.py       # UNet + MSE+SSIM Loss (개선)
│
├── 📁 result_images/  
│
│
└── README.md
```

---

## 🔬 프로젝트 1 — CT 노이즈 제거

### 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [FUMPE Dataset (figshare)](https://figshare.com/collections/FUMPE/4107803) |
| 대상 | Patient 34 흉부 CT |
| 슬라이스 수 | 189장 |
| 해상도 | 512 × 512 |
| Modality | CT (HELICAL, AXIAL) |
| Transfer Syntax | Explicit VR Little Endian |
| HU 범위 | -1024 ~ 3071 |

### 핵심 구현 내용

**DICOM 파싱 & HU 변환**
```python
ds = pydicom.dcmread("D0001.dcm")
hu = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
```

**Windowing (관심 영역 강조)**
```python
def apply_window(image, center=40, width=400):
    low  = center - width / 2
    high = center + width / 2
    return (np.clip(image, low, high) - low) / (high - low)
```
![](https://velog.velcdn.com/images/zlfktk/post/d45f0ba3-e9ff-438c-aca4-fc2579a2c8fd/image.png)

| 윈도우 | Center | Width | 용도 |
|--------|--------|-------|------|
| Brain  | 40     | 80    | 뇌 조직 |
| Lung   | -600   | 1500  | 폐/공기 |
| Bone   | 400    | 1800  | 뼈 구조 |
| Abdomen| 60     | 400   | 복부 장기 |

**3D 볼륨 구성**
```python
slices.sort(key=lambda x: float(x.SliceLocation))
volume = np.stack([apply_window(s.pixel_array * ...) for s in slices], axis=0)
# shape: (189, 512, 512)
```

### 학습 결과

| Epoch | MSE Loss |
|-------|----------|
| 1     | 0.043266 |
| 3     | 0.012760 |
| 5     | 0.009024 |

Loss **0.043 → 0.009** (79% 감소), 5 에폭 만에 안정적 수렴

![](https://velog.velcdn.com/images/zlfktk/post/cb644efe-570a-4c06-9c7c-ace0bad99383/image.png)

![](https://velog.velcdn.com/images/zlfktk/post/957daea5-6bec-4f1c-a861-ce08df599a64/image.png)

---

## 🧠 프로젝트 2 — MRI 복원 (k-space 언더샘플링)

### 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | [fastMRI (NYU Langone)](https://fastmri.org) |
| 대상 | Knee MRI singlecoil_val |
| 파일 수 | 200개 .h5 파일 |
| k-space shape | (35, 640, 368) complex64 |
| 정답 이미지 | reconstruction_rss (35, 320, 320) |
| MRI 시퀀스 | CORPDFS_FBK (Coronal PD Fat Suppressed) |

### 실제 회사들과 동일한 태스크 구조

```
k-space 25% 언더샘플링 (촬영 시간 4배 단축)
            ↓
         UNet
            ↓
    완전 복원 이미지
```

**k-space → 이미지 변환 (역 푸리에 변환)**
```python
def kspace_to_image(kspace_slice):
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_slice)))
    return np.abs(image).astype(np.float32)
```

**언더샘플링 마스크 (가속도 4배)**
```python
def create_undersampling_mask(num_cols, acceleration=4, center_fraction=0.08):
    # 중심부(저주파) 100% 샘플링 + 외곽 랜덤 샘플링
    # → 전체 25% 만으로 MRI 구조 보존
```
![](https://velog.velcdn.com/images/zlfktk/post/31cff0c4-71c3-481c-b5c5-ce529dd7ac86/image.png)


### Loss 함수 개선 과정

**v1: MSE Loss만 사용 → 흐릿한 결과**

MSE는 픽셀 평균 오차를 최소화하므로, 불확실한 영역에서 모델이 "평균값"을 출력하게 되어 엣지가 뭉개지는 현상 발생

**v2: MSE + SSIM Combined Loss → 선명도 개선**

```python
class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        return 0.5 * mse(pred, target) + 0.5 * ssim_loss(pred, target)
```

SSIM은 밝기/대비/구조를 동시에 고려하여 엣지 복원에 효과적

### 정량 평가 결과

| 모델 | PSNR | SSIM |
|------|------|------|
| 언더샘플링 입력 (베이스라인) | 22.54 dB | 0.5563 |
| UNet + MSE Loss | 22.64 dB | 0.5657 |
| **UNet + MSE+SSIM Loss** | **24.58 dB** | **0.6751** |

Loss 함수 개선만으로 PSNR **+2.04 dB**, SSIM **+0.119** 향상

---

## 🏗️ 공통 모델 아키텍처 — UNet

```
입력 (B, 1, 320, 320)
    │
    ├── Encoder
    │     enc1: Conv(1→32)   + MaxPool
    │     enc2: Conv(32→64)  + MaxPool
    │     enc3: Conv(64→128) + MaxPool
    │
    ├── Bottleneck: Conv(128→256)
    │
    ├── Decoder (Skip Connection)
    │     dec3: up(256→128) + cat(enc3) → 128
    │     dec2: up(128→64)  + cat(enc2) → 64
    │     dec1: up(64→32)   + cat(enc1) → 32
    │
    └── 출력 (B, 1, 320, 320) ← Sigmoid
```

Skip Connection으로 Encoder의 공간 정보를 Decoder에 전달하여 세밀한 복원 가능

---

## 🚀 실행 방법

```bash
# 환경 설치
pip install pydicom torch numpy matplotlib h5py scikit-image

# CT 프로젝트
cd CT_denoising
python prac_dicom.py        # DICOM 탐색
python 3d_dataload.py       # 3D 볼륨 시각화
python denoising.py         # 학습

# MRI 프로젝트
cd MRI_reconstruction
python h5_read.py           # h5 구조 탐색
python revision_pipeline.py # 학습 (MSE+SSIM Loss)
```

---

## 💡 핵심 학습 내용

**DICOM 구조**
- Tag/VR/UID 체계, Transfer Syntax, IOD 계층 구조
- HU (Hounsfield Unit) 변환 및 Windowing

**MRI 특성**
- k-space (주파수 공간) 개념 및 역 푸리에 변환
- 언더샘플링 아티팩트와 복원 원리
- CT(절대적 HU값) vs MRI(상대적 강도값) 차이

**딥러닝**
- UNet Encoder-Decoder + Skip Connection
- MSE Loss의 한계와 SSIM Loss로 개선
- PSNR/SSIM 정량 평가

---

## 🔜 다음 목표

- [ ] ResNet 기반 분류 모델 추가
- [ ] 3D UNet 확장 (볼륨 단위 학습)
- [ ] Validation set 분리 & Early Stopping
- [ ] 더 많은 데이터로 재학습 (max_files 증가)

fastMRI의 reconstruction_rss를 정답으로 학습했지만, 실제 임상 환경에서는 fully-sampled ground truth가 존재하지 않는 경우가 많다.
Self-supervised reconstruction (e.g., data consistency 기반 학습) 방향도 추가로 탐구해보고 싶다.
---

## 📚 참고 자료

- [DICOM 3.0의 기초 — 대한의학영상정보학회](https://ksiim.org/api/society/journal/download/123/157.pdf)
- [fastMRI: An Open Dataset and Benchmarks for Accelerated MRI (Zbontar et al., 2018)](https://arxiv.org/abs/1811.08839)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [FUMPE Dataset (figshare)](https://figshare.com/collections/FUMPE/4107803)
