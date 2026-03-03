import torch
import torch.nn as nn

# ── UNet 구성 블록 ───────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv → BN → ReLU 를 2번 반복하는 기본 블록"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# ── UNet 전체 구조 ───────────────────────────────────────────────
class UNet(nn.Module):
    """
    Encoder (수축) → Bottleneck → Decoder (확장)
    Skip Connection으로 Encoder 특징을 Decoder에 전달
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder (다운샘플링)
        self.enc1 = ConvBlock(in_channels, 64)   # (1→64)
        self.enc2 = ConvBlock(64, 128)            # (64→128)
        self.enc3 = ConvBlock(128, 256)           # (128→256)
        self.enc4 = ConvBlock(256, 512)           # (256→512)
        self.pool = nn.MaxPool2d(2)               # 해상도 절반

        # Bottleneck (가장 깊은 곳)
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (업샘플링)
        self.up4    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4   = ConvBlock(1024, 512)   # skip연결로 채널 2배 → 절반

        self.up3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3   = ConvBlock(512, 256)

        self.up2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2   = ConvBlock(256, 128)

        self.up1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1   = ConvBlock(128, 64)

        # 최종 출력 (세그멘테이션 맵)
        self.final  = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # 0~1 확률값으로 변환

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # (B, 64,  512, 512)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 256, 256)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 128, 128)
        e4 = self.enc4(self.pool(e3))  # (B, 512,  64,  64)

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))  # (B, 1024, 32, 32)

        # Decoder + Skip Connection
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # (B, 512, 64, 64)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 256, 128, 128)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 256, 256)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64, 512, 512)

        return self.sigmoid(self.final(d1))  # (B, 1, 512, 512)

# ── 모델 테스트 ──────────────────────────────────────────────────
model = UNet(in_channels=1, out_channels=1)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")

# 더미 입력으로 forward pass 테스트
dummy = torch.randn(2, 1, 512, 512)  # batch=2, channel=1, 512x512
output = model(dummy)
print(f"입력 shape : {dummy.shape}")
print(f"출력 shape : {output.shape}")   # (2, 1, 512, 512) 나오면 성공
print(f"출력 값 범위: {output.min():.3f} ~ {output.max():.3f}")  # 0~1 사이여야 정상