"""
Pix2Pix architecture for sketch-to-face generation.

Contains:
    - UNetGenerator   : U-Net encoder-decoder with skip connections
    - PatchGANDiscriminator : 70x70 PatchGAN discriminator
    - weights_init    : weight initialisation helper

Reference:
    Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks"
    https://arxiv.org/abs/1611.07085

Input/Output:
    Generator input  : sketch tensor (B, 3, 256, 256) in [-1, 1]
    Generator output : fake photo   (B, 3, 256, 256) in [-1, 1]  (tanh)

    Discriminator input : [sketch | photo] concatenated -> (B, 6, 256, 256)
    Discriminator output: patch map (B, 1, 30, 30)
"""

import torch
import torch.nn as nn

# Weight initialisation

def weights_init(m):
    """
    Initialise Conv and BatchNorm layers as per the original Pix2Pix paper:
        Conv weights  : Normal(0, 0.02)
        BatchNorm     : Normal(1.0, 0.02), bias=0
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# U-Net Generator building blocks

class UNetDown(nn.Module):
    """
    Encoder block: Conv -> BatchNorm -> LeakyReLU
    Downsamples by stride=2.
    First encoder block skips BatchNorm as per original paper.
    """

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """
    Decoder block: ConvTranspose -> BatchNorm -> ReLU -> Dropout (optional)
    Skip connection: concatenates encoder feature map from same level.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)   # skip connection


# U-Net Generator

class UNetGenerator(nn.Module):
    """
    U-Net Generator as used in Pix2Pix.

    Architecture (256x256 input):
        Encoder: 8 downsampling blocks  256 -> 1
        Decoder: 8 upsampling blocks    1   -> 256
        Skip connections between encoder and decoder at each level.

    Args:
        in_channels  : input channels  (3 for RGB sketch)
        out_channels : output channels (3 for RGB photo)
        features     : base feature count (default 64)
    """

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # Encoder 
        # No BatchNorm on first encoder block
        self.down1 = UNetDown(in_channels,      features,       normalize=False)  # 256 -> 128
        self.down2 = UNetDown(features,         features * 2)                     # 128 -> 64
        self.down3 = UNetDown(features * 2,     features * 4)                     # 64  -> 32
        self.down4 = UNetDown(features * 4,     features * 8)                     # 32  -> 16
        self.down5 = UNetDown(features * 8,     features * 8)                     # 16  -> 8
        self.down6 = UNetDown(features * 8,     features * 8)                     # 8   -> 4
        self.down7 = UNetDown(features * 8,     features * 8)                     # 4   -> 2
        self.down8 = UNetDown(features * 8,     features * 8,   normalize=False)  # 2   -> 1  (bottleneck)

        # Decoder 
        # Dropout (0.5) on first three decoder blocks as per paper
        # in_channels doubled because of skip connection concatenation
        self.up1 = UNetUp(features * 8,     features * 8,   dropout=0.5)   # 1   -> 2    in: 512
        self.up2 = UNetUp(features * 8 * 2, features * 8,   dropout=0.5)   # 2   -> 4    in: 1024
        self.up3 = UNetUp(features * 8 * 2, features * 8,   dropout=0.5)   # 4   -> 8    in: 1024
        self.up4 = UNetUp(features * 8 * 2, features * 8)                  # 8   -> 16   in: 1024
        self.up5 = UNetUp(features * 8 * 2, features * 4)                  # 16  -> 32   in: 1024
        self.up6 = UNetUp(features * 4 * 2, features * 2)                  # 32  -> 64   in: 512
        self.up7 = UNetUp(features * 2 * 2, features)                      # 64  -> 128  in: 256

        # Final layer — no BatchNorm, no skip concat, tanh output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()                                                       # 128 -> 256, output in [-1,1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)   # bottleneck

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# PatchGAN Discriminator

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator as used in Pix2Pix.

    Classifies overlapping 70x70 patches as real or fake
    rather than the whole image. This encourages high-frequency
    sharpness which is critical for face texture.

    Input: sketch + photo concatenated along channel dim -> (B, 6, H, W)
    Output: patch prediction map (B, 1, 30, 30)

    Args:
        in_channels : input channels — sketch(3) + photo(3) = 6
        features    : base feature count (default 64)
    """

    def __init__(self, in_channels=6, features=64):
        super().__init__()

        # No BatchNorm on first layer
        self.model = nn.Sequential(
            # Layer 1 — no BN
            nn.Conv2d(in_channels, features,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(features, features * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(features * 2, features * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 — stride 1
            nn.Conv2d(features * 4, features * 8,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer — single channel patch map
            nn.Conv2d(features * 8, 1,
                      kernel_size=4, stride=1, padding=1),
        )

    def forward(self, sketch, photo):
        # Concatenate sketch and photo along channel dim
        x = torch.cat([sketch, photo], dim=1)   # (B, 6, H, W)
        return self.model(x)                     # (B, 1, 30, 30)

# Smoke test — python pix2pix.py

if __name__ == '__main__':
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print()

    B = 2   # batch size

    # Dummy sketch and photo tensors
    sketch = torch.randn(B, 3, 256, 256).to(device)
    photo  = torch.randn(B, 3, 256, 256).to(device)

    # ── Generator ──────────────────────────────────────────────────
    G = UNetGenerator(in_channels=3, out_channels=3, features=64).to(device)
    G.apply(weights_init)

    fake_photo = G(sketch)

    print(f'Generator')
    print(f'  Input  : {tuple(sketch.shape)}')
    print(f'  Output : {tuple(fake_photo.shape)}')
    print(f'  Range  : [{fake_photo.min():.3f}, {fake_photo.max():.3f}]  (should be in [-1,1] via tanh)')
    print(f'  Params : {sum(p.numel() for p in G.parameters()):,}')
    print()

    assert fake_photo.shape == (B, 3, 256, 256), f'Generator output shape wrong: {fake_photo.shape}'
    assert fake_photo.min() >= -1.01 and fake_photo.max() <= 1.01, 'Generator output out of [-1,1]'

    # Discriminator 
    D = PatchGANDiscriminator(in_channels=6, features=64).to(device)
    D.apply(weights_init)

    # Real pair
    real_pred = D(sketch, photo)
    # Fake pair
    fake_pred = D(sketch, fake_photo.detach())

    print(f'Discriminator')
    print(f'  Input  : sketch {tuple(sketch.shape)} + photo {tuple(photo.shape)}')
    print(f'  Output : {tuple(real_pred.shape)}  (patch map)')
    print(f'  Params : {sum(p.numel() for p in D.parameters()):,}')
    print()

    assert real_pred.shape == (B, 1, 30, 30), f'Discriminator output shape wrong: {real_pred.shape}'
    assert fake_pred.shape == (B, 1, 30, 30), f'Discriminator fake output shape wrong: {fake_pred.shape}'

    print('All assertions passed.')
