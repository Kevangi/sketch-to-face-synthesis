import torch
import torch.nn as nn


# helpers 
def _norm(ch: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)

def _sn(layer: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(layer)


# Generator blocks
class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1,
                            bias=not normalize)]
        if normalize:
            layers.append(_norm(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1,
                                     bias=False),
                  _norm(out_ch)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# U-Net Generator
class UNetGenerator(nn.Module):
    """
    U-Net generator for 256x256 sketch-to-face.

    Encoder (8 blocks):
        e1:   3->ngf      no norm   256->128
        e2:   ngf->ngf*2            128->64
        e3:   ngf*2->ngf*4           64->32
        e4:   ngf*4->ngf*8           32->16
        e5-7: ngf*8->ngf*8           16->2
        e8:   ngf*8->ngf*8  no norm   2->1  (bottleneck)

    Decoder mirrors with skip connections; dropout=0.5 on top 3 blocks.

    Args:
        in_ch  : input channels  (default 3)
        out_ch : output channels (default 3)
        ngf    : base filter count (default 64)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64):
        super().__init__()
        mf = ngf * 8

        self.e1 = _EncBlock(in_ch, ngf,    normalize=False)
        self.e2 = _EncBlock(ngf,   ngf*2)
        self.e3 = _EncBlock(ngf*2, ngf*4)
        self.e4 = _EncBlock(ngf*4, mf)
        self.e5 = _EncBlock(mf,    mf)
        self.e6 = _EncBlock(mf,    mf)
        self.e7 = _EncBlock(mf,    mf)
        self.e8 = _EncBlock(mf,    mf,     normalize=False)

        self.d1 = _DecBlock(mf,    mf,     dropout=0.5)
        self.d2 = _DecBlock(mf*2,  mf,     dropout=0.5)
        self.d3 = _DecBlock(mf*2,  mf,     dropout=0.5)
        self.d4 = _DecBlock(mf*2,  mf)
        self.d5 = _DecBlock(mf*2,  ngf*4)
        self.d6 = _DecBlock(ngf*8, ngf*2)
        self.d7 = _DecBlock(ngf*4, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_ch, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.final(torch.cat([d7, e1], 1))


# PatchGAN Discriminator 
class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN with SpectralNorm.

    Returns (logits, features) so the training notebook can compute
    feature matching loss without a second forward pass.

    Args:
        in_ch : input channels (default 6 = 3 sketch + 3 photo)
        ndf   : base filter count (default 64)
    """

    def __init__(self, in_ch: int = 6, ndf: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            _sn(nn.Conv2d(in_ch,  ndf,    4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            _sn(nn.Conv2d(ndf,    ndf*2,  4, stride=2, padding=1, bias=False)),
            _norm(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            _sn(nn.Conv2d(ndf*2,  ndf*4,  4, stride=2, padding=1, bias=False)),
            _norm(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4 = nn.Sequential(
            _sn(nn.Conv2d(ndf*4,  ndf*8,  4, stride=1, padding=1, bias=False)),
            _norm(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = _sn(nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1))

    def forward(self, sketch, photo):
        """
        Returns
            logits   : (B, 1, 30, 30) raw patch predictions
            features : [f1, f2, f3, f4] intermediate tensors for FM loss
        """
        x  = torch.cat([sketch, photo], dim=1)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return self.out(f4), [f1, f2, f3, f4]
