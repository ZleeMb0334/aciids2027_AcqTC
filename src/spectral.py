"""Spectral instrumentation: residuals, log spectra, plus band profiles.

The band profile phi(x) supports post-hoc sensitivity measurement.  The paper
reports effective band count from those sensitivities.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- SRM residual
# The three high-pass SRM kernels used as a fixed residual front-end in image
# forensics (Fridrich & Kodovsky 2012; used by Zhou et al. 2017, Luo et al. 2021).
_SRM_RAW = [
    # 1st order 3x3 (embedded in 5x5)
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    # 2nd order 3x3
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    # 3rd order 5x5 (SQUARE 5x5)
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
]
_SRM_NORM = [2.0, 4.0, 12.0]


def srm_kernels() -> torch.Tensor:
    """(3, 1, 5, 5) normalised SRM high-pass kernels."""
    k = torch.tensor(_SRM_RAW, dtype=torch.float32)
    k = k / torch.tensor(_SRM_NORM).view(-1, 1, 1)
    return k.unsqueeze(1)


class ResidualExtractor(nn.Module):
    """x -> high-pass residual. mode in {srm, gauss, none}."""

    def __init__(self, mode: str = "srm"):
        super().__init__()
        self.mode = mode
        if mode == "srm":
            k = srm_kernels()                       # (3,1,5,5)
            k = k.repeat_interleave(3, dim=0)       # (9,1,5,5): 3 filters x 3 colour ch
            self.register_buffer("w", k)
        elif mode == "gauss":
            g = torch.tensor([1., 4., 6., 4., 1.])
            g2 = (g[:, None] * g[None, :]) / 256.0
            self.register_buffer("w", g2.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1]. returns (B,C,H,W) residual."""
        if self.mode == "none":
            return x
        if self.mode == "gauss":
            low = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.w, groups=3)
            return x - low
        # srm: apply each of the 3 kernels to each of the 3 channels -> 9 maps
        xp = F.pad(x, (2, 2, 2, 2), mode="reflect")
        out = [F.conv2d(xp[:, c:c + 1], self.w[c * 3:(c + 1) * 3][:, :1]) for c in range(3)]
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------- log spectrum
def log_spectrum(r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,C,H,W) -> (B,C,H,W) fftshifted log magnitude spectrum."""
    F_ = torch.fft.fft2(r.float())
    F_ = torch.fft.fftshift(F_, dim=(-2, -1))
    return torch.log(F_.abs() + eps)


# ---------------------------------------------------------------- band profile
class BandProfile(nn.Module):
    """Radial x angular binning of a spectrum into B = n_r * n_a bands.

    Returns phi in R^{B}: the mean log-magnitude inside each band. The band
    partition is a fixed function of image size.  This makes phi comparable
    across images.
    """

    def __init__(self, size: int = 224, n_r: int = 32, n_a: int = 2):
        super().__init__()
        self.n_r, self.n_a, self.B = n_r, n_a, n_r * n_a
        yy, xx = torch.meshgrid(torch.arange(size) - size // 2,
                                torch.arange(size) - size // 2, indexing="ij")
        rad = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        rad = rad / (rad.max() + 1e-8)
        ang = torch.atan2(yy.float(), xx.float()).abs() / np.pi   # [0,1]
        ridx = (rad * n_r).long().clamp(0, n_r - 1)
        aidx = (ang * n_a).long().clamp(0, n_a - 1)
        band = ridx * n_a + aidx                                   # (H,W) in [0,B)
        self.register_buffer("band", band)
        onehot = F.one_hot(band.flatten(), self.B).float()          # (HW,B)
        self.register_buffer("counts", onehot.sum(0).clamp(min=1))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B,C,H,W) log-magnitude -> (B, n_bands) mean over C and pixels."""
        b, c, h, w = spec.shape
        s = spec.mean(1).reshape(b, -1)                              # (B,HW)
        idx = self.band.flatten().unsqueeze(0).expand(b, -1)         # (B,HW)
        out = torch.zeros(b, self.B, device=spec.device, dtype=s.dtype)
        out.scatter_add_(1, idx, s)
        return out / self.counts.unsqueeze(0)

    def band_mask(self, band_ids) -> torch.Tensor:
        """(H,W) bool mask selecting the given band ids."""
        m = torch.zeros_like(self.band, dtype=torch.bool)
        for b in band_ids:
            m |= (self.band == b)
        return m


# ---------------------------------------------------------- reliance summaries
def reliance_entropy(w: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy retained for the reported exploratory ablation."""
    a = np.abs(np.asarray(w, dtype=np.float64))
    s = a.sum()
    if s <= eps:
        return 0.0
    p = a / s
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def predicted_cost(w: np.ndarray, margin: float = 1.0, kappa: float = 1.0) -> float:
    """Local quadratic score-moving cost from Observation 1."""
    a = np.asarray(w, dtype=np.float64)
    d = (a ** 2 / kappa).sum()
    return float(margin ** 2 / d) if d > 0 else float("inf")
