"""Attack suite A1-A4 (see PLAN.md 3.5).

A1  Wesselkamp-style spectral fingerprint removal   (SEEN in training)
A2  Unseen removal family: radial matching, whitening, resample re-encode
A3  Adaptive white-box PGD on the fused model
A4  Benign degradation: JPEG, resize, plus blur

Every attack takes images in [0,1] (B,3,H,W).  Paper tables report PSNR on the
modified images.
"""
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ------------------------------------------------------------------ reference
class SpectralReference:
    """Mean log-magnitude spectra of a real and a fake reference set.

    Estimated once, on a held-out reference split, with NO access to the
    detector -- this is what makes A1/A2 detector-agnostic, as in the threat
    model of Wesselkamp et al. (2022).
    """

    def __init__(self, real_mean: torch.Tensor, fake_mean: torch.Tensor):
        self.real = real_mean       # (H,W) log magnitude, fftshifted
        self.fake = fake_mean
        self.diff = fake_mean - real_mean

    @staticmethod
    def estimate(loader, device="cuda", max_batches=40):
        rs = fs = None
        rn = fn = 0
        for i, (x, y, *_ ) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            sp = torch.fft.fftshift(torch.fft.fft2(x.mean(1)), dim=(-2, -1)).abs()
            sp = torch.log(sp + 1e-8)
            for lab, acc in ((0, "r"), (1, "f")):
                m = (y.to(device) == lab)
                if m.any():
                    s = sp[m].sum(0)
                    if acc == "r":
                        rs = s if rs is None else rs + s; rn += int(m.sum())
                    else:
                        fs = s if fs is None else fs + s; fn += int(m.sum())
        assert rn > 0 and fn > 0, "reference split must contain both classes"
        return SpectralReference(rs / rn, fs / fn)

    def peak_bands(self, topk: int = 1024) -> torch.Tensor:
        """Boolean mask of the topk frequencies by |fake - real| log-magnitude.

        This is the 'fingerprint' the attack strips: the frequencies where the
        generator's spectrum most exceeds the real-image spectrum.
        """
        d = self.diff.flatten()
        idx = torch.topk(d.abs(), topk).indices
        m = torch.zeros_like(d, dtype=torch.bool)
        m[idx] = True
        return m.view_as(self.diff)

    def save(self, path):
        torch.save({"real": self.real.cpu(), "fake": self.fake.cpu()}, path)

    @staticmethod
    def load(path, device="cuda"):
        d = torch.load(path, map_location=device)
        return SpectralReference(d["real"].to(device), d["fake"].to(device))


# ------------------------------------------------------------------ A1 removal
def a1_fingerprint_removal(x: torch.Tensor, ref: SpectralReference,
                           strength: float = 1.0, topk: int = 1024) -> torch.Tensor:
    """Wesselkamp-style removal: at the fingerprint frequencies, pull the
    magnitude towards the real-image reference, keeping the phase.

    strength s in [0,1]: mag' = mag * (1-s) + target * s (in log domain).
    Detector-agnostic and generator-agnostic: needs only the two reference means.
    """
    if strength <= 0:
        return x
    mask = ref.peak_bands(topk).to(x.device)                      # (H,W)
    Fx = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    mag, phase = Fx.abs().clamp(min=1e-8), torch.angle(Fx)
    logm = torch.log(mag)
    # target log-magnitude = own magnitude shifted by the real-fake difference
    target = logm - ref.diff.unsqueeze(0).unsqueeze(0)
    newlog = torch.where(mask, logm * (1 - strength) + target * strength, logm)
    Fx2 = torch.exp(newlog) * torch.exp(1j * phase)
    out = torch.fft.ifft2(torch.fft.ifftshift(Fx2, dim=(-2, -1))).real
    return out.clamp(0, 1)


# ------------------------------------------------------------------ A2 unseen
def a2_radial_match(x: torch.Tensor, ref: SpectralReference, strength: float = 1.0,
                    n_r: int = 64) -> torch.Tensor:
    """Match the azimuthally-averaged log power profile to the real reference.

    This family edits every frequency through its radial band.  A1 edits a
    sparse peak set.
    """
    if strength <= 0:
        return x
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device) - h // 2,
                            torch.arange(w, device=x.device) - w // 2, indexing="ij")
    rad = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
    ridx = (rad / (rad.max() + 1e-8) * n_r).long().clamp(0, n_r - 1)
    Fx = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    logm, phase = torch.log(Fx.abs().clamp(min=1e-8)), torch.angle(Fx)
    delta = torch.zeros(n_r, device=x.device)
    for b in range(n_r):
        m = (ridx == b)
        if m.any():
            delta[b] = (ref.fake[m].mean() - ref.real[m].mean())
    shift = delta[ridx]                                            # (H,W)
    newlog = logm - strength * shift.unsqueeze(0).unsqueeze(0)
    Fx2 = torch.exp(newlog) * torch.exp(1j * phase)
    return torch.fft.ifft2(torch.fft.ifftshift(Fx2, dim=(-2, -1))).real.clamp(0, 1)


def a2_whiten(x: torch.Tensor, ref: SpectralReference, strength: float = 1.0) -> torch.Tensor:
    """Full-spectrum whitening towards the real reference (dense, not sparse)."""
    if strength <= 0:
        return x
    Fx = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    logm, phase = torch.log(Fx.abs().clamp(min=1e-8)), torch.angle(Fx)
    newlog = logm - strength * ref.diff.unsqueeze(0).unsqueeze(0)
    Fx2 = torch.exp(newlog) * torch.exp(1j * phase)
    return torch.fft.ifft2(torch.fft.ifftshift(Fx2, dim=(-2, -1))).real.clamp(0, 1)


def a2_resample(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Downsample-upsample re-encode: destroys upsampling fingerprints without
    any spectral model at all. strength maps to the downscale factor."""
    if strength <= 0:
        return x
    f = 1.0 - 0.5 * float(strength)                                # 1.0 -> 0.5
    h, w = x.shape[-2:]
    small = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False,
                          antialias=True)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


# ------------------------------------------------------------------ A4 benign
def _pil_roundtrip(x: torch.Tensor, fmt: str, quality: int) -> torch.Tensor:
    out = []
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt, quality=quality)
        buf.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB"))).permute(2, 0, 1))
    return (torch.stack(out).float() / 255.0).to(x.device)


def a4_jpeg(x, quality=70):
    return _pil_roundtrip(x, "JPEG", quality)


def a4_webp(x, quality=70):
    return _pil_roundtrip(x, "WEBP", quality)


def a4_resize(x, factor=0.5):
    h, w = x.shape[-2:]
    s = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def a4_blur(x, sigma=1.0):
    k = int(2 * round(3 * sigma) + 1)
    g = torch.arange(k, device=x.device).float() - k // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    xp = F.pad(x, (k // 2,) * 4, mode="reflect")
    xp = F.conv2d(xp, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(xp, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3).clamp(0, 1)


# ------------------------------------------------------------------ A3 adaptive
def a3_pgd(model, x, y_target, eps=4 / 255, steps=100, alpha=None, gate_aware=False):
    """White-box PGD against the *fused* model. The attacker knows the defence.

    y_target = 0 (real): the attacker wants fakes classified real.
    gate_aware: optional diagnostic that also minimizes the semantic gate weight.
    Reported detector attacks use the fused classification loss alone.
    """
    alpha = alpha or eps / 8
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(x_adv, return_parts=True)
        loss = F.binary_cross_entropy_with_logits(
            out["logit"], torch.full_like(out["logit"], float(y_target)))
        if gate_aware:
            loss = loss + 0.5 * out["gate"].mean()
        g = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * g.sign()
            x_adv = x + (x_adv - x).clamp(-eps, eps)
            x_adv = x_adv.clamp(0, 1)
        x_adv = x_adv.detach()
    return x_adv


# ------------------------------------------------------------------ budgets
def psnr(x, x2):
    mse = ((x - x2) ** 2).flatten(1).mean(1).clamp(min=1e-12)
    return (10 * torch.log10(1.0 / mse)).cpu().numpy()


class LPIPSBudget:
    def __init__(self, device="cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x, x2):
        return self.net(x * 2 - 1, x2 * 2 - 1).flatten().cpu().numpy()


ATTACKS = {
    "a1_removal": ("A1 spectral fingerprint removal", "seen"),
    "a2_radial": ("A2 radial profile matching", "unseen"),
    "a2_whiten": ("A2 spectral whitening", "unseen"),
    "a2_resample": ("A2 resample re-encode", "unseen"),
    "a3_pgd": ("A3 adaptive white-box PGD", "adaptive"),
    "a3_pgd_gate": ("A3 gate-aware white-box PGD", "adaptive"),
    "a4_jpeg": ("A4 JPEG", "benign"),
    "a4_webp": ("A4 WebP", "benign"),
    "a4_resize": ("A4 resize", "benign"),
    "a4_blur": ("A4 blur", "benign"),
}
