"""Dual-branch detector: LN-tuned semantic branch plus spectral branch.

The trainable footprint is measured by count_trainable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral import ResidualExtractor, BandProfile, log_spectrum

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ------------------------------------------------------------- semantic branch
class SemanticBranch(nn.Module):
    """Frozen vision encoder, LayerNorm affine parameters only trainable."""

    def __init__(self, arch="ViT-B-16-quickgelu", pretrained="openai", tune="ln", proj_dim=256):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = model.visual
        del model
        for p in self.visual.parameters():
            p.requires_grad_(False)
        if tune == "ln":
            for m in self.visual.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad_(True)
        elif tune == "full":
            for p in self.visual.parameters():
                p.requires_grad_(True)
        # tune == "probe": backbone stays frozen entirely
        d = getattr(self.visual, "output_dim", None) or self.visual.proj.shape[1]
        self.proj = nn.Linear(d, proj_dim)
        self.head = nn.Linear(proj_dim, 1)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[-1] != 224:
            # class-independent resize: applied to every input regardless of
            # label, so it cannot reintroduce an acquisition shortcut.
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        f = self.visual((x - self.mean) / self.std)
        z = F.normalize(self.proj(f), dim=-1)     # hypersphere
        return z, self.head(z).squeeze(-1)


# ---------------------------------------------------------------- noise branch
class NoiseBranch(nn.Module):
    """Residual -> log-spectrum CNN + band-profile MLP.

    The band-profile path stays linear in phi.  Its weights provide a
    self-check for the post-hoc band-ablation measurement.
    """

    def __init__(self, residual="srm", size=128, n_r=32, n_a=2, emb=256):
        super().__init__()
        self.res = ResidualExtractor(residual)
        self.bp = BandProfile(size=size, n_r=n_r, n_a=n_a)
        cin = 9 if residual == "srm" else 3
        self.cnn = nn.Sequential(
            nn.Conv2d(cin, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.phi_lin = nn.Linear(self.bp.B, 1, bias=False)     # <- w of Prop. 1
        self.emb = nn.Linear(128 + self.bp.B, emb)
        self.head = nn.Linear(emb, 1)

    def band_profile(self, x):
        return self.bp(log_spectrum(self.res(x)))

    def forward(self, x, return_phi=False, phi_grad=False):
        r = self.res(x)
        sp = log_spectrum(r)
        if sp.shape[-1] != self.bp.band.shape[-1]:
            raise ValueError(f"spectrum {tuple(sp.shape[-2:])} != band grid "
                             f"{tuple(self.bp.band.shape)}; set NoiseBranch(size=...)")
        phi = self.bp(sp)
        if phi_grad:
            phi = phi.detach().requires_grad_(True)
            self._phi = phi
        c = self.cnn(sp)
        h = self.emb(torch.cat([c, phi], dim=1))
        logit = self.head(h).squeeze(-1) + self.phi_lin(phi).squeeze(-1)
        return (h, logit, phi) if return_phi else (h, logit)

    def reliance(self, x=None, mode="weight"):
        """Reliance profile for the explicit linear band path.

        mode='weight' (default) returns the linear band path's weights, which are
        exactly w for that path: bounded, cheap, plus differentiable without a
        second backward pass. This is what the regulariser acts on.

        mode='grad' returns E_x |d c_n / d phi_b| over the whole branch. It is the
        more complete quantity.  It requires double backpropagation through a
        re-attached phi, which we found destabilises training (reported as an
        ablation); the whole-branch reliance is instead MEASURED post hoc by band
        ablation in reliance.py, which needs no gradients at all.
        """
        if mode == "weight" or x is None:
            return self.phi_lin.weight.squeeze(0)
        _, logit, phi = self.forward(x, return_phi=True, phi_grad=True)
        g = torch.autograd.grad(logit.sum(), phi, create_graph=True)[0]
        return g.abs().mean(0)


# ----------------------------------------------------------------- the fusion
class FusionDetector(nn.Module):
    def __init__(self, semantic_kw=None, noise_kw=None, fusion="gate"):
        super().__init__()
        self.sem = SemanticBranch(**(semantic_kw or {}))
        self.noi = NoiseBranch(**(noise_kw or {}))
        self.fusion = fusion
        ds = self.sem.proj.out_features
        dn = self.noi.emb.out_features
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(ds + dn, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 1))
        elif fusion == "concat":
            self.fc = nn.Sequential(nn.Linear(ds + dn, 128), nn.ReLU(inplace=True),
                                    nn.Linear(128, 1))
        elif fusion == "static":
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_parts=False):
        z, ls = self.sem(x)
        if self.fusion == "semantic":
            # Semantic-only variants share this training harness with fusion
            # models. Bypass the inactive spectral module so their measured
            # training and deployment paths match the paper's one-pass model.
            a = torch.ones_like(ls)
            if return_parts:
                h = torch.zeros((len(x), self.noi.emb.out_features),
                                device=x.device, dtype=z.dtype)
                return {"logit": ls, "gate": a, "sem_logit": ls,
                        "noise_logit": torch.zeros_like(ls), "z": z, "h": h}
            return ls
        h, ln = self.noi(x)
        if self.fusion == "gate":
            a = torch.sigmoid(self.gate(torch.cat([z, h], dim=1))).squeeze(-1)
            logit = a * ls + (1 - a) * ln
        elif self.fusion == "concat":
            a = torch.full_like(ls, 0.5)
            logit = self.fc(torch.cat([z, h], dim=1)).squeeze(-1)
        elif self.fusion == "static":
            a = torch.sigmoid(self.alpha).expand_as(ls)
            logit = a * ls + (1 - a) * ln
        elif self.fusion == "mean":
            a = torch.full_like(ls, 0.5)
            logit = 0.5 * (ls + ln)
        elif self.fusion == "noise":
            a = torch.zeros_like(ls); logit = ln
        else:
            raise ValueError(self.fusion)
        if return_parts:
            return {"logit": logit, "gate": a, "sem_logit": ls, "noise_logit": ln,
                    "z": z, "h": h}
        return logit


def count_trainable(m):
    # The semantic deployment path bypasses and can discard the diagnostic
    # spectral branch. Count the parameters that participate in that path.
    parameters = m.sem.parameters() if m.fusion == "semantic" else m.parameters()
    parameters = list(parameters)
    t = sum(p.numel() for p in parameters if p.requires_grad)
    a = sum(p.numel() for p in parameters)
    return t, a, 100.0 * t / a


def trainable_breakdown(m):
    """Decompose the trainable budget so the efficiency claim is auditable:
    LayerNorm affine parameters of the frozen backbone vs everything added."""
    import torch.nn as _nn
    ln = 0
    for mod in m.sem.visual.modules():
        if isinstance(mod, _nn.LayerNorm):
            ln += sum(p.numel() for p in mod.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in m.sem.visual.parameters())
    sem_head = sum(p.numel() for p in list(m.sem.proj.parameters()) + list(m.sem.head.parameters()))
    noise = sum(p.numel() for p in m.noi.parameters() if p.requires_grad)
    fuse = sum(p.numel() for n, p in m.named_parameters()
               if p.requires_grad and (n.startswith("gate") or n.startswith("fc") or n == "alpha"))
    return {"ln_affine": ln, "backbone_total": backbone,
            "ln_pct_of_backbone": 100.0 * ln / backbone,
            "sem_head": sem_head, "noise_branch": noise, "fusion": fuse}
