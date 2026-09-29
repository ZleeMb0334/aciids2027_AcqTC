"""Fast end-to-end smoke test: data -> model -> losses -> attacks -> metrics."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from models import FusionDetector, count_trainable
from losses import focal_bce, paired_supcon, removal_consistency
from attacks import SpectralReference, a1_fingerprint_removal, a2_radial_match, \
    a2_resample, a4_jpeg, psnr
from spectral import reliance_entropy
from torch.utils.data import DataLoader

dev = "cuda"
t0 = time.time()
idx = D.build_index("test", cache=False)
recs = idx["records"]
gens = sorted({r["generator"] for r in recs})
print(f"[smoke] index: {len(recs)} rows, generators={gens}")
for g in gens:
    n0 = sum(1 for r in recs if r["generator"] == g and r["label"] == 0)
    n1 = sum(1 for r in recs if r["generator"] == g and r["label"] == 1)
    print(f"   {g:24s} real={n0:6d} fake={n1:6d}")

sub = list(np.random.default_rng(0).choice(len(recs), size=min(512, len(recs)), replace=False))
ds = D.GenImageDS(idx, [int(i) for i in sub], train=False)
dl = DataLoader(ds, batch_size=32, num_workers=4)
x, y, g, _ = next(iter(dl))
print(f"[smoke] batch {tuple(x.shape)} range=({x.min():.3f},{x.max():.3f}) labels={y.tolist()[:8]}")

m = FusionDetector(semantic_kw={"tune": "ln"}, noise_kw={"residual": "srm"}).to(dev)
tp, ap, pct = count_trainable(m)
print(f"[smoke] trainable {tp:,}/{ap:,} = {pct:.3f}%")
x = x.to(dev); yf = y.float().to(dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    p = m(x, return_parts=True)
print(f"[smoke] logit {tuple(p['logit'].shape)} gate_mean={p['gate'].mean():.3f}")
lc = focal_bce(p["logit"], yf); lm = paired_supcon(p["z"].float(), yf)
print(f"[smoke] focal={float(lc):.4f} supcon={float(lm):.4f}")

ref = SpectralReference.estimate(dl, dev, max_batches=8)
print(f"[smoke] ref spectra {tuple(ref.real.shape)} diff|max|={ref.diff.abs().max():.3f}")
xf = x[:8]
for name, fn in (("a1", lambda z: a1_fingerprint_removal(z, ref, 1.0)),
                 ("a2_radial", lambda z: a2_radial_match(z, ref, 1.0)),
                 ("a2_resample", lambda z: a2_resample(z, 1.0)),
                 ("a4_jpeg", lambda z: a4_jpeg(z, 70))):
    xa = fn(xf)
    print(f"[smoke] {name:12s} psnr={np.mean(psnr(xf, xa)):.2f} dB  "
          f"linf={float((xa-xf).abs().max()):.4f}")

lr_, dr = removal_consistency(m.noi, xf, a1_fingerprint_removal(xf, ref, 1.0))
print(f"[smoke] robust={float(lr_):.4f} drift={float(dr):.4f}")
w = m.noi.phi_lin.weight.detach().cpu().numpy().ravel()
print(f"[smoke] phi dim={w.size} H(w)={reliance_entropy(w):.3f} (max={np.log(w.size):.3f})")
print(f"[smoke] OK in {time.time()-t0:.0f}s")
