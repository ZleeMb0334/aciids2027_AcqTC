"""GATE 1 (PLAN.md 8): does A1 fingerprint removal actually break a frequency
detector, and does it work on diffusion generators as well as on GANs?

Per generator: train a frequency-only detector on half its data, evaluate on the
other half clean, then under A1 at several strengths. Reports AUROC drop against
the perceptual budget (PSNR), so 'robustness' cannot be bought with visible damage.
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from models import NoiseBranch
from attacks import SpectralReference, a1_fingerprint_removal, a2_radial_match, \
    a2_resample, a4_jpeg, psnr
from spectral import reliance_entropy
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--n", type=int, default=4000, help="images per generator")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--acq", default="matched", choices=["raw", "matched", "history"])
ap.add_argument("--quality", type=int, default=95)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--out", default="/workspace/df-fusion/results")
a = ap.parse_args()
dev = "cuda"
idx = D.build_index("test", cache=False)
recs = idx["records"]
gens = sorted({r["generator"] for r in recs})
rows = []
os.makedirs(a.out, exist_ok=True)

for g in gens:
    sel = [i for i, r in enumerate(recs) if r["generator"] == g]
    lab = np.array([recs[i]["label"] for i in sel])
    if len(np.unique(lab)) < 2 or min((lab == 0).sum(), (lab == 1).sum()) < 500:
        print(f"[gate1] skip {g}: not enough of both classes", flush=True); continue
    rng = np.random.default_rng(0)
    tr, te = [], []
    for l in (0, 1):
        s = [sel[j] for j in np.where(lab == l)[0]]
        s = list(rng.permutation(s))[: a.n // 2]
        k = len(s) // 2
        tr += s[:k]; te += s[k:]
    qpool = D.sample_qtable_pool(idx, sel, n=256, seed=0) if a.acq == "history" else None
    if a.acq == "history":
        print(f"[gate1] {g}: qtable pool = {len(qpool)}", flush=True)
    kw = dict(acq=a.acq, quality=a.quality, size=a.size, qpool=qpool)
    dtr = D.GenImageDS(idx, [int(i) for i in tr], train=True, seed=0, **kw)
    dte = D.GenImageDS(idx, [int(i) for i in te], train=False, seed=0, **kw)
    ltr = DataLoader(dtr, batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True)
    lte = DataLoader(dte, batch_size=a.bs, shuffle=False, num_workers=8)

    m = NoiseBranch(residual="srm", size=a.size).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    for ep in range(a.epochs):
        m.train()
        for x, y, _, _ in ltr:
            x, y = x.to(dev), y.float().to(dev)
            _, lg = m(x)
            loss = F.binary_cross_entropy_with_logits(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    ref = SpectralReference.estimate(ltr, dev, max_batches=25)
    ref.save(f"{a.out}/gate1_ref__{a.acq}__{g}.pt")

    # measured reliance weights of the linear band path
    w = m.phi_lin.weight.detach().cpu().numpy().ravel()
    H = reliance_entropy(w)

    def evaluate(fn=None):
        ys, ss, ps = [], [], []
        with torch.no_grad():
            for x, y, _, _ in lte:
                x = x.to(dev)
                if fn is not None:
                    xa = fn(x)
                    ps.append(psnr(x, xa)); x = xa
                _, lg = m(x)
                ys.append(y.numpy()); ss.append(lg.float().cpu().numpy())
        y, s = np.concatenate(ys), np.concatenate(ss)
        return roc_auc_score(y, s), (float(np.mean(np.concatenate(ps))) if ps else float("inf"))

    clean, _ = evaluate()
    rec = {"generator": g, "family": "GAN" if g in D.GAN_GENS else "diffusion",
           "acq": a.acq, "quality": a.quality, "size": a.size,
           "n_train": len(tr), "n_test": len(te), "auroc_clean": clean,
           "H_w": H, "H_max": float(np.log(len(w)))}
    for s in (0.25, 0.5, 0.75, 1.0):
        au, pp = evaluate(lambda z, s=s: a1_fingerprint_removal(z, ref, s))
        rec[f"a1_s{s}_auroc"] = au; rec[f"a1_s{s}_psnr"] = pp
    for nm, fn in (("a2_radial", lambda z: a2_radial_match(z, ref, 1.0)),
                   ("a2_resample", lambda z: a2_resample(z, 1.0)),
                   ("a4_jpeg70", lambda z: a4_jpeg(z, 70))):
        au, pp = evaluate(fn)
        rec[f"{nm}_auroc"] = au; rec[f"{nm}_psnr"] = pp
    rows.append(rec)
    print(f"[gate1] {g:24s} fam={rec['family']:9s} clean={clean:.4f} "
          f"a1@1.0={rec['a1_s1.0_auroc']:.4f} (psnr {rec['a1_s1.0_psnr']:.1f}dB) "
          f"drop={clean-rec['a1_s1.0_auroc']:+.4f} H(w)={H:.3f}", flush=True)

with open(f"{a.out}/gate1_{a.acq}.json", "w") as f:
    json.dump(rows, f, indent=1)
print(f"[gate1] wrote gate1_{a.acq}.json", flush=True)
