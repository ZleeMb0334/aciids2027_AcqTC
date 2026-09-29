"""E2 (C1): are cross-generator generalisation and attack robustness the same axis?

Trains each detector family under a chosen acquisition protocol, then measures
BOTH axes on one protocol: LOGO generalisation and robustness to A1/A2/A3/A4,
every attack reported against its achieved perceptual budget.
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from models import FusionDetector, count_trainable
from losses import focal_bce, paired_supcon, removal_consistency
from attacks import (SpectralReference, a1_fingerprint_removal, a2_radial_match,
                     a2_whiten, a2_resample, a3_pgd, a4_jpeg, a4_resize, a4_blur, psnr)
from spectral import reliance_entropy

ap = argparse.ArgumentParser()
ap.add_argument("--acq", default="history", choices=["raw", "matched", "history"])
ap.add_argument("--held-out", default="glide")
ap.add_argument("--methods", default="semantic,noise,gate,gate_robust")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--n-per-gen", type=int, default=3000)
ap.add_argument("--n-test", type=int, default=2000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="/workspace/df-fusion/results")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)

idx = D.load_indices()
sp = D.logo_split(idx, a.held_out, a.n_per_gen, a.n_test, seed=a.seed)
qpool = D.sample_qtable_pool(idx["train"], sp["train"], 256, 0) if a.acq == "history" else None
kw = dict(acq=a.acq, size=a.size, qpool=qpool)
print(f"[orth] acq={a.acq} held_out={a.held_out} train={len(sp['train'])} "
      f"test_in={len(sp['test_in'])} test_out={len(sp['test_out'])} qpool={len(qpool or [])}",
      flush=True)


def loader(which, subset, train):
    src = idx["train"] if which == "train" else idx["test"]
    ds = D.GenImageDS(src, subset, train=train, seed=a.seed, **kw)
    if train:
        samp = D.PairedBatchSampler(ds.recs, a.bs, mode="class", seed=a.seed)
        return DataLoader(ds, batch_sampler=samp, num_workers=12, pin_memory=True)
    return DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=12, pin_memory=True)


ltr = loader("train", sp["train"], True)
lva = loader("train", sp["val"], False)
lin = loader("test", sp["test_in"], False)
lout = loader("test", sp["test_out"], False)

ref = SpectralReference.estimate(lva, dev, max_batches=25)
ref.save(f"{a.out}/ref__{a.acq}__{a.held_out}.pt")
lp = None
rows = []

for method in a.methods.split(","):
    t0 = time.time()
    robust = method.endswith("_robust")
    fusion = method.replace("_robust", "")
    m = FusionDetector(semantic_kw={"tune": "ln"},
                       noise_kw={"residual": "srm", "size": a.size},
                       fusion=fusion).to(dev)
    tp, tot, pct = count_trainable(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4,
                            weight_decay=1e-4)
    for ep in range(a.epochs):
        m.train()
        smax = min(1.0, (ep / max(1, a.epochs - 1)) / 0.4)
        for x, y, g, _ in ltr:
            x, y = x.to(dev, non_blocking=True), y.float().to(dev, non_blocking=True)
            p = m(x, return_parts=True)
            loss = focal_bce(p["logit"], y) + 0.5 * paired_supcon(p["z"].float(), y)
            if robust and (y > 0.5).any():
                xf = x[y > 0.5]
                s = float(np.random.uniform(0, smax))
                with torch.no_grad():
                    xr = a1_fingerprint_removal(xf, ref, strength=s)
                    xr2 = a2_resample(xf, strength=s)
                lr1, _ = removal_consistency(m.noi, xf, xr)
                lr2, _ = removal_consistency(m.noi, xf, xr2)
                loss = loss + smax * 0.5 * (lr1 + lr2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0)
            opt.step()
    m.eval()

    def run(ld, fn=None, adaptive=False):
        ys, ss, ps = [], [], []
        for x, y, g, _ in ld:
            x = x.to(dev); yb = y.numpy()
            if adaptive:
                fk = torch.tensor(yb == 1, device=dev)
                xa = x.clone()
                if fk.any():
                    xa[fk] = a3_pgd(m, x[fk], 0.0, eps=fn, steps=40,
                                    gate_aware=(fusion == "gate"))
                    ps.append(psnr(x[fk], xa[fk]))
                x = xa
            elif fn is not None:
                with torch.no_grad():
                    xa = fn(x)
                ps.append(psnr(x, xa)); x = xa
            with torch.no_grad():
                ss.append(m(x).float().cpu().numpy())
            ys.append(yb)
        y, s = np.concatenate(ys), np.concatenate(ss)
        return float(roc_auc_score(y, s)), (float(np.mean(np.concatenate(ps))) if ps else float("inf"))

    r = {"method": method, "acq": a.acq, "held_out": a.held_out, "seed": a.seed,
         "trainable": tp, "trainable_pct": pct, "train_s": time.time() - t0}
    r["auroc_in"], _ = run(lin)
    r["auroc_out"], _ = run(lout)
    r["gap"] = r["auroc_in"] - r["auroc_out"]
    for nm, fn in (("a1", lambda z: a1_fingerprint_removal(z, ref, 1.0)),
                   ("a2_radial", lambda z: a2_radial_match(z, ref, 1.0)),
                   ("a2_whiten", lambda z: a2_whiten(z, ref, 1.0)),
                   ("a2_resample", lambda z: a2_resample(z, 1.0)),
                   ("a4_jpeg70", lambda z: a4_jpeg(z, 70)),
                   ("a4_resize", lambda z: a4_resize(z, 0.5)),
                   ("a4_blur", lambda z: a4_blur(z, 1.0))):
        r[f"{nm}_auroc"], r[f"{nm}_psnr"] = run(lin, fn)
    for eps in (1 / 255, 2 / 255, 4 / 255, 8 / 255):
        au, pp = run(lin, eps, adaptive=True)
        r[f"a3_eps{eps*255:.0f}_auroc"], r[f"a3_eps{eps*255:.0f}_psnr"] = au, pp
    w = m.noi.phi_lin.weight.detach().cpu().numpy().ravel()
    r["H_w"] = reliance_entropy(w); r["H_max"] = float(np.log(w.size))
    rows.append(r)
    print(f"[orth] {method:14s} in={r['auroc_in']:.4f} out={r['auroc_out']:.4f} "
          f"gap={r['gap']:+.4f} | a1={r['a1_auroc']:.4f} a2res={r['a2_resample_auroc']:.4f} "
          f"jpeg={r['a4_jpeg70_auroc']:.4f} | pgd4={r['a3_eps4_auroc']:.4f} "
          f"(psnr {r['a3_eps4_psnr']:.1f}) H={r['H_w']:.3f} [{r['train_s']:.0f}s]", flush=True)
    torch.save({"model": m.state_dict()}, f"{a.out}/ckpt_orth__{a.acq}__{a.held_out}__{method}.pt")
    with open(f"{a.out}/orthogonality__{a.acq}__{a.held_out}__s{a.seed}.json", "w") as f:
        json.dump(rows, f, indent=1)
print("[orth] done", flush=True)
